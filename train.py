# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import torch.nn.functional as F
import transformers
from transformers.trainer_utils import set_seed, enable_full_determinism
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm
from math import ceil
import math
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
import wandb
from transformers import TrainerCallback
from test import evaluation
import test as _test_module
import shutil

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)
from src.cot_utils import (
    extract_superpose_steps,
    prepare_superpose_step_text_for_tokenization,
    split_natural_cot_sentences,
    split_reference_and_superpose_steps,
)

IGNORE_INDEX = -100

def _normalize_dataset_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower()


def _is_icot_full_dataset(name: Optional[str]) -> bool:
    normalized = _normalize_dataset_name(name)
    return "icot" in normalized and "full" in normalized


def _is_whynlp_gsm8k_aug_nl(name: Optional[str]) -> bool:
    normalized = _normalize_dataset_name(name).replace("_", "-")
    return "whynlp" in normalized and "gsm8k-aug-nl" in normalized


def _uses_natural_language_cot_steps(name: Optional[str]) -> bool:
    normalized = _normalize_dataset_name(name)
    return (
        _is_icot_full_dataset(normalized)
        or _is_whynlp_gsm8k_aug_nl(normalized)
        or "commonsense" in normalized
        or "strategy" in normalized
    )


def _reports_to_wandb(report_to) -> bool:
    if isinstance(report_to, (list, tuple, set)):
        return ("wandb" in report_to) or ("all" in report_to)
    if isinstance(report_to, str):
        return report_to in ("wandb", "all")
    return False


def _is_primary_process(training_args) -> bool:
    return int(getattr(training_args, "process_index", 0)) == 0


def _extract_nl_cot_steps(example) -> Sequence[str]:
    raw_steps = example.get("steps") if isinstance(example, dict) else None
    if isinstance(raw_steps, (list, tuple)):
        return [str(step).strip() for step in raw_steps if str(step).strip()]
    return split_natural_cot_sentences(str(example.get("cot", "")))


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_manual_log_step = None

    def _compute_total_update_steps(self):
        if hasattr(self, "_cached_total_update_steps"):
            return self._cached_total_update_steps
        if self.args.max_steps and self.args.max_steps > 0:
            total = self.args.max_steps
        else:
            grad_accum = max(1, self.args.gradient_accumulation_steps)
            per_device = self.args.per_device_train_batch_size
            world_size = max(1, self.args.world_size)
            dataset_size = len(self.train_dataset)
            updates_per_epoch = math.ceil(dataset_size / (per_device * world_size * grad_accum))
            total = math.ceil(updates_per_epoch * self.args.num_train_epochs)
        self._cached_total_update_steps = int(total)
        return self._cached_total_update_steps

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step

        total_update_steps = self._compute_total_update_steps()
        step_ratio = step / total_update_steps if total_update_steps > 0 else 0.0

        inputs["step_ratio"] = step_ratio
        inputs["step"] = step
        inputs = {k: v for k, v in inputs.items() if k != "answer_values"}
        outputs = model(**inputs)
        loss = outputs["loss"]

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Non-finite loss at step={step}: {loss.detach()}")

        should_log_this_step = (
            step % self.args.logging_steps == 0
            and getattr(self.accelerator, "sync_gradients", True)
            and step != self._last_manual_log_step
        )
        if should_log_this_step:
            logs = {
                "loss": float(loss.detach()),
                "ce_loss": float(outputs["ce_loss"]),
                "distill_loss": float(outputs["distill_loss"]),
                "ref_ce_loss": float(outputs["ref_ce_loss"]),
            }
            if "ce_loss_weighted" in outputs:
                logs["ce_loss_weighted"] = float(outputs["ce_loss_weighted"])
            if "ce_curriculum_weight" in outputs:
                logs["ce_curriculum_weight"] = float(outputs["ce_curriculum_weight"])
            if "kl_loss" in outputs:
                logs["kl_loss"] = float(outputs["kl_loss"])
            if "superpose_kl_raw" in outputs:
                logs["superpose_kl_raw"] = float(outputs["superpose_kl_raw"])
            if "superpose_kl_weighted_raw" in outputs:
                logs["superpose_kl_weighted_raw"] = float(outputs["superpose_kl_weighted_raw"])
            if "superpose_kl_weight" in outputs:
                logs["superpose_kl_weight"] = float(outputs["superpose_kl_weight"])
            if "superpose_kl_bag_raw" in outputs:
                logs["superpose_kl_bag_raw"] = float(outputs["superpose_kl_bag_raw"])
            if "superpose_kl_scalar_positional_raw" in outputs:
                logs["superpose_kl_scalar_positional_raw"] = float(outputs["superpose_kl_scalar_positional_raw"])
            if "superpose_kl_scalar_positional_loss" in outputs:
                logs["superpose_kl_scalar_positional_loss"] = float(outputs["superpose_kl_scalar_positional_loss"])
            if "superpose_aux_raw" in outputs:
                logs["superpose_aux_raw"] = float(outputs["superpose_aux_raw"])
            if "superpose_aux_loss" in outputs:
                logs["superpose_aux_loss"] = float(outputs["superpose_aux_loss"])
            if "superpose_l2_loss" in outputs:
                logs["superpose_l2_loss"] = float(outputs["superpose_l2_loss"])
            if "superpose_l2_raw" in outputs:
                logs["superpose_l2_raw"] = float(outputs["superpose_l2_raw"])
            if "superpose_l2_weighted_raw" in outputs:
                logs["superpose_l2_weighted_raw"] = float(outputs["superpose_l2_weighted_raw"])
            if "superpose_l2_weight" in outputs:
                logs["superpose_l2_weight"] = float(outputs["superpose_l2_weight"])
            if "superpose_target_slot_variance" in outputs:
                logs["superpose_target_slot_variance"] = float(outputs["superpose_target_slot_variance"])
            if "var_loss" in outputs:
                logs["var_loss"] = float(outputs["var_loss"])
            if "latent_diversity" in outputs:
                logs["latent_diversity"] = float(outputs["latent_diversity"])
            if "diversity_loss" in outputs:
                logs["diversity_loss"] = float(outputs["diversity_loss"])
            logs["global_update_step"] = int(step)

            bad = {k: v for k, v in logs.items() if not math.isfinite(v)}
            if bad:
                raise RuntimeError(f"Non-finite logged metrics at step={step}: {bad}")

            self.log(logs)
            self._last_manual_log_step = step
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is None:
            return

        cleaned_logs = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                if not math.isfinite(float(v)):
                    continue
                cleaned_logs[k] = float(v)
            elif torch.is_tensor(v):
                vv = float(v.detach().float().cpu())
                if not math.isfinite(vv):
                    continue
                cleaned_logs[k] = vv
            else:
                cleaned_logs[k] = v

        if not cleaned_logs:
            return

        super().log(cleaned_logs)

        has_builtin_wandb = _reports_to_wandb(self.args.report_to)

        if wandb.run is not None and self.is_world_process_zero() and not has_builtin_wandb:
            wandb.log(cleaned_logs, step=self.state.global_step)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\\d+\\.?\\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (training_args.reasoning_mode or "codi").lower() != "codi":
        raise ValueError("Only reasoning_mode='codi' is supported.")

    if training_args.seed is not None:
        set_seed(training_args.seed)

    if _is_primary_process(training_args) and "wandb" in training_args.report_to and wandb.run is None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mux"),
            name=training_args.expt_name,
            config=training_args
        )
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )


    model = CODI(model_args, training_args, lora_config)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.pad_token_id

    def load_local_json_with_cleanup(path: str):
        """
        Load a JSON file and tolerate stray single-character lines that break parsing.
        """
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[warn] JSON decode failed for {path} ({e}). Retrying after removing stray lines.")
            with open(path) as f:
                lines = f.readlines()
            cleaned = [ln for ln in lines if ln.strip() not in {"z", "Z"}]
            return json.loads("".join(cleaned))

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            raise ValueError("Could not find answer prompt position in token sequence.")

    def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        bot_id: int,
        eot_id: int,
        superpose_targets: Optional[Sequence[object]] = None,
        superpose_natural_language_steps: bool = False,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")

        def _normalize_cot_text(cot_text: str) -> str:
            cot_text = str(cot_text) if cot_text is not None else ""
            if not cot_text:
                return ""
            return cot_text if cot_text[-1].isspace() else cot_text + " "

        def _normalize_superpose_target(cot_target):
            if isinstance(cot_target, (list, tuple)):
                return [str(step).strip() for step in cot_target if str(step).strip()]
            return _normalize_cot_text(cot_target)

        superpose_strip_markers = bool(getattr(training_args, "ccot_superpose_strip_markers", False))

        def _tokenize_superpose_step(
            step_text: str,
            *,
            is_first_step: bool,
            preserve_boundary_whitespace: bool,
            strip_markers_override: Optional[bool] = None,
        ):
            step_text = prepare_superpose_step_text_for_tokenization(
                step_text,
                is_first_step=is_first_step,
                strip_markers=(
                    superpose_strip_markers
                    if strip_markers_override is None
                    else bool(strip_markers_override)
                ),
                preserve_boundary_whitespace=preserve_boundary_whitespace,
            )
            if not step_text:
                return torch.empty(0, dtype=torch.long)
            ids = tokenizer.encode(step_text, add_special_tokens=False)
            return torch.tensor(ids, dtype=torch.long)

        def _build_reference_superpose_cot(cot_target):
            steps = extract_superpose_steps(
                cot_target,
                natural_language_steps=superpose_natural_language_steps,
            )
            prepared_steps = []
            token_lens = []
            for step_idx, step in enumerate(steps):
                step_text = prepare_superpose_step_text_for_tokenization(
                    step,
                    is_first_step=(step_idx == 0),
                    strip_markers=superpose_strip_markers,
                    preserve_boundary_whitespace=True,
                )
                if not step_text:
                    continue
                prepared_steps.append(step_text)
                token_lens.append(len(tokenizer.encode(step_text, add_special_tokens=False)))
            cot_text = "".join(prepared_steps)
            if cot_text and not cot_text[-1].isspace():
                cot_text = cot_text + " "
            return len(token_lens), token_lens, cot_text

        normalized_targets = [_normalize_cot_text(t) for t in targets]
        if superpose_targets is None:
            normalized_superpose_targets = normalized_targets
        else:
            if len(superpose_targets) != len(targets):
                raise ValueError(
                    "superpose_targets must have the same length as targets: "
                    f"{len(superpose_targets)} != {len(targets)}"
                )
            normalized_superpose_targets = [_normalize_superpose_target(t) for t in superpose_targets]
        cot_step_counts = []
        cot_step_token_counts = []
        normalized_ref_superpose_targets = []
        for cot in normalized_targets:
            step_count, token_lens, ref_superpose_cot = _build_reference_superpose_cot(cot)
            cot_step_counts.append(step_count)
            cot_step_token_counts.append(token_lens)
            normalized_ref_superpose_targets.append(ref_superpose_cot)
        superpose_mode = bool(
            getattr(training_args, "ccot_superpose_kl", False)
            or getattr(training_args, "ccot_superpose_embed_l2", False)
        )
        superpose_kl_context_free_step_bag = bool(
            getattr(training_args, "ccot_superpose_kl", False)
            and
            getattr(training_args, "ccot_superpose_kl_context_free_step_bag", False)
        )
        cot_superpose_step_token_ids = None
        cot_superpose_kl_step_token_ids = None
        cot_superpose_step_counts = None
        if superpose_mode:
            cot_superpose_step_token_ids = []
            if superpose_kl_context_free_step_bag:
                cot_superpose_kl_step_token_ids = []
            cot_superpose_step_counts = []
            for cot in normalized_superpose_targets:
                steps = extract_superpose_steps(
                    cot,
                    natural_language_steps=superpose_natural_language_steps,
                )
                tok_steps = [
                    _tokenize_superpose_step(
                        step,
                        is_first_step=(step_idx == 0),
                        preserve_boundary_whitespace=True,
                    )
                    for step_idx, step in enumerate(steps)
                ]
                cot_superpose_step_token_ids.append(tok_steps)
                if superpose_kl_context_free_step_bag:
                    kl_tok_steps = [
                        _tokenize_superpose_step(
                            step,
                            is_first_step=(step_idx == 0),
                            preserve_boundary_whitespace=False,
                        )
                        for step_idx, step in enumerate(steps)
                    ]
                    cot_superpose_kl_step_token_ids.append(kl_tok_steps)
                cot_superpose_step_counts.append(len(tok_steps))
        answer_values = [extract_answer_number(ans) for ans in answers]

        if superpose_mode:
            sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
            cot_id = _tokenize_fn(normalized_targets, tokenizer)["input_ids"]
            ref_superpose_cot_id = _tokenize_fn(normalized_ref_superpose_targets, tokenizer)["input_ids"]
            answers_tok = _tokenize_fn(answers, tokenizer)["input_ids"]

            if not training_args.remove_eos:
                sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
                cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
                ref_superpose_cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in ref_superpose_cot_id]
            answers_tok = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_tok]

            if tokenizer.bos_token_id is not None:
                sources_id = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in sources_id]
                cot_id = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in cot_id]
                ref_superpose_cot_id = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in ref_superpose_cot_id]
                answers_tok = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in answers_tok]

            ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_tok)]
            ref_superpose_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, ref_superpose_cot_id, answers_tok)]
            ref_labels = []
            ref_superpose_labels = []
            for x, y in zip(ref_input_ids, sources_id):
                z = x.clone()
                z[:len(y)] = -100
                ref_labels.append(z)
            for x, y in zip(ref_superpose_input_ids, sources_id):
                z = x.clone()
                z[:len(y)] = -100
                ref_superpose_labels.append(z)

            encoder_input_ids = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
            if training_args.remove_eos:
                decoder_input_ids = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_tok]
            else:
                decoder_input_ids = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_tok]

            answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
            if answer_prompts[0][0] == tokenizer.bos_token_id:
                answer_prompts[0] = answer_prompts[0][1:]
                answer_prompts[1] = answer_prompts[1][1:]

            ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in ref_input_ids]
            model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in decoder_input_ids]
            ref_eos_position = [len(x) - 1 for x in ref_input_ids]
            model_eos_position = [len(x) - 1 for x in decoder_input_ids]

            data = dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=decoder_input_ids,
                ref_answer_position=ref_answer_position,
                model_answer_position=model_answer_position,
                ref_eos_position=ref_eos_position,
                model_eos_position=model_eos_position,
                ref_labels=ref_labels,
                cot_step_counts=cot_step_counts,
                cot_step_token_counts=cot_step_token_counts,
                ref_superpose_input_ids=ref_superpose_input_ids,
                ref_superpose_labels=ref_superpose_labels,
                cot_superpose_step_token_ids=cot_superpose_step_token_ids,
                cot_superpose_step_counts=cot_superpose_step_counts,
                answer_values=answer_values,
            )
            if cot_superpose_kl_step_token_ids is not None:
                data["cot_superpose_kl_step_token_ids"] = cot_superpose_kl_step_token_ids
            return data

        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(normalized_targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if tokenizer.bos_token_id is not None:
            cot_id = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in cot_id]
            answers_id = [x[1:] if len(x) > 0 and int(x[0]) == tokenizer.bos_token_id else x for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)

        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id:
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]

        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id,
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position,
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels,
                            cot_step_counts=cot_step_counts, cot_step_token_counts=cot_step_token_counts,
                                answer_values=answer_values)


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")

            self.data_name = data_name
            questions, cots, answers = [], [], []
            superpose_cots = []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            superpose_mode = bool(
                getattr(training_args, "ccot_superpose_kl", False)
                or getattr(training_args, "ccot_superpose_embed_l2", False)
            )
            superpose_include_last_target_only = bool(
                getattr(training_args, "ccot_superpose_include_last_cot_target_only", False)
            )
            icot_full_filter_non_digit_answers = bool(
                getattr(training_args, "icot_full_filter_non_digit_answers", False)
            )
            skipped_icot_full_non_digit_answers = 0
            for num_iter, example in enumerate(raw_data):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if _is_icot_full_dataset(self.data_name) or _is_whynlp_gsm8k_aug_nl(self.data_name):
                    if example["answer"] is None:
                        continue

                    cot_text = " ".join(_extract_nl_cot_steps(example))
                    token_num = len(tokenizer.encode(example["question"] + cot_text + str(example["answer"])))
                    if token_num > training_args.max_token_num:
                        continue

                    ref_cot_steps, superpose_cot_steps = split_reference_and_superpose_steps(
                        _extract_nl_cot_steps(example),
                        include_last_cot=bool(training_args.include_last_cot),
                        include_last_for_superpose_targets=superpose_include_last_target_only,
                    )

                    answer = str(example["answer"]).split(" ")[-1]
                    if icot_full_filter_non_digit_answers and (not answer or not answer[0].isdigit()):
                        skipped_icot_full_non_digit_answers += 1
                        continue
                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    questions.append(question)

                    cot = " ".join(ref_cot_steps).strip()
                    if cot:
                        cot = cot + "\n"
                    cots.append(cot)
                    superpose_cots.append(superpose_cot_steps)
                    answers.append(answer)
                elif "icot" in self.data_name:
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue

                    cot_list = []
                    ref_cot_steps, superpose_cot_steps = split_reference_and_superpose_steps(
                        f"{example['cot']}".split(" "),
                        include_last_cot=bool(training_args.include_last_cot),
                        include_last_for_superpose_targets=superpose_include_last_target_only,
                    )

                    len_cot = len(ref_cot_steps)
                    for i in range(training_args.num_latent):
                        cot_list.append(" ".join(ref_cot_steps[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]

                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(ref_cot_steps))
                    superpose_cots.append(" ".join(superpose_cot_steps))
                    answers.append(answer)
                elif (
                    data_args.train_path is not None
                    or data_args.eval_path is not None
                    or "openai" in str(self.data_name).lower()
                    or "chain" in str(self.data_name).lower()
                ):
                    question = f"{example['question']}"
                    raw_chains = example.get("chains", [])
                    if not raw_chains:
                        continue

                    chain_mode = getattr(data_args, "chain_mode", "first")
                    chain_k = max(1, int(getattr(data_args, "chain_k", 1)))
                    joiner = getattr(data_args, "chain_joiner", " ")

                    if chain_mode == "first":
                        selected_chains = raw_chains[:1]
                    elif chain_mode == "all":
                        selected_chains = raw_chains
                    elif chain_mode in ["first_k", "k"]:
                        if len(raw_chains) < chain_k:
                            continue
                        selected_chains = raw_chains[:chain_k]
                    else:
                        raise ValueError(f"Unsupported chain_mode: {chain_mode}")

                    answer_txt = str(example["answer"]).strip().replace("####", "")
                    answer = f"The answer is: {answer_txt}"

                    for chain in selected_chains:
                        cot = joiner.join(chain).strip()
                        if cot and not cot.endswith("\n"):
                            cot = cot + "\n"
                        token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                        if token_num > training_args.max_token_num:
                            continue
                        questions.append(question)
                        cots.append(cot)
                        superpose_cots.append(cot)
                        answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    if superpose_mode:
                        cot_steps = split_natural_cot_sentences(example['cot'])
                        ref_cot_steps, superpose_cot_steps = split_reference_and_superpose_steps(
                            cot_steps,
                            include_last_cot=bool(training_args.include_last_cot),
                            include_last_for_superpose_targets=superpose_include_last_target_only,
                        )
                        cot = " ".join(ref_cot_steps).strip()
                        if cot:
                            cot = cot + "\n"
                    else:
                        cot = example['cot'].strip() + "\n"
                        superpose_cot_steps = cot
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    token_num = len(tokenizer.encode(question + " " + str(example['cot']).strip() + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    superpose_cots.append(superpose_cot_steps)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                superpose_cots = superpose_cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]

            no_cot_flag = getattr(training_args, 'no_cot', False)
            no_cot_mask_flag = getattr(training_args, 'no_cot_latent_mask', False)
            if no_cot_flag or no_cot_mask_flag:
                tag = "[NO-COT MASKED MODE]" if no_cot_mask_flag else "[NO-COT MODE]"
                logging.warning(f"{tag} Clearing all CoT - training to predict answers directly from questions")
                cots = [""] * len(cots)
                superpose_cots = [""] * len(superpose_cots)
            if icot_full_filter_non_digit_answers and skipped_icot_full_non_digit_answers > 0:
                logging.warning(
                    "[ICOT-FULL] Skipped %d examples with non-digit-leading final answer tokens.",
                    skipped_icot_full_non_digit_answers,
                )
            if superpose_mode:
                if superpose_include_last_target_only and not bool(training_args.include_last_cot):
                    logging.warning(
                        "[SUPERPOSE-CCOT MODE] Keeping the last CoT step only in the latent superpose targets; "
                        "the reference/CODI path still excludes it."
                    )
                else:
                    logging.warning("[SUPERPOSE-CCOT MODE] Keeping ground-truth CoTs for latent superposed-token supervision targets.")
                if bool(
                    getattr(training_args, "ccot_superpose_kl", False)
                    and getattr(training_args, "ccot_superpose_kl_context_free_step_bag", False)
                ):
                    logging.warning(
                        "[SUPERPOSE-CCOT MODE] KL targets use stripped standalone-step bags; "
                        "embed/teacher-aligned targets remain in-context."
                    )

            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(
                questions,
                cots,
                answers,
                tokenizer,
                bot,
                eot,
                superpose_targets=superpose_cots if superpose_mode else None,
                superpose_natural_language_steps=_uses_natural_language_cot_steps(self.data_name),
            )
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            has_superpose_targets = "cot_superpose_step_token_ids" in instances[0]
            has_superpose_kl_targets = "cot_superpose_kl_step_token_ids" in instances[0]

            encoder_input_ids = [instance["encoder_input_ids"] for instance in instances]
            decoder_input_ids = [instance["decoder_input_ids"] for instance in instances]
            ref_input_ids = [instance["ref_input_ids"] for instance in instances]
            labels = [instance["labels"] for instance in instances]
            ref_answer_position = [instance["ref_answer_position"] for instance in instances]
            model_answer_position = [instance["model_answer_position"] for instance in instances]
            ref_labels = [instance["ref_labels"] for instance in instances]
            answer_values = [instance["answer_values"] for instance in instances]
            cot_step_counts = [instance.get("cot_step_counts", 0) for instance in instances]
            cot_step_token_counts = [instance.get("cot_step_token_counts", []) for instance in instances]
            ref_superpose_input_ids = [instance["ref_superpose_input_ids"] for instance in instances] if "ref_superpose_input_ids" in instances[0] else None
            ref_superpose_labels = [instance["ref_superpose_labels"] for instance in instances] if "ref_superpose_labels" in instances[0] else None
            cot_superpose_step_token_ids = [instance["cot_superpose_step_token_ids"] for instance in instances] if has_superpose_targets else None
            cot_superpose_kl_step_token_ids = [instance["cot_superpose_kl_step_token_ids"] for instance in instances] if has_superpose_kl_targets else None
            cot_superpose_step_counts = [instance.get("cot_superpose_step_counts", 0) for instance in instances] if has_superpose_targets else None

            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)

            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX)
            if ref_superpose_input_ids is not None:
                ref_superpose_input_ids = torch.nn.utils.rnn.pad_sequence(
                    ref_superpose_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                ref_superpose_labels = torch.nn.utils.rnn.pad_sequence(
                    ref_superpose_labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX,
                )

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            batch = dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
                ref_step_counts=torch.tensor(cot_step_counts, dtype=torch.long),
                answer_values=torch.tensor(answer_values, dtype=torch.float),
            )
            if ref_superpose_input_ids is not None:
                batch["ref_superpose_input_ids"] = ref_superpose_input_ids
                batch["ref_superpose_labels"] = ref_superpose_labels
            max_steps = max((len(x) for x in cot_step_token_counts), default=0)
            if max_steps > 0:
                padded = torch.zeros((len(instances), max_steps), dtype=torch.long)
                for i, lst in enumerate(cot_step_token_counts):
                    if not lst:
                        continue
                    ln = min(len(lst), max_steps)
                    padded[i, :ln] = torch.tensor(lst[:ln], dtype=torch.long)
                batch["ref_step_token_counts"] = padded
            if has_superpose_targets:
                max_superpose_steps = max((len(x) for x in cot_superpose_step_token_ids), default=0)
                max_superpose_tokens = max(
                    (step_ids.numel() for steps in cot_superpose_step_token_ids for step_ids in steps),
                    default=0,
                )
                superpose_ids = torch.full(
                    (len(instances), max_superpose_steps, max_superpose_tokens),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                superpose_token_mask = torch.zeros(
                    (len(instances), max_superpose_steps, max_superpose_tokens),
                    dtype=torch.float,
                )
                superpose_step_mask = torch.zeros(
                    (len(instances), max_superpose_steps),
                    dtype=torch.float,
                )
                for i, steps in enumerate(cot_superpose_step_token_ids):
                    if not steps:
                        continue
                    for j, step_ids in enumerate(steps[:max_superpose_steps]):
                        ln = min(int(step_ids.numel()), max_superpose_tokens)
                        if ln <= 0:
                            continue
                        superpose_ids[i, j, :ln] = step_ids[:max_superpose_tokens]
                        superpose_token_mask[i, j, :ln] = 1.0
                        superpose_step_mask[i, j] = 1.0
                batch["cot_superpose_step_token_ids"] = superpose_ids
                batch["cot_superpose_step_token_mask"] = superpose_token_mask
                batch["cot_superpose_step_mask"] = superpose_step_mask
                if cot_superpose_step_counts is not None:
                    batch["cot_superpose_step_counts"] = torch.tensor(cot_superpose_step_counts, dtype=torch.long)
            if has_superpose_kl_targets:
                max_superpose_kl_steps = max((len(x) for x in cot_superpose_kl_step_token_ids), default=0)
                max_superpose_kl_tokens = max(
                    (step_ids.numel() for steps in cot_superpose_kl_step_token_ids for step_ids in steps),
                    default=0,
                )
                superpose_kl_ids = torch.full(
                    (len(instances), max_superpose_kl_steps, max_superpose_kl_tokens),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                superpose_kl_token_mask = torch.zeros(
                    (len(instances), max_superpose_kl_steps, max_superpose_kl_tokens),
                    dtype=torch.float,
                )
                for i, steps in enumerate(cot_superpose_kl_step_token_ids):
                    if not steps:
                        continue
                    for j, step_ids in enumerate(steps[:max_superpose_kl_steps]):
                        ln = min(int(step_ids.numel()), max_superpose_kl_tokens)
                        if ln <= 0:
                            continue
                        superpose_kl_ids[i, j, :ln] = step_ids[:max_superpose_kl_tokens]
                        superpose_kl_token_mask[i, j, :ln] = 1.0
                batch["cot_superpose_kl_step_token_ids"] = superpose_kl_ids
                batch["cot_superpose_kl_step_token_mask"] = superpose_kl_token_mask
            return batch

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        if data_args.train_path is not None or "openai" in str(data_args.data_name).lower() or "chain" in str(data_args.data_name).lower():
            train_path = data_args.train_path
            if train_path is None:
                raise ValueError("data_args.train_path must be set for chain/openai datasets.")
            dataset = load_local_json_with_cleanup(train_path)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        if _is_whynlp_gsm8k_aug_nl(data_args.data_name):
            dataset = load_dataset("whynlp/gsm8k-aug-nl")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        if "icot" in data_args.data_name:
            if 'full' in data_args.data_name:
                dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
            else:
                dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")


    def build_eval_dataset(data_args, tokenizer, bot, eot):
        if data_args.eval_path is not None or "openai" in str(data_args.eval_data_name).lower() or "chain" in str(data_args.eval_data_name).lower():
            eval_path = data_args.eval_path
            if eval_path is None:
                raise ValueError("data_args.eval_path must be set for chain/openai eval datasets.")
            raw_data = load_local_json_with_cleanup(eval_path)
        elif _is_whynlp_gsm8k_aug_nl(data_args.eval_data_name):
            raw_data = load_dataset("whynlp/gsm8k-aug-nl")["test"]
        elif _is_icot_full_dataset(data_args.eval_data_name):
            raw_data = load_dataset("zen-E/GSM8k-Aug")["test"]
        elif "commonsense" in data_args.eval_data_name:
            raw_data = load_dataset("zen-E/CommonsenseQA-GPT4omini")["validation"]
        elif "strategy-multi-cot" in data_args.eval_data_name:
            raw_data = load_dataset("zen-E/StrategyQA_GPT4o_CoTx10")["validation"]
        elif "strategy" in data_args.eval_data_name:
            raw_data = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["validation"]
        else:
            raw_data = load_dataset("zen-E/GSM8k-Aug")["test"]
        if training_args.exp_mode:
            if isinstance(raw_data, list):
                raw_data = raw_data[:training_args.exp_data_num]
            else:
                max_n = min(training_args.exp_data_num, len(raw_data))
                raw_data = raw_data.select(range(max_n))

        return SupervisedDataset(
            data_name=data_args.eval_data_name,
            raw_data=raw_data,
            tokenizer=tokenizer,
            bot=bot,
            eot=eot,
        )

    def make_eval_data_loader(dataset, tokenizer, batch_size):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        )


    class EvalCallback(TrainerCallback):
        def __init__(self, training_args, model_args, data_args, val_loader):
            super().__init__()
            self.training_args = training_args
            self.model_args = model_args
            self.data_args = data_args
            self.val_loader = val_loader
            self.save_count = 0
            self.has_evaluated = False
            self._evaluated_ckpt_dirs = set()

            self.best_score = float("-inf")
            self.best_ckpt_dir = None
            self.best_global_step = None

        def _extract_score(self, eval_result):
            """Use the first returned value from evaluation(...) as the score (single accuracy %)."""
            if isinstance(eval_result, (tuple, list)):
                return float(eval_result[0])
            return float(eval_result)

        def _sync_best_dir(self, args, ckpt_dir):
            """
            Create/refresh output_dir/best to mirror ckpt_dir.
            Preference: symlink (fast). Fallback: copytree (portable).
            """
            best_dir = os.path.join(args.output_dir, "best")

            if os.path.islink(best_dir) or os.path.exists(best_dir):
                try:
                    if os.path.islink(best_dir):
                        os.unlink(best_dir)
                    else:
                        shutil.rmtree(best_dir)
                except Exception:
                    pass

            abs_ckpt = os.path.abspath(ckpt_dir)
            abs_best = os.path.abspath(best_dir)
            try:
                os.symlink(abs_ckpt, abs_best, target_is_directory=True)
            except Exception:
                shutil.copytree(abs_ckpt, abs_best)

            manifest = {
                "metric": "test_accuracy",
                "greater_is_better": True,
                "global_step": self.best_global_step,
                "score_percent": self.best_score,
                "ckpt_dir": ckpt_dir,
            }
            with open(os.path.join(abs_best, "best_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

        def _normalize_ckpt_dir(self, ckpt_dir):
            return os.path.realpath(os.path.abspath(ckpt_dir))

        def _run_eval_for_ckptdir(self, args, state, ckpt_dir):
            eval_key = self._normalize_ckpt_dir(ckpt_dir)
            if eval_key in self._evaluated_ckpt_dirs:
                print(f"[EvalCallback] Skipping duplicate evaluation for ckpt: {eval_key}")
                return False

            print(f"[EvalCallback] running evaluation for ckpt: {ckpt_dir}")
            self.model_args.ckpt_dir = ckpt_dir
            if self.data_args.batch_size <= 1:
                self.data_args.batch_size = self.training_args.per_device_eval_batch_size
            _test_module.do_print = False

            if args.top_k > 1:
                test_acc, test_sample_acc, test_maj_acc, test_topk_acc = evaluation(
                    self.model_args,
                    self.data_args,
                    self.training_args,
                    split="test",
                    top_k=args.top_k,
                    log_to_wandb=False,
                )
                train_acc, train_sample_acc, train_maj_acc, train_topk_acc = evaluation(
                    self.model_args,
                    self.data_args,
                    self.training_args,
                    split="train",
                    top_k=args.top_k,
                    log_to_wandb=False,
                )
            else:
                test_acc, test_sample_acc = evaluation(
                    self.model_args,
                    self.data_args,
                    self.training_args,
                    split="test",
                    top_k=args.top_k,
                    log_to_wandb=False,
                )
                train_acc, train_sample_acc = evaluation(
                    self.model_args,
                    self.data_args,
                    self.training_args,
                    split="train",
                    top_k=args.top_k,
                    log_to_wandb=False,
                )


            tag = ("strategy" if "strategy" in self.data_args.eval_data_name
                else "commonsense" if "commonsense" in self.data_args.eval_data_name
                else "gsm8k")

            eval_payload = {
                "epoch": state.epoch,
                f"{tag}_test_accuracy_greedy": test_acc,
                f"{tag}_test_accuracy_sample": test_sample_acc,
                f"{tag}_train_accuracy_greedy": train_acc,
                f"{tag}_train_accuracy_sample": train_sample_acc,
            }
            if args.top_k > 1:
                eval_payload[f"{tag}_test_majority_accuracy"] = test_maj_acc
                eval_payload[f"{tag}_test_top{args.top_k}_accuracy"] = test_topk_acc
                eval_payload[f"{tag}_train_majority_accuracy"] = train_maj_acc
                eval_payload[f"{tag}_train_top{args.top_k}_accuracy"] = train_topk_acc

            try:
                print(f"[EvalCallback] {tag.title()} Test greedy @step {state.global_step}: {test_acc:.2f}% | sample@1 {test_sample_acc:.2f}%")
                print(f"[EvalCallback] {tag.title()} Train greedy @step {state.global_step}: {train_acc:.2f}% | sample@1 {train_sample_acc:.2f}%")
                if args.top_k > 1:
                    print(f"[EvalCallback] {tag.title()} Test Majority @step {state.global_step}: {test_maj_acc:.2f}%")
                    print(f"[EvalCallback] {tag.title()} Test Top-{args.top_k} @step {state.global_step}: {test_topk_acc:.2f}%")
                    print(f"[EvalCallback] {tag.title()} Train Majority @step {state.global_step}: {train_maj_acc:.2f}%")
                    print(f"[EvalCallback] {tag.title()} Train Top-{args.top_k} @step {state.global_step}: {train_topk_acc:.2f}%")
            except Exception:
                print(f"[EvalCallback] {tag.title()} Test greedy @step {state.global_step}: {test_acc}")
                print(f"[EvalCallback] {tag.title()} Train greedy @step {state.global_step}: {train_acc}")
                if args.top_k > 1:
                    print(f"[EvalCallback] {tag.title()} Test Majority @step {state.global_step}: {test_maj_acc}")
                    print(f"[EvalCallback] {tag.title()} Test Top-{args.top_k} @step {state.global_step}: {test_topk_acc}")
                    print(f"[EvalCallback] {tag.title()} Train Majority @step {state.global_step}: {train_maj_acc}")
                    print(f"[EvalCallback] {tag.title()} Train Top-{args.top_k} @step {state.global_step}: {train_topk_acc}")
            if wandb.run is not None:
                wandb.log(eval_payload, step=state.global_step)
            self._evaluated_ckpt_dirs.add(eval_key)

            score = self._extract_score(test_acc)
            if score > self.best_score:
                self.best_score = score
                self.best_ckpt_dir = ckpt_dir
                self.best_global_step = state.global_step
                self._sync_best_dir(args, ckpt_dir)
                print(f"[EvalCallback] New BEST ({self.best_score:.2f}%) at step {self.best_global_step} -> {self.best_ckpt_dir}")
            else:
                print(f"[EvalCallback] No improvement (best={self.best_score:.2f}%).")

        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            if self.training_args.resume_from_checkpoint and getattr(self.training_args, "eval_on_resume", True):
                ckpt_dir = self.training_args.resume_from_checkpoint
                print("[EvalCallback] Pre-train checkpoint detected; running early evaluation...")
                self._run_eval_for_ckptdir(args, state, ckpt_dir)
            else:
                print("[EvalCallback] Not resuming; skipping early evaluation.")
            return control

        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            self.save_count += 1
            if self.training_args.eval_every_save is not None:
                if self.training_args.eval_every_save <= 0:
                    print(
                        f"[EvalCallback] Checkpoint saved (count={self.save_count}); "
                        f"eval_every_save={self.training_args.eval_every_save} disables on-save evaluation."
                    )
                    return control
                if (self.save_count % self.training_args.eval_every_save) != 0:
                    print(f"[EvalCallback] Checkpoint saved (count={self.save_count}); skipping evaluation (eval_every_save={self.training_args.eval_every_save}).")
                    return control

            print("[EvalCallback] Checkpoint saved. Running evaluation on saved weights...")
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            did_eval = self._run_eval_for_ckptdir(args, state, ckpt_dir)
            print("[EvalCallback] evaluation complete.")
            self.has_evaluated = bool(did_eval) or self.has_evaluated
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            if self.training_args.resume_from_checkpoint and not getattr(self.training_args, "eval_on_resume", True):
                print("[EvalCallback] eval_on_resume is False and resume_from_checkpoint is set; skipping final evaluation.")
                return control

            if self.has_evaluated:
                return control

            if self.training_args.eval_every_save is not None and self.training_args.eval_every_save < 0:
                print("[EvalCallback] eval_every_save<0; skipping final evaluation.")
                return control

            last_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            ckpt_dir = last_ckpt if os.path.isdir(last_ckpt) else args.output_dir
            has_bin = os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin"))
            has_safetensors = os.path.isfile(os.path.join(ckpt_dir, "model.safetensors"))
            if not (has_bin or has_safetensors):
                print(f"[EvalCallback] No evaluation ran during training; skipping final evaluation because checkpoint not found in {ckpt_dir}.")
                return control

            print("[EvalCallback] No evaluation ran during training; running final evaluation...")
            did_eval = self._run_eval_for_ckptdir(args, state, ckpt_dir)
            self.has_evaluated = bool(did_eval) or self.has_evaluated
            return control


    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    print("Preprocessing validation set...")
    val_dataset = build_eval_dataset(data_args, tokenizer, model.bot_id, model.eot_id)
    val_loader = make_eval_data_loader(val_dataset, tokenizer, training_args.eval_batch_size)


    dataset_size = len(data_module["train_dataset"])
    steps_per_epoch = ceil(dataset_size / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * training_args.num_train_epochs

    save_steps = math.ceil(training_args.save_ratio * total_steps)
    if save_steps <= 0:
        save_steps = 1
        print(f"[warn] save_ratio={training_args.save_ratio} is too small for total_steps={total_steps}; defaulting save_steps to 1.")
    training_args.save_steps = save_steps
    print(f"Model will be saved after {save_steps} steps ({training_args.save_ratio * 100}% of training)")

    callbacks = [EvalCallback(training_args, model_args, data_args, val_loader)]

    print(f"[torch_compile] torch_compile={training_args.torch_compile}")

    trainer = CustomTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module, callbacks=callbacks)

    trainer.val_loader = val_loader
    resume_checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
