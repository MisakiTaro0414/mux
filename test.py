#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True


def _default_device() -> torch.device:
    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        visible = os.environ.get(env_name)
        if visible is None:
            continue
        visible = visible.strip()
        if visible and visible.lower() not in {"-1", "none", "void", "nodevfiles"}:
            return torch.device("cuda")
        if visible.lower() in {"-1", "none", "void", "nodevfiles"}:
            return torch.device("cpu")
    return torch.device("cpu")


device = _default_device()
print(device)


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


def _load_eval_split(data_args, split: str):
    split = str(split).lower()
    question_name = "question"
    answer_name = "answer"
    dataset_name = str(data_args.data_name)
    normalized_name = dataset_name.strip().lower().replace("_", "-")

    if normalized_name == "gsm-hard":
        hf = load_dataset("juyoung-trl/gsm-hard")
        test_set = hf["train"]
        question_name = "instruction"
        answer_name = "response"
    elif normalized_name in {"multi-arith", "multiarith"}:
        hf = load_dataset("ChilleD/MultiArith")
        test_set = hf["test"]
        answer_name = "final_ans"
    elif normalized_name == "svamp":
        hf = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([hf["train"], hf["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif normalized_name == "commonsense":
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        if split == "train" and "train" in dataset:
            test_set = dataset["train"]
        elif "validation" in dataset:
            test_set = dataset["validation"]
        elif "test" in dataset:
            test_set = dataset["test"]
        else:
            test_set = dataset["train"]
    elif "gsm8k" in normalized_name or "icot" in normalized_name:
        if split == "train":
            if getattr(data_args, "train_path", None):
                test_set = read_json(data_args.train_path)
                if test_set is None:
                    raise FileNotFoundError(f"Failed to load training data from {data_args.train_path}")
            else:
                test_set = load_dataset("zen-E/GSM8k-Aug")["train"]
        else:
            if getattr(data_args, "eval_path", None):
                test_set = read_json(data_args.eval_path)
                if test_set is None:
                    raise FileNotFoundError(f"Failed to load evaluation data from {data_args.eval_path}")
            else:
                hf = load_dataset("zen-E/GSM8k-Aug")
                if "test" in hf:
                    test_set = hf["test"]
                elif "validation" in hf:
                    test_set = hf["validation"]
                else:
                    test_set = hf["train"]
    else:
        raise NotImplementedError

    if test_set is None:
        raise RuntimeError(f"Failed to resolve dataset for data_name={dataset_name!r}")

    logging.warning(
        "Resolved eval dataset | requested=%s normalized=%s split=%s size=%s q_field=%s a_field=%s",
        dataset_name, normalized_name, split, len(test_set), question_name, answer_name,
    )

    return test_set, question_name, answer_name


def _load_checkpoint_state_dict(ckpt_dir: str):
    if os.path.isfile(ckpt_dir):
        if ckpt_dir.endswith(".safetensors"):
            return load_file(ckpt_dir)
        return torch.load(ckpt_dir, map_location="cpu", weights_only=False)

    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    pytorch_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        return load_file(safetensors_path)
    if os.path.isfile(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu", weights_only=False)

    if os.path.isdir(ckpt_dir):
        available = ", ".join(sorted(os.listdir(ckpt_dir))) or "<empty>"
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {ckpt_dir}. Available: {available}"
        )
    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_dir}")


def evaluation(model_args, data_args, training_args, split: str = "test", top_k=1, log_to_wandb=False, wandb_step=None):
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
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)

    state_dict = _load_checkpoint_state_dict(model_args.ckpt_dir)
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    test_set, question_name, answer_name = _load_eval_split(data_args, split=split)

    if getattr(training_args, "exp_mode", False) and getattr(training_args, "exp_data_num", None):
        max_n = min(training_args.exp_data_num, len(test_set))
        if isinstance(test_set, list):
            test_set = test_set[:max_n]
        else:
            test_set = test_set.select(range(max_n))
        logging.warning("exp_mode: truncated eval set to %d examples", max_n)

    logging.warning("Formatting inputs...")
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = []

    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            answer.append(example == "True")
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size} | eval steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i * data_args.batch_size: (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i * data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )

        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    def generate_batch(batch, greedy: bool):
        preds = []
        len_cot_local = []
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            past_key_values = None
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"],
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            inf_latent_iterations = training_args.inf_latent_iterations
            for _ in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)

            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
            output = eot_emb

            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]
            for _ in range(gen_kwargs["max_new_tokens"]):
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).reshape(-1)
                else:
                    logits = logits / gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).reshape(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot_local.append(len(pred_token))
                try:
                    decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                except TypeError:
                    decoded_pred = ""
                if do_print:
                    print(f"Generated: {decoded_pred}")
                    print(
                        f"Extracted answer: {extract_answer_number(decoded_pred, data_args.data_name)}; "
                        f"GT: {answer[step * data_args.batch_size + mini_step]}"
                    )
                preds.append(extract_answer_number(decoded_pred, data_args.data_name))
        return preds, len_cot_local

    greedy_preds = []
    len_cot_all = []
    for step, batch in enumerate(question_data):
        logging.warning(f"Eval progress: batch {step + 1}/{eval_step}")
        gp, gl = generate_batch(batch, greedy=True)
        greedy_preds.extend(gp)
        len_cot_all.extend(gl)

    accuracy_greedy = compute_accuracy(answer, greedy_preds)
    avg_cot_len = sum(len_cot_all) / max(len(len_cot_all), 1)
    print(f"Greedy accuracy: {100 * accuracy_greedy:.2f}% | avg_cot_len: {avg_cot_len:.2f}")

    greedy_ret = 100 * accuracy_greedy
    if top_k > 1:
        return greedy_ret, greedy_ret, greedy_ret, greedy_ret
    else:
        return greedy_ret, greedy_ret


def extract_answer_number(sentence: str, dataset_name: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        dataset_name = str(dataset_name)
        if "commonsense" in dataset_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                raise ValueError
            return pred[0]
        elif "strategy" in dataset_name or "prontoqa" in dataset_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    return float(pred[-1])


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1
    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    greedy_list = []
    for i in range(training_args.inf_num_iterations):
        result = evaluation(model_args, data_args, training_args, top_k=training_args.top_k)
        if training_args.top_k > 1:
            greedy_list.append(result[0])
        else:
            greedy_list.append(result[0])
    print(f"Average greedy accuracy over {training_args.inf_num_iterations} runs: {sum(greedy_list) / len(greedy_list):.2f}%")
