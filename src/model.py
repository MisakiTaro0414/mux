import inspect
import os
import time

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from collections import Counter
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_name: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "Optional local JSON path for training data (chain format supported)."}
    )
    eval_path: Optional[str] = field(
        default=None, metadata={"help": "Optional local JSON path for eval/test data (chain format supported)."}
    )
    chain_mode: str = field(
        default="first",
        metadata={"help": "How to use reasoning chains: first | all | first_k."},
    )
    chain_k: int = field(
        default=1,
        metadata={"help": "K for chain_mode=first_k; examples with fewer chains are skipped."},
    )
    chain_joiner: str = field(
        default="\n",
        metadata={"help": "String used to join steps inside a chain."},
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    reasoning_mode: str = field(
        default="codi",
        metadata={"help": "Reasoning mode to use. Only 'codi' is supported."},
    )
    icot_train_path: str = field(default="", metadata={"help": "The training data path"})
    num_latent: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    teacher_only: bool = field(default=False, metadata={"help": "If true, skip student/latent path and train only the teacher CE (supervised fine-tuning)."})
    ccot_only: bool = field(default=False, metadata={"help": "If true, train only the student CE loss (continuous CoT) without distillation and teacher loss."})
    ccot_kl_temperature: float = field(default=1.0, metadata={"help": "Temperature for KL softmax computation."})
    ccot_superpose_kl: bool = field(default=False, metadata={"help": "Enable latent-step KL to superposed token targets from structured CoT steps (no teacher model)."})
    ccot_superpose_kl_use_bag: bool = field(default=True, metadata={"help": "Include the original bag-of-tokens superpose KL term."})
    ccot_superpose_kl_scalar_positional: bool = field(default=False, metadata={"help": "Add single-distribution order-aware KL using distinct scalar position weights."})
    ccot_superpose_kl_scalar_positional_weight: float = field(default=1.0, metadata={"help": "Weight of the scalar-position-weighted KL term."})
    ccot_superpose_kl_scalar_positional_mode: str = field(default="geometric", metadata={"help": "Scalar positional weighting scheme: 'geometric', 'sinusoidal', 'transformer_sinusoidal', 'rope', 'learned', or 'fb'."})
    ccot_superpose_kl_scalar_positional_decay: float = field(default=0.9, metadata={"help": "Decay base for geometric scalar positional KL weights."})
    ccot_superpose_kl_scalar_positional_sin_amp: float = field(default=1.0, metadata={"help": "Amplitude for sinusoidal-like scalar positional KL weights."})
    ccot_superpose_kl_scalar_positional_fourier_pairs: int = field(default=16, metadata={"help": "Number of sin/cos frequency pairs for scalar positional KL modes."})
    ccot_superpose_kl_scalar_positional_base: float = field(default=10000.0, metadata={"help": "Base for Transformer/RoPE-style sinusoidal frequency schedule."})
    ccot_superpose_kl_scalar_positional_freq_sampling: str = field(default="spread", metadata={"help": "How to choose Fourier frequency pairs: 'prefix' or 'spread'."})
    ccot_superpose_kl_scalar_positional_max_positions: int = field(default=128, metadata={"help": "Maximum position table size for learned scalar positional KL weights."})
    ccot_superpose_kl_target_q_feedback: bool = field(default=False, metadata={"help": "In superpose-KL mode, enable q-feedback latent inputs."})
    ccot_superpose_kl_q_feedback_train_source: str = field(default="target", metadata={"help": "Training-only source for q-feedback: 'target' or 'generated'."})
    ccot_superpose_kl_train_lm_head: bool = field(default=False, metadata={"help": "Make the final LM head fully trainable by replacing it with a standalone linear layer."})
    ccot_superpose_kl_train_lm_head_reinit: bool = field(default=False, metadata={"help": "Replace LM head with a randomly initialized trainable linear layer."})
    ccot_superpose_uniform_chunk_to_latents: bool = field(default=False, metadata={"help": "Uniformly chunk CoT steps into num_latent contiguous groups when M>K."})
    ccot_superpose_random_chunk_to_latents: bool = field(default=False, metadata={"help": "Randomly chunk CoT steps into num_latent contiguous groups when M>K."})
    ccot_superpose_k_plus_one_slots: bool = field(default=True, metadata={"help": "Expose num_latent+1 auxiliary superpose slots."})
    ccot_superpose_num_supervised_latents: int = field(default=0, metadata={"help": "If >0, supervise only the first N superpose auxiliary latent slots."})
    ccot_superpose_kl_weight: float = field(default=1.0, metadata={"help": "Weight multiplier for the superpose-KL auxiliary component."})
    ccot_superpose_beta: float = field(default=0.1, metadata={"help": "Initial weight for latent superposed-token KL term."})
    ccot_superpose_beta_min: float = field(default=0.03, metadata={"help": "Minimum weight for latent superposed-token KL after scheduling."})
    ccot_superpose_strip_markers: bool = field(default=True, metadata={"help": "Strip leading '<<' and trailing '>>' before tokenizing each structured CoT step target."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"})
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."})
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the reference loss."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    icot_full_filter_non_digit_answers: bool = field(
        default=True,
        metadata={
            "help": "For icot-full only, skip examples whose final answer token does not start with a digit."
        },
    )
    ccot_superpose_include_last_cot_target_only: bool = field(
        default=True,
        metadata={
            "help": "In superpose KL modes, keep the last CoT step only in the auxiliary superpose targets."
        },
    )
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Keep all columns from dataset."})
    save_ratio: float = field(
        default=0.1,
        metadata={
            "help": "The epoch ratio to save the model."
        },
    )
    eval_every_save: int = field(
        default=1,
        metadata={
            "help": "Run EvalCallback every N save events."
        },
    )
    top_k: int = field(
        default=1,
        metadata={"help": "Top-k sampling for inference."}
    )
    eval_max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum new tokens to generate during evaluation."}
    )
    eval_temperature: float = field(
        default=0.6,
        metadata={"help": "Sampling temperature for non-greedy evaluation generation."}
    )
    eval_top_k_logits: int = field(
        default=40,
        metadata={"help": "Top-k truncation over logits for non-greedy evaluation sampling (<=0 disables)."}
    )
    eval_disable_greedy: bool = field(
        default=False,
        metadata={"help": "If True, skip greedy decoding during evaluation and run sampling only."}
    )
    eval_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, evaluate on at most this many samples per split."}
    )
    eval_on_resume: bool = field(
        default=True,
        metadata={"help": "If False, skip automatic evaluation when resuming from checkpoint."}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_default_hf_timeouts():
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def _unique_existing_dirs(paths: List[Optional[str]]) -> List[str]:
    resolved = []
    seen = set()
    for path in paths:
        if not path:
            continue
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if expanded in seen or not os.path.isdir(expanded):
            continue
        resolved.append(expanded)
        seen.add(expanded)
    return resolved


def _candidate_hf_cache_roots(cache_dir: Optional[str]) -> List[str]:
    hf_home = os.getenv("HF_HOME")
    default_hf_home = os.path.expanduser("~/.cache/huggingface")
    return _unique_existing_dirs([
        cache_dir,
        os.getenv("HF_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
        os.path.join(hf_home, "hub") if hf_home else None,
        os.path.join(default_hf_home, "hub"),
    ])


def _snapshot_looks_usable(snapshot_dir: str) -> bool:
    if not os.path.isdir(snapshot_dir):
        return False
    names = set(os.listdir(snapshot_dir))
    has_config = "config.json" in names
    has_tokenizer = any(
        name in names
        for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )
    has_weights = any(
        name.endswith((".safetensors", ".bin"))
        or name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        for name in names
    )
    return has_config and has_tokenizer and has_weights


def _find_cached_snapshot(model_name_or_path: str, cache_dir: Optional[str] = None) -> Optional[str]:
    repo_id = model_name_or_path.strip().strip("/")
    if "/" not in repo_id:
        return None
    cache_leaf = f"models--{repo_id.replace('/', '--')}"
    for root in _candidate_hf_cache_roots(cache_dir):
        snapshots_dir = os.path.join(root, cache_leaf, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        snapshot_dirs = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
        for snapshot_dir in snapshot_dirs:
            if _snapshot_looks_usable(snapshot_dir):
                return snapshot_dir
    return None


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    supported_kwargs = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in signature.parameters:
            supported_kwargs[key] = value
    return fn(**supported_kwargs)


def resolve_pretrained_model_path(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Tuple[str, bool]:
    expanded_path = os.path.abspath(os.path.expanduser(os.path.expandvars(model_name_or_path)))
    if os.path.isdir(expanded_path):
        return expanded_path, True
    if model_name_or_path.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Local model path does not exist: {model_name_or_path}")

    cached_snapshot = _find_cached_snapshot(model_name_or_path, cache_dir=cache_dir)
    if cached_snapshot is not None:
        print(f"[info] Using cached Hugging Face snapshot for {model_name_or_path}: {cached_snapshot}")
        return cached_snapshot, True

    if _env_flag("HF_HUB_OFFLINE"):
        raise FileNotFoundError(
            f"HF_HUB_OFFLINE=1 but no cached snapshot was found for {model_name_or_path}."
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_name_or_path, False

    _set_default_hf_timeouts()
    max_workers = max(1, int(os.getenv("HF_HUB_DOWNLOAD_WORKERS", "1")))
    retries = max(1, int(os.getenv("HF_HUB_DOWNLOAD_RETRIES", "4")))
    backoff_seconds = max(1.0, float(os.getenv("HF_HUB_DOWNLOAD_BACKOFF", "15")))
    etag_timeout = float(os.getenv("HF_HUB_ETAG_TIMEOUT", "120"))

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_path = _call_with_supported_kwargs(
                snapshot_download,
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                token=token,
                local_files_only=False,
                resume_download=True,
                etag_timeout=etag_timeout,
                max_workers=max_workers,
            )
            return os.path.abspath(snapshot_path), True
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = backoff_seconds * attempt
            print(
                f"[warn] snapshot_download failed for {model_name_or_path} "
                f"(attempt {attempt}/{retries}): {exc}. Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)

    cached_snapshot = _find_cached_snapshot(model_name_or_path, cache_dir=cache_dir)
    if cached_snapshot is not None:
        print(
            f"[warn] Falling back to cached snapshot after repeated download failures: {cached_snapshot}"
        )
        return cached_snapshot, True

    raise RuntimeError(
        "Failed to fetch the Hugging Face model snapshot. "
        f"Tried {retries} time(s) for {model_name_or_path}. "
        "Set MODEL_NAME to a local snapshot path or increase HF_HUB_* timeout settings."
    ) from last_error

class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.hf_token = model_args.token or os.getenv("HF_TOKEN")
        self.model_source, self.model_source_is_local = resolve_pretrained_model_path(
            self.model_name,
            cache_dir=training_args.cache_dir,
            token=self.hf_token,
        )
        model_wrapper_class = AutoModelForCausalLM
        common_model_kwargs = dict(
            cache_dir=training_args.cache_dir,
            token=self.hf_token,
            local_files_only=self.model_source_is_local,
        )
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_source,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    **common_model_kwargs,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_source,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    **common_model_kwargs,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )


        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # Tokenizer and special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            use_fast=False,
            cache_dir=training_args.cache_dir,
            token=self.hf_token,
            local_files_only=self.model_source_is_local,
        )
        scratch_version = int(getattr(self.training_args, "ccot_scratch_version", 1))
        scratch_marker = getattr(self.training_args, "ccot_scratch_marker_v1", "<SCRATCH>")
        final_marker = getattr(self.training_args, "ccot_final_marker_v1", "<FINAL>")
        if scratch_version == 2:
            scratch_marker = getattr(self.training_args, "ccot_scratch_marker_v2", "[SCRATCH]")
            final_marker = getattr(self.training_args, "ccot_final_marker_v2", "[FINAL]")
        self.scratch_marker = scratch_marker
        self.final_marker = final_marker
        use_scratch_tokenizer_markers = bool(getattr(self.training_args, "ccot_scratch_kl", False)) and scratch_version == 1

        if use_scratch_tokenizer_markers:
            special_tokens = {}
            additional_tokens = []
            if self.tokenizer.pad_token_id is None:
                special_tokens["pad_token"] = "[PAD]"
            for tok in ["<BOT>", "<EOT>", scratch_marker, final_marker]:
                if tok not in self.tokenizer.get_vocab():
                    additional_tokens.append(tok)
            if additional_tokens:
                existing_specials = getattr(self.tokenizer, "special_tokens_map_extended", {}).get("additional_special_tokens", [])
                existing_specials = [str(x) for x in list(existing_specials)]
                merged_specials = list(existing_specials)
                for tok in additional_tokens:
                    if tok not in merged_specials:
                        merged_specials.append(tok)
                special_tokens["additional_special_tokens"] = merged_specials
            if special_tokens:
                self.tokenizer.add_special_tokens(special_tokens)

            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else ori_vocab_size
            self.bot_id = self.tokenizer.convert_tokens_to_ids("<BOT>")
            self.eot_id = self.tokenizer.convert_tokens_to_ids("<EOT>")
            self.scratch_token_id = self.tokenizer.convert_tokens_to_ids(scratch_marker)
            self.final_token_id = self.tokenizer.convert_tokens_to_ids(final_marker)

            new_vocab_size = len(self.tokenizer)
            old_vocab_size = self.codi.get_input_embeddings().weight.size(0)
            self.added_new_tokens = max(0, new_vocab_size - old_vocab_size)
            if new_vocab_size > old_vocab_size:
                try:
                    self.codi.resize_token_embeddings(new_vocab_size, mean_resizing=False)
                except TypeError:
                    self.codi.resize_token_embeddings(new_vocab_size)
        else:
            self.pad_token_id = ori_vocab_size
            self.bot_id = ori_vocab_size + 1
            self.eot_id = ori_vocab_size + 2
            self.scratch_token_id = None
            self.final_token_id = None
            try:
                self.codi.resize_token_embeddings(ori_vocab_size + 3, mean_resizing=False)
            except TypeError:
                self.codi.resize_token_embeddings(ori_vocab_size + 3)
            self.added_new_tokens = 3
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.tokenizer.pad_token_id = self.pad_token_id

        self.no_cot_latent_mask = getattr(training_args, "no_cot_latent_mask", False)
        self.dim = self.codi.config.hidden_size
        requested_reasoning_mode = (training_args.reasoning_mode or "codi").lower()
        if requested_reasoning_mode != "codi":
            raise ValueError(
                f"reasoning_mode={requested_reasoning_mode!r} is no longer supported. Use reasoning_mode='codi'."
            )
        self.reasoning_mode = "codi"
        self.base_num_heads = getattr(self.codi.config, "num_attention_heads", None)
        if self.base_num_heads is None:
            self.base_num_heads = getattr(self.codi.config, "n_head", 8)
        self.base_num_latent = training_args.num_latent
        self.num_latent = self.base_num_latent

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
        self.superpose_kl_teacher_head = None
        self.ccot_kl_temperature = getattr(training_args, 'ccot_kl_temperature', 1.0)
        self.ccot_superpose_kl = getattr(training_args, 'ccot_superpose_kl', False)
        self.ccot_superpose_kl_use_bag = bool(getattr(training_args, "ccot_superpose_kl_use_bag", True))
        self.ccot_superpose_kl_scalar_positional = bool(getattr(training_args, "ccot_superpose_kl_scalar_positional", False))
        self.ccot_superpose_kl_scalar_positional_weight = float(getattr(training_args, "ccot_superpose_kl_scalar_positional_weight", 1.0))
        _scalar_pos_mode = str(getattr(training_args, "ccot_superpose_kl_scalar_positional_mode", "geometric")).strip().lower()
        if _scalar_pos_mode in {"geo", "exp"}:
            _scalar_pos_mode = "geometric"
        elif _scalar_pos_mode in {"sin", "sine"}:
            _scalar_pos_mode = "sinusoidal"
        elif _scalar_pos_mode in {"transformer", "transformer_sin", "tf_sinusoidal", "vaswani"}:
            _scalar_pos_mode = "transformer_sinusoidal"
        elif _scalar_pos_mode in {"rope", "rope_like", "rope-scalar"}:
            _scalar_pos_mode = "rope"
        elif _scalar_pos_mode in {"learn", "table"}:
            _scalar_pos_mode = "learned"
        elif _scalar_pos_mode in {"fb", "forward_backward", "bidirectional"}:
            _scalar_pos_mode = "fb"
        if _scalar_pos_mode not in {"geometric", "sinusoidal", "transformer_sinusoidal", "rope", "learned", "fb"}:
            raise ValueError(
                f"Unsupported ccot_superpose_kl_scalar_positional_mode={_scalar_pos_mode!r}. "
                "Use one of: geometric, sinusoidal, transformer_sinusoidal, rope, learned, fb."
            )
        self.ccot_superpose_kl_scalar_positional_mode = _scalar_pos_mode
        self.ccot_superpose_kl_scalar_positional_decay = float(getattr(training_args, "ccot_superpose_kl_scalar_positional_decay", 0.9))
        self.ccot_superpose_kl_scalar_positional_sin_amp = float(getattr(training_args, "ccot_superpose_kl_scalar_positional_sin_amp", 1.0))
        self.ccot_superpose_kl_scalar_positional_fourier_pairs = int(
            getattr(training_args, "ccot_superpose_kl_scalar_positional_fourier_pairs", 16)
        )
        self.ccot_superpose_kl_scalar_positional_base = float(
            getattr(training_args, "ccot_superpose_kl_scalar_positional_base", 10000.0)
        )
        _freq_sampling = str(
            getattr(training_args, "ccot_superpose_kl_scalar_positional_freq_sampling", "spread")
        ).strip().lower()
        if _freq_sampling in {"fullspread", "linspace", "even", "uniform"}:
            _freq_sampling = "spread"
        elif _freq_sampling in {"head", "first"}:
            _freq_sampling = "prefix"
        if _freq_sampling not in {"prefix", "spread"}:
            raise ValueError(
                f"Unsupported ccot_superpose_kl_scalar_positional_freq_sampling={_freq_sampling!r}. "
                "Use 'prefix' or 'spread'."
            )
        self.ccot_superpose_kl_scalar_positional_freq_sampling = _freq_sampling
        self.ccot_superpose_kl_scalar_positional_max_positions = int(
            getattr(training_args, "ccot_superpose_kl_scalar_positional_max_positions", 128)
        )
        self.ccot_superpose_kl_target_q_feedback = bool(
            getattr(training_args, "ccot_superpose_kl_target_q_feedback", False)
        )
        _q_feedback_train_source = str(
            getattr(training_args, "ccot_superpose_kl_q_feedback_train_source", "target")
        ).strip().lower()
        if _q_feedback_train_source in {"teacher", "teacher_target", "data", "target_q"}:
            _q_feedback_train_source = "target"
        elif _q_feedback_train_source in {"model", "pred", "prediction", "student"}:
            _q_feedback_train_source = "generated"
        if _q_feedback_train_source not in {"target", "generated"}:
            raise ValueError(
                "Unsupported ccot_superpose_kl_q_feedback_train_source="
                f"{_q_feedback_train_source!r}. Use 'target' or 'generated'."
            )
        self.ccot_superpose_kl_q_feedback_train_source = _q_feedback_train_source
        if self.ccot_superpose_kl_target_q_feedback and not self.ccot_superpose_kl:
            raise ValueError(
                "ccot_superpose_kl_target_q_feedback=True requires ccot_superpose_kl=True."
            )
        self.ccot_superpose_kl_train_lm_head = bool(
            getattr(training_args, "ccot_superpose_kl_train_lm_head", False)
        )
        self.ccot_superpose_kl_train_lm_head_reinit = bool(
            getattr(training_args, "ccot_superpose_kl_train_lm_head_reinit", False)
        )
        if self.ccot_superpose_kl_train_lm_head_reinit:
            self.ccot_superpose_kl_train_lm_head = True
        self.ccot_superpose_uniform_chunk_to_latents = bool(
            getattr(training_args, "ccot_superpose_uniform_chunk_to_latents", False)
        )
        self.ccot_superpose_random_chunk_to_latents = bool(
            getattr(training_args, "ccot_superpose_random_chunk_to_latents", False)
        )
        self.ccot_superpose_k_plus_one_slots = bool(
            getattr(training_args, "ccot_superpose_k_plus_one_slots", True)
        )
        self.ccot_superpose_num_supervised_latents = int(
            getattr(training_args, "ccot_superpose_num_supervised_latents", 0)
        )
        if self.ccot_superpose_num_supervised_latents < 0:
            raise ValueError(
                "ccot_superpose_num_supervised_latents must be >= 0."
            )
        if self.ccot_superpose_uniform_chunk_to_latents and self.ccot_superpose_random_chunk_to_latents:
            raise ValueError(
                "Enable only one of ccot_superpose_uniform_chunk_to_latents or "
                "ccot_superpose_random_chunk_to_latents."
            )
        self.superpose_kl_scalar_positional_logits = None
        if self.ccot_superpose_kl_scalar_positional_mode == "learned":
            max_pos = max(1, self.ccot_superpose_kl_scalar_positional_max_positions)
            self.superpose_kl_scalar_positional_logits = nn.Parameter(torch.zeros(max_pos))
        self.ccot_superpose_kl_weight = float(getattr(training_args, "ccot_superpose_kl_weight", 1.0))
        if self.ccot_superpose_kl_weight < 0.0:
            raise ValueError("ccot_superpose_kl_weight must be >= 0.")
        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.training:
            self.init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the underlying model."""
        if hasattr(self.codi, 'gradient_checkpointing_enable'):
            self.codi.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        elif hasattr(self.codi, 'enable_input_require_grads'):
            self.codi.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the underlying model."""
        if hasattr(self.codi, 'gradient_checkpointing_disable'):
            self.codi.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        """
        Break shared LM-head/input-embedding storage for safe serialization.
        """
        sd = super().state_dict(*args, **kwargs)
        shared_pairs = [
            ("codi.base_model.model.lm_head.weight", "codi.base_model.model.model.embed_tokens.weight"),
            ("codi.lm_head.weight", "codi.transformer.wte.weight"),
        ]
        for head_key, embed_key in shared_pairs:
            if head_key in sd and embed_key in sd:
                if sd[head_key].data_ptr() == sd[embed_key].data_ptr():
                    sd[head_key] = sd[head_key].clone()
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        if state_dict is None:
            return super().load_state_dict(state_dict, strict=strict)
        target_rows = self.codi.get_input_embeddings().weight.size(0)
        embed_keys = [k for k in state_dict.keys() if k.endswith("embed_tokens.weight") or k.endswith("wte.weight")]
        head_keys = [k for k in state_dict.keys() if k.endswith("lm_head.weight")]
        ckpt_rows = max((state_dict[k].size(0) for k in embed_keys + head_keys), default=target_rows)
        if ckpt_rows > target_rows:
            try:
                self.codi.resize_token_embeddings(ckpt_rows, mean_resizing=False)
            except TypeError:
                self.codi.resize_token_embeddings(ckpt_rows)
        need_expand = False
        if getattr(self, "added_new_tokens", 0) > 0:
            target_rows = self.codi.get_input_embeddings().weight.size(0)
            for k in embed_keys + head_keys:
                if state_dict[k].size(0) < target_rows:
                    need_expand = True
                    break
        if not need_expand:
            return super().load_state_dict(state_dict, strict=strict)
        expanded = self._expand_state_dict_for_new_tokens(state_dict.copy())
        return super().load_state_dict(expanded, strict=strict)

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception:
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def _distill_answer_positions(self, ref_answer_position, model_answer_position):
        model_name = str(self.model_name).lower()
        if "llama" in model_name or "qwen" in model_name:
            return ref_answer_position, model_answer_position
        return (ref_answer_position - 1).clamp_min(0), (model_answer_position - 1).clamp_min(0)

    def _prepare_latent_input(
        self,
        hidden_state: torch.Tensor,
        step: Optional[int] = None,
        force_sample: Optional[bool] = None,
    ):
        """
        Transform a hidden state into the next latent input embedding.
        Returns (latent_embd [B,1,H], None)
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(1)

        orig_dtype = hidden_state.dtype
        if self.use_prj:
            hidden_state = self.prj(hidden_state)

        latent_embd = hidden_state.unsqueeze(1)
        latent_embd = latent_embd.to(orig_dtype)
        latent_embd = self._to_model_dtype(latent_embd)
        return latent_embd, None

    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor to the base embedding dtype for safe PEFT/LoRA linear usage."""
        base_dtype = self.codi.get_input_embeddings().weight.dtype
        return tensor.to(base_dtype)

    def _get_output_head(self, model=None):
        model = self.codi if model is None else model
        if hasattr(model, "get_output_embeddings"):
            head = model.get_output_embeddings()
            if head is not None:
                return head
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else None
        if base_model is not None and hasattr(base_model, "get_output_embeddings"):
            head = base_model.get_output_embeddings()
            if head is not None:
                return head
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                return getattr(model, attr)
            if base_model is not None and hasattr(base_model, attr):
                return getattr(base_model, attr)
        raise RuntimeError("Could not locate the model output projection (LM head).")

    def _set_output_head(self, new_head: nn.Module, model=None) -> None:
        model = self.codi if model is None else model
        if hasattr(model, "set_output_embeddings"):
            try:
                model.set_output_embeddings(new_head)
                return
            except Exception:
                pass
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else None
        if base_model is not None and hasattr(base_model, "set_output_embeddings"):
            try:
                base_model.set_output_embeddings(new_head)
                return
            except Exception:
                pass
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                setattr(model, attr, new_head)
                return
            if base_model is not None and hasattr(base_model, attr):
                setattr(base_model, attr, new_head)
                return
        raise RuntimeError("Failed to set the model output projection (LM head).")

    def _project_with_teacher_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.superpose_kl_teacher_head is None:
            raise RuntimeError("Superpose teacher LM head is not initialized.")
        teacher_head = self.superpose_kl_teacher_head.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.superpose_kl_teacher_head = teacher_head
        return teacher_head(hidden_states)

    def _configure_superpose_kl_lm_head(self) -> None:
        """
        Optionally replace the LM head with a standalone trainable linear layer in
        superpose-KL mode.
        """
        train_lm_head = bool(getattr(self, "ccot_superpose_kl_train_lm_head", False))
        reinit_lm_head = bool(getattr(self, "ccot_superpose_kl_train_lm_head_reinit", False))
        if not (train_lm_head or reinit_lm_head):
            return
        if not bool(getattr(self, "ccot_superpose_kl", False)):
            raise RuntimeError(
                "ccot_superpose_kl_train_lm_head/ccot_superpose_kl_train_lm_head_reinit are only supported when ccot_superpose_kl=True."
            )

        old_head = self._get_output_head(self.codi)
        if old_head is None or not hasattr(old_head, "weight"):
            raise RuntimeError("The current LM head does not expose a weight matrix.")
        old_weight = old_head.weight.detach()
        if old_weight.ndim != 2:
            raise RuntimeError(f"Expected LM head weight to be 2D [vocab, hidden], got shape={tuple(old_weight.shape)}")
        vocab_size, hidden_size = old_weight.shape
        has_bias = getattr(old_head, "bias", None) is not None
        init_std = float(getattr(self.codi.config, "initializer_range", 0.02))

        teacher_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=has_bias,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            teacher_head.weight.copy_(old_weight)
            if has_bias:
                old_bias = old_head.bias.detach() if old_head.bias is not None else None
                if old_bias is not None:
                    teacher_head.bias.copy_(old_bias)
                else:
                    teacher_head.bias.zero_()
        for param in teacher_head.parameters():
            param.requires_grad = True
        self.superpose_kl_teacher_head = teacher_head

        new_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=has_bias,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            if reinit_lm_head:
                new_head.weight.normal_(mean=0.0, std=init_std)
                if has_bias:
                    new_head.bias.zero_()
            else:
                new_head.weight.copy_(old_weight)
                if has_bias:
                    old_bias = old_head.bias.detach() if old_head.bias is not None else None
                    if old_bias is not None:
                        new_head.bias.copy_(old_bias)
                    else:
                        new_head.bias.zero_()

        for param in new_head.parameters():
            param.requires_grad = True

        self._set_output_head(new_head, self.codi)
        if hasattr(self.codi, "config") and hasattr(self.codi.config, "tie_word_embeddings"):
            self.codi.config.tie_word_embeddings = False
        base_model = self.codi.get_base_model() if hasattr(self.codi, "get_base_model") else None
        if base_model is not None and hasattr(base_model, "config") and hasattr(base_model.config, "tie_word_embeddings"):
            base_model.config.tie_word_embeddings = False

    def _lookup_superpose_target_embeddings(
        self,
        token_ids: torch.LongTensor,
        emb_layer: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Lookup token embeddings for superpose target construction."""
        emb_layer = emb_layer or self.get_embd(self.codi, self.model_name)
        ids = token_ids.to(emb_layer.weight.device)
        return emb_layer(ids)

    def init(self):
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            restore_path = self.training_args.restore_from
            try:
                if restore_path.endswith(".safetensors"):
                    state_dict = load_file(restore_path)
                elif restore_path.endswith(".bin"):
                    state_dict = torch.load(restore_path, map_location="cpu")
                else:
                    safetensors_path = os.path.join(restore_path, "model.safetensors")
                    pytorch_path = os.path.join(restore_path, "pytorch_model.bin")
                    if os.path.exists(safetensors_path):
                        state_dict = load_file(safetensors_path)
                    elif os.path.exists(pytorch_path):
                        state_dict = torch.load(pytorch_path, map_location="cpu")
                    else:
                        state_dict = load_file(restore_path)
            except Exception as e:
                print(f"[Warning] Failed to load with safetensors: {e}")
                print(f"[Warning] Trying torch.load...")
                state_dict = torch.load(restore_path, map_location="cpu")

            state_dict = self._expand_state_dict_for_new_tokens(state_dict)
            self.load_state_dict(state_dict, strict=False)
            print(f"Finished loading from {self.training_args.restore_from}")

        if self.ccot_superpose_kl_train_lm_head or self.ccot_superpose_kl_train_lm_head_reinit:
            self._configure_superpose_kl_lm_head()
        print_trainable_parameters(self)

    def _expand_state_dict_for_new_tokens(self, state_dict):
        """Pad checkpoint embeddings/LM head when we added new special tokens."""
        target_rows = self.codi.get_input_embeddings().weight.size(0)
        sample_key = next((k for k in state_dict.keys() if k.endswith("embed_tokens.weight") or k.endswith("wte.weight")), None)
        if sample_key is not None and state_dict[sample_key].size(0) == target_rows:
            print("[Vocab Expansion] Checkpoint embeddings match target size; no expansion needed.")
            return state_dict
        pad_like_ids = [t for t in [self.pad_token_id, self.bot_id, self.eot_id] if t is not None]

        def _pad_rows(param, name):
            if param.size(0) == target_rows:
                return param
            if param.size(0) > target_rows:
                return param[:target_rows, ...]
            missing = target_rows - param.size(0)
            new_rows = torch.empty((missing, param.size(1)), device=param.device, dtype=param.dtype)
            new_rows.normal_(mean=0.0, std=0.02)
            for tid in pad_like_ids:
                if tid is not None and tid >= param.size(0) and tid < target_rows:
                    new_rows[tid - param.size(0)].zero_()
            return torch.cat([param, new_rows], dim=0)

        for key in list(state_dict.keys()):
            if key.endswith("embed_tokens.weight") or key.endswith("wte.weight"):
                state_dict[key] = _pad_rows(state_dict[key], key)
        for key in list(state_dict.keys()):
            if key.endswith("lm_head.weight"):
                state_dict[key] = _pad_rows(state_dict[key], key)
        return state_dict

    def _forward_ccot_default(
        self,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Default CODI path, kept aligned with the reference CODI implementation."""
        if encoder_input_ids is None or decoder_input_ids is None or labels is None:
            raise RuntimeError("CCOT forward requires encoder_input_ids, decoder_input_ids, and labels.")

        device = encoder_input_ids.device
        if encoder_attention_mask is None:
            encoder_attention_mask = encoder_input_ids.ne(self.tokenizer.pad_token_id)
        if ref_attention_mask is None and ref_input_ids is not None:
            ref_attention_mask = ref_input_ids.ne(self.tokenizer.pad_token_id)
        if not self.fix_attn_mask:
            ref_attention_mask = None

        use_codi_losses = bool(float(self.distill_loss_factor) > 0.0 or float(self.ref_loss_factor) > 0.0)
        if use_codi_losses:
            if (
                ref_input_ids is None
                or ref_labels is None
                or ref_answer_position is None
                or model_answer_position is None
            ):
                raise RuntimeError(
                    "Default CCOT path with distill/reference losses requires ref_input_ids, ref_labels, "
                    "ref_answer_position, and model_answer_position."
                )

        past_key_values = None
        prompt_outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = prompt_outputs.past_key_values
        latent_embd = prompt_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if self.use_prj:
            with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                latent_embd = self.prj(latent_embd)

        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((encoder_attention_mask.size(0), self.num_latent), device=device)

        ref_outputs = None
        ref_outputs_with_grad = None
        if use_codi_losses:
            if float(self.distill_loss_factor) > 0.0:
                with torch.no_grad():
                    ref_outputs = self.codi(
                        input_ids=ref_input_ids,
                        output_hidden_states=True,
                        attention_mask=ref_attention_mask,
                    )
            if float(self.ref_loss_factor) > 0.0:
                ref_outputs_with_grad = self.codi(
                    input_ids=ref_input_ids,
                    output_hidden_states=bool(float(self.distill_loss_factor) > 0.0),
                    attention_mask=ref_attention_mask,
                )
                if ref_outputs is None:
                    ref_outputs = ref_outputs_with_grad

        if self.training_args.print_ref_model_stats and ref_outputs is not None:
            probe_ref_position = ref_answer_position
            if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower():
                probe_ref_position = probe_ref_position + 1
            ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
            input_positions = (probe_ref_position - 1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
            ref_probs_at_positions = ref_probs.gather(1, input_positions)
            probe_positions = ref_input_ids.gather(1, probe_ref_position.unsqueeze(1)).unsqueeze(1)
            ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
            print(f"stage0: mean of the prob of the target token: {ref_probs_of_target.mean()}")

        ref_distill_position, model_distill_position = self._distill_answer_positions(
            ref_answer_position,
            model_answer_position,
        )

        ce_loss = torch.tensor(0.0, device=device)
        ce_loss_total = torch.tensor(0.0, device=device)
        distill_loss = torch.tensor(0.0, device=device)
        distill_loss_total = torch.tensor(0.0, device=device)
        logits = None
        answer_outputs = None

        for latent_idx in range(self.num_latent):
            with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                latent_outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
            past_key_values = latent_outputs.past_key_values
            latent_embd = latent_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if self.use_prj:
                with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    latent_embd = self.prj(latent_embd)

            if latent_idx == self.num_latent - 1:
                embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
                answer_mask = None
                if dynamic_mask is not None:
                    decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool, device=device)
                    answer_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1).bool()
                with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    answer_outputs = self.codi(
                        inputs_embeds=embds,
                        use_cache=True,
                        output_hidden_states=bool(float(self.distill_loss_factor) > 0.0),
                        past_key_values=past_key_values,
                        attention_mask=answer_mask,
                    )
                logits = answer_outputs.logits
                effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                target_ids = labels[:, 1:].reshape(-1)
                ce_loss = self.loss_fct(effective_logits, target_ids)
                ce_loss_total = ce_loss_total + ce_loss

                if float(self.distill_loss_factor) > 0.0:
                    if ref_outputs is None:
                        raise RuntimeError("distill_loss_factor>0 in default CCOT mode requires reference hidden states.")
                    distill_acc = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
                    for out, ref_out in zip(answer_outputs.hidden_states, ref_outputs.hidden_states):
                        ref_selected = ref_out.gather(
                            1,
                            ref_distill_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)),
                        )
                        out_selected = out.gather(
                            1,
                            model_distill_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)),
                        )
                        distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                        if self.distill_loss_div_std:
                            if self.distill_loss_type == "l2":
                                distill_loss_tmp = distill_loss_tmp / ref_selected.std()
                            distill_loss_tmp = distill_loss_tmp / ref_selected.std()
                        distill_acc = distill_acc + distill_loss_tmp
                    distill_loss = distill_acc / len(answer_outputs.hidden_states)
                    distill_loss_total = distill_loss_total + distill_loss

        ref_ce_loss = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
        if float(self.ref_loss_factor) > 0.0:
            if ref_outputs_with_grad is None:
                raise RuntimeError("ref_loss_factor>0 in default CCOT mode requires reference logits.")
            ref_logits = ref_outputs_with_grad.logits
            effective_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
            ref_target_ids = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids) * self.ref_loss_factor

        distill_loss_total = distill_loss_total * self.distill_loss_factor
        if logits is None:
            raise RuntimeError("Default CODI forward did not produce answer logits; num_latent must be > 0.")

        ce_weight = 1.0
        ce_loss_weighted = ce_loss_total * ce_weight
        total_loss = (
            ce_loss_weighted
            + distill_loss_total
            + ref_ce_loss
        )

        if self.print_loss:
            print(
                f"ccot loss={total_loss}, ce_loss={ce_loss_total}, ce_weight={ce_weight}, ce_loss_weighted={ce_loss_weighted}, "
                f"distill_loss={distill_loss_total}, ref_ce_loss={ref_ce_loss}"
            )

        return {
            "loss": total_loss,
            "logits": logits,
            "ce_loss": ce_loss_total.detach(),
            "ce_loss_weighted": ce_loss_weighted.detach(),
            "ce_curriculum_weight": float(ce_weight),
            "distill_loss": distill_loss_total.detach(),
            "ref_ce_loss": ref_ce_loss.detach(),
            "kl_loss": 0.0,
        }

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        cot_superpose_step_token_ids: Optional[torch.LongTensor] = None,
        cot_superpose_step_token_mask: Optional[torch.Tensor] = None,
        cot_superpose_kl_step_token_ids: Optional[torch.LongTensor] = None,
        cot_superpose_kl_step_token_mask: Optional[torch.Tensor] = None,
        cot_superpose_step_mask: Optional[torch.Tensor] = None,
        cot_superpose_step_counts: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_step_counts: Optional[torch.LongTensor] = None,
        ref_step_token_counts: Optional[torch.LongTensor] = None,
        ref_superpose_input_ids: Optional[torch.LongTensor] = None,
        ref_superpose_labels: Optional[torch.LongTensor] = None,
        ref_step_content_offsets: Optional[torch.LongTensor] = None,
        ref_step_content_token_counts: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
    ):
        if getattr(self.training_args, "ccot_superpose_kl", False):
            return self._forward_ccot_superpose(
                cot_superpose_step_token_ids=cot_superpose_step_token_ids,
                cot_superpose_step_token_mask=cot_superpose_step_token_mask,
                cot_superpose_kl_step_token_ids=cot_superpose_kl_step_token_ids,
                cot_superpose_kl_step_token_mask=cot_superpose_kl_step_token_mask,
                cot_superpose_step_mask=cot_superpose_step_mask,
                final_input_ids=encoder_input_ids,
                final_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                ref_answer_position=ref_answer_position,
                model_answer_position=model_answer_position,
                ref_labels=ref_labels,
                ref_step_token_counts=ref_step_token_counts,
                ref_superpose_input_ids=ref_superpose_input_ids,
                ref_superpose_labels=ref_superpose_labels,
                ref_step_content_offsets=ref_step_content_offsets,
                ref_step_content_token_counts=ref_step_content_token_counts,
                step=step,
                step_ratio=step_ratio,
            )
        if self.training_args.teacher_only:
            if not self.fix_attn_mask:
                ref_attention_mask = None
            teacher_out = self.codi(
                input_ids=ref_input_ids,
                output_hidden_states=False,
                attention_mask=ref_attention_mask,
            )
            ref_logits = teacher_out.logits
            effective_ref_logits = ref_logits[:, :-1, :]
            effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
            ref_target_ids = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids) * self.ref_loss_factor
            return {
                "loss": ref_ce_loss,
                "logits": ref_logits,
                "ce_loss": 0.0,
                "distill_loss": 0.0,
                "ref_ce_loss": float(ref_ce_loss.detach()),
            }
        return self._forward_ccot_default(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
            ref_answer_position=ref_answer_position,
            model_answer_position=model_answer_position,
            ref_labels=ref_labels,
            encoder_attention_mask=encoder_attention_mask,
            step=step,
        )

    def _kl_from_logits(self, student_logits, ref_logits, mask):
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        per_token_kl = (student_probs * (student_log_probs - ref_log_probs)).sum(dim=-1)
        if mask is not None:
            per_token_kl = per_token_kl * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (per_token_kl.sum(dim=1) / denom).mean()
        return per_token_kl.mean()

    def _kl_from_superposed_token_targets(
        self,
        student_logits: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        KL(target || student) at each latent step, where the target is the empirical
        distribution induced by the tokenized structured step.
        """
        if student_logits is None:
            raise RuntimeError("student_logits is required for superposed-token KL.")
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superposed-token KL requires target step token ids and masks in the batch.")
        if student_logits.ndim != 3:
            raise ValueError(f"Expected student_logits to be [B,K,V], got shape={tuple(student_logits.shape)}")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        device = student_logits.device
        _, num_latent_slots, vocab_size = student_logits.shape
        _, num_steps, _ = target_token_ids.shape
        aligned_steps = min(num_latent_slots, num_steps)
        if aligned_steps <= 0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        logits = student_logits[:, :aligned_steps, :]
        if temperature is not None and float(temperature) > 0.0 and float(temperature) != 1.0:
            logits = logits / float(temperature)
        log_probs = F.log_softmax(logits, dim=-1)

        ids = target_token_ids[:, :aligned_steps, :].to(device)
        token_mask = target_token_mask[:, :aligned_steps, :].to(device).to(log_probs.dtype)
        token_counts = token_mask.sum(dim=-1)

        if step_mask is not None:
            valid_step_mask = step_mask[:, :aligned_steps].to(device).to(log_probs.dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(log_probs.dtype)
        else:
            valid_step_mask = (token_counts > 0).to(log_probs.dtype)

        safe_ids = torch.where(token_mask > 0, ids, torch.zeros_like(ids))
        gathered_log_probs = log_probs.gather(dim=-1, index=safe_ids)

        ce_per_step = -((gathered_log_probs * token_mask).sum(dim=-1) / token_counts.clamp(min=1.0))

        with torch.no_grad():
            entropy_per_step = self._entropy_from_weighted_token_targets(
                token_ids=ids,
                token_weights=token_mask / token_counts.clamp(min=1.0).unsqueeze(-1),
                token_mask=token_mask,
                pad_token_id=vocab_size,
            )

        kl_per_step = (ce_per_step - entropy_per_step) * valid_step_mask
        denom = valid_step_mask.sum(dim=1).clamp(min=1.0)
        return (kl_per_step.sum(dim=1) / denom).mean()

    def _entropy_from_weighted_token_targets(
        self,
        token_ids: torch.LongTensor,
        token_weights: torch.Tensor,
        token_mask: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Exact entropy H(q) for per-step weighted token multisets.
        """
        if token_ids.ndim != 3 or token_weights.ndim != 3 or token_mask.ndim != 3:
            raise ValueError(
                "Expected token_ids/token_weights/token_mask to be [B,S,T], "
                f"got {tuple(token_ids.shape)}, {tuple(token_weights.shape)}, {tuple(token_mask.shape)}"
            )

        batch_size, num_steps, max_tokens = token_ids.shape
        if batch_size == 0 or num_steps == 0 or max_tokens == 0:
            return token_weights.new_zeros((batch_size, num_steps))

        flat_ids = token_ids.reshape(-1, max_tokens)
        flat_weights = token_weights.reshape(-1, max_tokens)
        flat_valid = token_mask.reshape(-1, max_tokens) > 0

        safe_ids = torch.where(flat_valid, flat_ids, torch.full_like(flat_ids, int(pad_token_id)))
        sorted_ids, sort_idx = safe_ids.sort(dim=-1)
        sorted_weights = flat_weights.gather(dim=-1, index=sort_idx)
        sorted_valid = sorted_ids.ne(int(pad_token_id))
        sorted_weights = sorted_weights * sorted_valid.to(sorted_weights.dtype)

        group_start = sorted_valid.clone()
        group_start[:, 1:] = group_start[:, 1:] & sorted_ids[:, 1:].ne(sorted_ids[:, :-1])
        group_index = group_start.to(torch.long).cumsum(dim=-1) - 1

        grouped_masses = torch.zeros_like(sorted_weights)
        grouped_masses.scatter_add_(dim=-1, index=group_index.clamp(min=0), src=sorted_weights)

        entropy = -(grouped_masses * torch.log(grouped_masses.clamp(min=1e-12))).sum(dim=-1)
        return entropy.view(batch_size, num_steps)

    def _superpose_scalar_position_weights(
        self,
        T: int,
        token_counts: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Build positive scalar weights w_i for positions i=0..T-1.
        """
        if T <= 0:
            return torch.zeros((*token_counts.shape, 0), device=device, dtype=dtype)
        mode = str(self.ccot_superpose_kl_scalar_positional_mode if mode is None else mode).strip().lower()
        pos_idx = torch.arange(T, device=device, dtype=dtype).view(1, 1, T)

        if mode == "geometric":
            decay = float(getattr(self, "ccot_superpose_kl_scalar_positional_decay", 0.9))
            decay = max(decay, 1e-8)
            base = torch.full((1, 1, 1), decay, device=device, dtype=dtype)
            w = torch.pow(base, pos_idx)
        elif mode == "geometric_reverse":
            decay = float(getattr(self, "ccot_superpose_kl_scalar_positional_decay", 0.9))
            decay = max(decay, 1e-8)
            base = torch.full((1, 1, 1), decay, device=device, dtype=dtype)
            rev_exp = ((token_counts.to(dtype) - 1.0).clamp(min=0.0).unsqueeze(-1) - pos_idx).clamp(min=0.0)
            w = torch.pow(base, rev_exp)
        elif mode == "sinusoidal":
            amp = float(getattr(self, "ccot_superpose_kl_scalar_positional_sin_amp", 1.0))
            denom_pos = (token_counts.to(dtype) - 1.0).clamp(min=1.0).unsqueeze(-1)
            pos_norm = pos_idx / denom_pos
            w = torch.exp(amp * torch.sin(0.5 * math.pi * pos_norm))
        elif mode in {"transformer_sinusoidal", "rope"}:
            amp = float(getattr(self, "ccot_superpose_kl_scalar_positional_sin_amp", 1.0))
            num_pairs_cfg = int(getattr(self, "ccot_superpose_kl_scalar_positional_fourier_pairs", 16))
            num_pairs = max(1, num_pairs_cfg)
            base = float(getattr(self, "ccot_superpose_kl_scalar_positional_base", 10000.0))
            base = max(base, 1.0001)
            freq_sampling = str(
                getattr(self, "ccot_superpose_kl_scalar_positional_freq_sampling", "spread")
            ).strip().lower()
            if freq_sampling not in {"prefix", "spread"}:
                freq_sampling = "spread"
            d_model_ref = int(getattr(self, "dim", 0))
            if d_model_ref <= 0:
                d_model_ref = 2 * num_pairs
            d_model_ref = max(d_model_ref, 2 * num_pairs)
            full_pairs = max(1, d_model_ref // 2)

            if num_pairs >= full_pairs:
                k_idx = torch.arange(full_pairs, device=device, dtype=dtype)
            elif freq_sampling == "prefix":
                k_idx = torch.arange(num_pairs, device=device, dtype=dtype)
            else:
                k_idx = torch.linspace(0, full_pairs - 1, steps=num_pairs, device=device, dtype=dtype)
                k_idx = torch.round(k_idx)
            inv_freq = torch.exp((-math.log(base) * (2.0 * k_idx)) / float(d_model_ref))
            angles = pos_idx.unsqueeze(-1) * inv_freq.view(1, 1, 1, num_pairs)
            sin_part = torch.sin(angles)
            cos_part = torch.cos(angles)

            if mode == "transformer_sinusoidal":
                score = (sin_part + cos_part).mean(dim=-1) / math.sqrt(2.0)
                w = torch.exp(amp * score)
            else:
                score = cos_part.mean(dim=-1)
                w = torch.exp(amp * score)
        elif mode == "learned":
            if self.superpose_kl_scalar_positional_logits is None:
                raise RuntimeError(
                    "Learned scalar positional KL mode selected but position parameter table is not initialized."
                )
            table = self.superpose_kl_scalar_positional_logits
            max_pos = int(table.numel())
            idx = torch.arange(T, device=device, dtype=torch.long).clamp(max=max_pos - 1)
            w_1d = F.softplus(table.to(device=device)[idx]).to(dtype) + torch.tensor(1e-6, device=device, dtype=dtype)
            w = w_1d.view(1, 1, T)
        else:
            raise ValueError(
                f"Unsupported scalar positional KL mode={mode!r}."
            )
        return w

    def _kl_from_superposed_token_targets_scalar_positional(
        self,
        student_logits: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Order-aware KL(target || student) using a single scalar-position-weighted target distribution.
        """
        resolved_mode = str(self.ccot_superpose_kl_scalar_positional_mode if mode is None else mode).strip().lower()
        if resolved_mode in {"geo", "exp"}:
            resolved_mode = "geometric"
        elif resolved_mode in {"sin", "sine"}:
            resolved_mode = "sinusoidal"
        elif resolved_mode in {"transformer", "transformer_sin", "tf_sinusoidal", "vaswani"}:
            resolved_mode = "transformer_sinusoidal"
        elif resolved_mode in {"rope", "rope_like", "rope-scalar"}:
            resolved_mode = "rope"
        elif resolved_mode in {"learn", "table"}:
            resolved_mode = "learned"
        elif resolved_mode in {"fb", "forward_backward", "bidirectional"}:
            resolved_mode = "fb"

        if resolved_mode == "fb":
            kl_f = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
                mode="geometric",
            )
            kl_b = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
                mode="geometric_reverse",
            )
            return 0.5 * (kl_f + kl_b)

        if student_logits is None:
            raise RuntimeError("student_logits is required for scalar positional superposed-token KL.")
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Scalar positional superposed-token KL requires target step token ids and masks in the batch.")
        if student_logits.ndim != 3:
            raise ValueError(f"Expected student_logits to be [B,K,V], got shape={tuple(student_logits.shape)}")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        device = student_logits.device
        _, num_latent_slots, vocab_size = student_logits.shape
        _, num_steps, _ = target_token_ids.shape
        aligned_steps = min(num_latent_slots, num_steps)
        if aligned_steps <= 0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        logits = student_logits[:, :aligned_steps, :]
        if temperature is not None and float(temperature) > 0.0 and float(temperature) != 1.0:
            logits = logits / float(temperature)
        log_probs = F.log_softmax(logits, dim=-1)

        ids = target_token_ids[:, :aligned_steps, :].to(device)
        token_mask = target_token_mask[:, :aligned_steps, :].to(device).to(log_probs.dtype)
        token_counts = token_mask.sum(dim=-1)

        if step_mask is not None:
            valid_step_mask = step_mask[:, :aligned_steps].to(device).to(log_probs.dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(log_probs.dtype)
        else:
            valid_step_mask = (token_counts > 0).to(log_probs.dtype)

        safe_ids = torch.where(token_mask > 0, ids, torch.zeros_like(ids))
        gathered_log_probs = log_probs.gather(dim=-1, index=safe_ids)

        T = ids.size(-1)
        pos_w = self._superpose_scalar_position_weights(
            T=T,
            token_counts=token_counts,
            dtype=log_probs.dtype,
            device=device,
            mode=resolved_mode,
        )
        raw_alpha = pos_w * token_mask
        alpha_sums = raw_alpha.sum(dim=-1)
        valid_step_mask = valid_step_mask * (alpha_sums > 0).to(log_probs.dtype)
        alpha = raw_alpha / alpha_sums.clamp(min=1e-12).unsqueeze(-1)

        ce_per_step = -(alpha * gathered_log_probs).sum(dim=-1)

        with torch.no_grad():
            entropy_per_step = self._entropy_from_weighted_token_targets(
                token_ids=ids,
                token_weights=alpha,
                token_mask=token_mask,
                pad_token_id=vocab_size,
            )

        kl_per_step = (ce_per_step - entropy_per_step) * valid_step_mask
        denom = valid_step_mask.sum(dim=1).clamp(min=1.0)
        return (kl_per_step.sum(dim=1) / denom).mean()

    def _superpose_sinusoidal_position_codes(
        self,
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        base: float = 10000.0,
    ) -> torch.Tensor:
        """Deterministic sinusoidal position codes [length, dim]."""
        if length <= 0:
            return torch.zeros((0, dim), device=device, dtype=dtype)
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        num_even = (dim + 1) // 2
        idx = torch.arange(num_even, device=device, dtype=torch.float32)
        base = max(float(base), 1.0001)
        inv_freq = torch.exp((-math.log(base) * (2.0 * idx)) / max(1.0, float(dim)))
        angles = pos * inv_freq.unsqueeze(0)
        pe = torch.zeros((length, dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
        if dim > 1:
            pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
        return pe.to(dtype)

    def _superpose_llama_rope_position_codes(
        self,
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        base: Optional[float] = None,
    ) -> torch.Tensor:
        """LLaMA-style RoPE positional code proxy [length, dim]."""
        if length <= 0:
            return torch.zeros((0, dim), device=device, dtype=dtype)
        model_theta = float(getattr(self.codi.config, "rope_theta", 10000.0))
        theta_cfg = float(0.0 if base is None else base)
        theta = model_theta if theta_cfg <= 0.0 else theta_cfg
        theta = max(theta, 1.0001)

        num_heads = int(getattr(self.codi.config, "num_attention_heads", 0))
        if num_heads > 0 and dim % num_heads == 0:
            head_dim = dim // num_heads
        else:
            head_dim = dim
        head_dim = max(2, head_dim)
        num_pairs = head_dim // 2
        if num_pairs <= 0:
            return torch.zeros((length, dim), device=device, dtype=dtype)

        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        idx = torch.arange(num_pairs, device=device, dtype=torch.float32)
        inv_freq = torch.exp((-math.log(theta) * (2.0 * idx)) / float(head_dim))
        angles = pos * inv_freq.unsqueeze(0)

        head_codes = torch.zeros((length, head_dim), device=device, dtype=torch.float32)
        head_codes[:, 0::2] = torch.cos(angles[:, : head_codes[:, 0::2].shape[1]])
        head_codes[:, 1::2] = torch.sin(angles[:, : head_codes[:, 1::2].shape[1]])

        if dim == head_dim:
            return head_codes.to(dtype)
        repeats = (dim + head_dim - 1) // head_dim
        full = head_codes.repeat(1, repeats)[:, :dim]
        return full.to(dtype)

    def _superpose_position_codes(
        self,
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mode = str(getattr(self, "ccot_superpose_poscode_type", "sinusoidal")).strip().lower()
        base_cfg = float(getattr(self, "ccot_superpose_poscode_base", 0.0))
        if mode == "llama_rope":
            return self._superpose_llama_rope_position_codes(
                length=length,
                dim=dim,
                device=device,
                dtype=dtype,
                base=base_cfg,
            )
        if base_cfg > 0.0:
            base = base_cfg
        else:
            base = 10000.0
        return self._superpose_sinusoidal_position_codes(
            length=length,
            dim=dim,
            device=device,
            dtype=dtype,
            base=base,
        )

    def _superposed_token_mean_embeddings(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean token embeddings for each structured CoT step.
        """
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superposed token mean embeddings require step token ids and masks.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        B, S, T = target_token_ids.shape
        device = target_token_ids.device
        emb_layer = self.get_embd(self.codi, self.model_name)
        emb_weight_dtype = emb_layer.weight.dtype if hasattr(emb_layer, "weight") else None
        out_dtype = out_dtype or (emb_weight_dtype if emb_weight_dtype is not None else torch.float32)

        token_mask = target_token_mask.to(device).to(out_dtype)
        token_counts = token_mask.sum(dim=-1)
        if step_mask is not None:
            valid_step_mask = step_mask.to(device).to(out_dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(out_dtype)
        else:
            valid_step_mask = (token_counts > 0).to(out_dtype)

        with torch.no_grad():
            flat_ids = target_token_ids.to(device).reshape(B * S, T)
            flat_embs = self._lookup_superpose_target_embeddings(flat_ids, emb_layer=emb_layer)
            target_embs = flat_embs.reshape(B, S, T, -1).to(out_dtype)
            target_means = (target_embs * token_mask.unsqueeze(-1)).sum(dim=2)
            target_means = target_means / token_counts.clamp(min=1.0).unsqueeze(-1)

        target_means = target_means * valid_step_mask.unsqueeze(-1)
        target_means = self._to_model_dtype(target_means)
        return target_means, valid_step_mask

    def _superposed_token_q_feedback_embeddings(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-step latent feedback embeddings from superpose vocab targets q(v).
        """
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superposed q-feedback embeddings require step token ids and masks.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        B, S, T = target_token_ids.shape
        device = target_token_ids.device
        emb_layer = self.get_embd(self.codi, self.model_name)
        emb_weight_dtype = emb_layer.weight.dtype if hasattr(emb_layer, "weight") else None
        out_dtype = out_dtype or (emb_weight_dtype if emb_weight_dtype is not None else torch.float32)

        token_mask = target_token_mask.to(device).to(out_dtype)
        token_counts = token_mask.sum(dim=-1)
        if step_mask is not None:
            valid_step_mask = step_mask.to(device).to(out_dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(out_dtype)
        else:
            valid_step_mask = (token_counts > 0).to(out_dtype)

        token_weights = token_mask / token_counts.clamp(min=1.0).unsqueeze(-1)
        with torch.no_grad():
            flat_ids = target_token_ids.to(device).reshape(B * S, T)
            flat_embs = self._lookup_superpose_target_embeddings(flat_ids, emb_layer=emb_layer)
            target_embs = flat_embs.reshape(B, S, T, -1).to(out_dtype)
            q_feedback_embeds = (target_embs * token_weights.unsqueeze(-1)).sum(dim=2)

        q_feedback_embeds = q_feedback_embeds * valid_step_mask.unsqueeze(-1)
        q_feedback_embeds = self._to_model_dtype(q_feedback_embeds)
        return q_feedback_embeds, valid_step_mask

    def _generated_q_feedback_embedding_from_logits(
        self,
        student_logits_step: torch.Tensor,
        temperature: float = 1.0,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Build latent feedback embedding from generated vocab distribution q(v)=softmax(Wz).
        """
        if student_logits_step is None:
            raise RuntimeError("student_logits_step is required for generated q-feedback.")
        if student_logits_step.ndim != 2:
            raise ValueError(
                f"Expected student_logits_step to be [B,V], got shape={tuple(student_logits_step.shape)}"
            )
        temp = float(temperature)
        if temp <= 0.0:
            raise ValueError(f"temperature must be > 0 for generated q-feedback, got {temp}")

        device = student_logits_step.device
        emb_layer = self.get_embd(self.codi, self.model_name)
        emb_weight = emb_layer.weight
        vocab_size = int(self.codi.config.vocab_size)
        emb_vocab = emb_weight[:vocab_size].to(device=device)
        out_dtype = out_dtype or emb_vocab.dtype

        logits = student_logits_step
        if temp != 1.0:
            logits = logits / temp
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        feedback = torch.matmul(probs, emb_vocab.to(torch.float32))
        feedback = feedback.to(out_dtype).unsqueeze(1)
        feedback = self._to_model_dtype(feedback)
        return feedback

    def _superpose_aux_num_slots(self) -> int:
        """Number of auxiliary latent-step supervision slots used in superpose modes."""
        base_slots = max(0, int(self.num_latent))
        if bool(getattr(self, "ccot_superpose_k_plus_one_slots", False)):
            return base_slots + 1
        return base_slots

    def _limit_superpose_step_mask(
        self,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor],
        max_supervised_slots: int,
    ) -> torch.Tensor:
        """
        Keep only the first `max_supervised_slots` valid superpose steps.
        """
        if target_token_mask is None:
            raise RuntimeError("target_token_mask is required to limit superpose step mask.")
        if target_token_mask.ndim != 3:
            raise ValueError(
                f"Expected target_token_mask to be [B,S,T], got shape={tuple(target_token_mask.shape)}"
            )
        if max_supervised_slots < 0:
            raise ValueError(f"max_supervised_slots must be >= 0, got {max_supervised_slots}")

        device = target_token_mask.device
        token_mask = target_token_mask.to(device)
        valid_from_tokens = token_mask.to(torch.float32).sum(dim=-1) > 0

        if step_mask is not None:
            if step_mask.ndim != 2:
                raise ValueError(f"Expected step_mask to be [B,S], got shape={tuple(step_mask.shape)}")
            base_mask = step_mask.to(device) > 0
            out_dtype = step_mask.dtype
        else:
            base_mask = torch.ones_like(valid_from_tokens, dtype=torch.bool, device=device)
            mask_dtype = token_mask.dtype
            out_dtype = mask_dtype if (mask_dtype == torch.bool or mask_dtype.is_floating_point) else torch.float32

        limited_mask = base_mask & valid_from_tokens
        if max_supervised_slots < limited_mask.size(1):
            limited_mask = limited_mask.clone()
            limited_mask[:, max_supervised_slots:] = False
        return limited_mask.to(out_dtype)

    def _chunk_superpose_steps_to_latents(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
        randomize: bool = False,
        chunk_sizes: Optional[list[list[int]]] = None,
        return_chunk_sizes: bool = False,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Chunk structured CoT steps into latent-aligned contiguous groups.
        """
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superpose chunking requires target_token_ids and target_token_mask.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        K = int(self.num_latent if num_latent is None else num_latent)
        if K < 0:
            raise ValueError(f"num_latent must be non-negative, got {K}")

        output_device = target_token_ids.device
        work_device = torch.device("cpu") if target_token_ids.is_cuda else output_device
        ids = target_token_ids.detach().to(work_device)
        tok_mask_in = target_token_mask.detach().to(work_device)
        tok_mask_bool = tok_mask_in.bool()
        B, S, T = ids.shape

        mask_dtype = tok_mask_in.dtype if (tok_mask_in.dtype.is_floating_point or tok_mask_in.dtype == torch.bool) else torch.float32
        token_counts = tok_mask_bool.sum(dim=-1)
        valid_steps = token_counts > 0
        if step_mask is not None:
            valid_steps = valid_steps & step_mask.detach().to(work_device).bool()

        if K == 0:
            chunk_ids = torch.zeros((B, 0, 0), device=work_device, dtype=ids.dtype)
            chunk_tok_mask = torch.zeros((B, 0, 0), device=work_device, dtype=mask_dtype)
            chunk_step_mask = torch.zeros((B, 0), device=work_device, dtype=mask_dtype)
            if work_device != output_device:
                chunk_ids = chunk_ids.to(output_device, non_blocking=True)
                chunk_tok_mask = chunk_tok_mask.to(output_device, non_blocking=True)
                chunk_step_mask = chunk_step_mask.to(output_device, non_blocking=True)
            return chunk_ids, chunk_tok_mask, chunk_step_mask

        def _chunk_sizes(m: int, k: int, sample_random: bool) -> list[int]:
            if m <= 0:
                return [0] * k
            if m <= k:
                return [1] * m + [0] * (k - m)
            if sample_random:
                if k <= 1:
                    return [m]
                cut_points = torch.randperm(m - 1, device=work_device)[: k - 1] + 1
                cut_points, _ = torch.sort(cut_points)
                prev = 0
                sizes = []
                for cut in cut_points.tolist():
                    sizes.append(int(cut - prev))
                    prev = int(cut)
                sizes.append(int(m - prev))
                return sizes
            base = m // k
            rem = m % k
            return [base] * (k - rem) + [base + 1] * rem

        if chunk_sizes is not None and len(chunk_sizes) != B:
            raise ValueError(
                f"Expected chunk_sizes to provide one plan per batch item: {len(chunk_sizes)} != {B}"
            )

        all_chunk_tokens: list[list[torch.Tensor]] = []
        resolved_chunk_sizes: list[list[int]] = []
        max_chunk_len = 0
        for b in range(B):
            valid_idx = torch.nonzero(valid_steps[b], as_tuple=False).squeeze(-1).tolist()
            m = len(valid_idx)
            sizes = list(chunk_sizes[b]) if chunk_sizes is not None else _chunk_sizes(m, K, bool(randomize))
            if len(sizes) != K:
                raise ValueError(
                    f"Chunk plan length must equal num_latent={K}, got {len(sizes)} for batch item {b}"
                )
            size_sum = sum(int(size) for size in sizes)
            if size_sum != m:
                raise ValueError(
                    f"Chunk plan must cover all valid steps for batch item {b}: sum={size_sum} != {m}"
                )
            resolved_chunk_sizes.append([int(size) for size in sizes])
            cursor = 0
            sample_chunks: list[torch.Tensor] = []
            for size in sizes:
                if size <= 0:
                    chunk = ids.new_empty((0,))
                    sample_chunks.append(chunk)
                    continue
                parts = []
                for _ in range(size):
                    s_idx = int(valid_idx[cursor])
                    cursor += 1
                    n_tok = int(token_counts[b, s_idx].item())
                    if n_tok > 0:
                        parts.append(ids[b, s_idx, :n_tok])
                if parts:
                    chunk = torch.cat(parts, dim=0)
                else:
                    chunk = ids.new_empty((0,))
                max_chunk_len = max(max_chunk_len, int(chunk.numel()))
                sample_chunks.append(chunk)
            all_chunk_tokens.append(sample_chunks)

        padded_len = max(1, int(max_chunk_len))
        chunk_ids = torch.zeros((B, K, padded_len), device=work_device, dtype=ids.dtype)
        if mask_dtype == torch.bool:
            chunk_tok_mask = torch.zeros((B, K, padded_len), device=work_device, dtype=torch.bool)
            chunk_step_mask = torch.zeros((B, K), device=work_device, dtype=torch.bool)
        else:
            chunk_tok_mask = torch.zeros((B, K, padded_len), device=work_device, dtype=mask_dtype)
            chunk_step_mask = torch.zeros((B, K), device=work_device, dtype=mask_dtype)

        for b in range(B):
            for k in range(K):
                chunk = all_chunk_tokens[b][k]
                n = int(chunk.numel())
                if n <= 0:
                    continue
                chunk_ids[b, k, :n] = chunk
                if chunk_tok_mask.dtype == torch.bool:
                    chunk_tok_mask[b, k, :n] = True
                    chunk_step_mask[b, k] = True
                else:
                    chunk_tok_mask[b, k, :n] = 1
                    chunk_step_mask[b, k] = 1

        if work_device != output_device:
            chunk_ids = chunk_ids.to(output_device, non_blocking=True)
            chunk_tok_mask = chunk_tok_mask.to(output_device, non_blocking=True)
            chunk_step_mask = chunk_step_mask.to(output_device, non_blocking=True)

        if return_chunk_sizes:
            return chunk_ids, chunk_tok_mask, chunk_step_mask, resolved_chunk_sizes
        return chunk_ids, chunk_tok_mask, chunk_step_mask

    def _build_latent_prefix(
        self,
        input_ids,
        attention_mask,
        step: Optional[int] = None,
        return_latent_pred_logits: bool = False,
        return_latent_hidden_states: bool = False,
        return_latent_hidden_states_all_layers: bool = False,
        include_final_aux_slot: bool = False,
        generated_q_feedback: bool = False,
        generated_q_feedback_temperature: Optional[float] = None,
    ):
        """
        Encode prompt and latent chain to obtain past_key_values and latent embeddings.
        """
        outputs = self.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values
        latent_hidden = outputs.hidden_states[-1][:, -1, :]
        latent_embd, _ = self._prepare_latent_input(
            latent_hidden,
            step=step,
        )

        latent_tokens = []
        last_logits = outputs.logits if hasattr(outputs, "logits") else None
        latent_pred_logits = []
        latent_hidden_states = []
        latent_hidden_states_all_layers = []
        if return_latent_pred_logits and last_logits is not None and (self.num_latent > 0 or include_final_aux_slot):
            latent_pred_logits.append(last_logits[:, -1, : self.codi.config.vocab_size])
        if return_latent_hidden_states and (self.num_latent > 0 or include_final_aux_slot):
            latent_hidden_states.append(latent_hidden)
        if return_latent_hidden_states_all_layers and (self.num_latent > 0 or include_final_aux_slot):
            prompt_slot_layers = torch.stack([h[:, -1, :] for h in outputs.hidden_states], dim=1)
            latent_hidden_states_all_layers.append(prompt_slot_layers)

        use_generated_q_feedback = bool(generated_q_feedback)
        generated_q_temp = float(
            self.ccot_kl_temperature
            if generated_q_feedback_temperature is None
            else generated_q_feedback_temperature
        )
        if use_generated_q_feedback and generated_q_temp <= 0.0:
            raise ValueError(f"generated_q_feedback_temperature must be > 0, got {generated_q_temp}")

        for latent_idx in range(self.num_latent):
            latent_input_embd = latent_embd
            if use_generated_q_feedback:
                if last_logits is None:
                    raise RuntimeError("Generated q-feedback requires logits for each latent slot.")
                logits_step = last_logits[:, -1, : self.codi.config.vocab_size]
                generated_feedback = self._generated_q_feedback_embedding_from_logits(
                    student_logits_step=logits_step,
                    temperature=generated_q_temp,
                    out_dtype=latent_embd.dtype,
                )
                latent_input_embd = self._to_model_dtype(generated_feedback)

            latent_out = self.codi(
                inputs_embeds=latent_input_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_out.past_key_values
            latent_tokens.append(latent_input_embd)
            latent_hidden = latent_out.hidden_states[-1][:, -1, :]
            if return_latent_hidden_states and (
                include_final_aux_slot or latent_idx < (self.num_latent - 1)
            ):
                latent_hidden_states.append(latent_hidden)
            if return_latent_hidden_states_all_layers and (
                include_final_aux_slot or latent_idx < (self.num_latent - 1)
            ):
                slot_layers = torch.stack([h[:, -1, :] for h in latent_out.hidden_states], dim=1)
                latent_hidden_states_all_layers.append(slot_layers)
            latent_embd, _ = self._prepare_latent_input(
                latent_hidden,
                step=step,
            )
            last_logits = latent_out.logits
            if return_latent_pred_logits and (
                include_final_aux_slot or latent_idx < (self.num_latent - 1)
            ):
                latent_pred_logits.append(last_logits[:, -1, : self.codi.config.vocab_size])

        latent_prefix = torch.cat(latent_tokens, dim=1) if latent_tokens else None
        var_loss = torch.tensor(0.0, device=input_ids.device)
        if return_latent_pred_logits or return_latent_hidden_states or return_latent_hidden_states_all_layers:
            extras = []
            if return_latent_pred_logits:
                if latent_pred_logits:
                    latent_pred_logits = torch.stack(latent_pred_logits, dim=1)
                else:
                    latent_pred_logits = torch.empty(
                        (input_ids.size(0), 0, self.codi.config.vocab_size),
                        device=input_ids.device,
                        dtype=last_logits.dtype if last_logits is not None else latent_embd.dtype,
                    )
                extras.append(latent_pred_logits)
            if return_latent_hidden_states:
                if latent_hidden_states:
                    latent_hidden_states = torch.stack(latent_hidden_states, dim=1)
                else:
                    latent_hidden_states = torch.empty(
                        (input_ids.size(0), 0, latent_embd.size(-1)),
                        device=input_ids.device,
                        dtype=latent_embd.dtype,
                    )
                extras.append(latent_hidden_states)
            if return_latent_hidden_states_all_layers:
                if latent_hidden_states_all_layers:
                    latent_hidden_states_all_layers = torch.stack(latent_hidden_states_all_layers, dim=1)
                else:
                    num_layers = len(outputs.hidden_states)
                    latent_hidden_states_all_layers = torch.empty(
                        (input_ids.size(0), 0, num_layers, latent_embd.size(-1)),
                        device=input_ids.device,
                        dtype=latent_embd.dtype,
                    )
                extras.append(latent_hidden_states_all_layers)
            return (past_key_values, latent_prefix, last_logits, var_loss, *extras)
        return past_key_values, latent_prefix, last_logits, var_loss

    def _forward_ccot_superpose(
        self,
        final_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_labels: Optional[torch.LongTensor] = None,
        ref_step_token_counts: Optional[torch.LongTensor] = None,
        ref_superpose_input_ids: Optional[torch.LongTensor] = None,
        ref_superpose_labels: Optional[torch.LongTensor] = None,
        ref_step_content_offsets: Optional[torch.LongTensor] = None,
        ref_step_content_token_counts: Optional[torch.LongTensor] = None,
        cot_superpose_step_token_ids: Optional[torch.LongTensor] = None,
        cot_superpose_step_token_mask: Optional[torch.Tensor] = None,
        cot_superpose_kl_step_token_ids: Optional[torch.LongTensor] = None,
        cot_superpose_kl_step_token_mask: Optional[torch.Tensor] = None,
        cot_superpose_step_mask: Optional[torch.Tensor] = None,
        final_attention_mask: Optional[torch.LongTensor] = None,
        step: Optional[int] = None,
        step_ratio: Optional[float] = None,
    ):
        use_superpose_kl = bool(getattr(self.training_args, "ccot_superpose_kl", False))
        if not use_superpose_kl:
            raise RuntimeError("_forward_ccot_superpose requires ccot_superpose_kl=True.")
        use_superpose_k_plus_one_slots = bool(
            getattr(self, "ccot_superpose_k_plus_one_slots", False)
        )
        superpose_aux_slots = self._superpose_aux_num_slots()
        requested_uniform_chunk = bool(getattr(self.training_args, "ccot_superpose_uniform_chunk_to_latents", False))
        requested_random_chunk = bool(getattr(self.training_args, "ccot_superpose_random_chunk_to_latents", False))
        superpose_num_supervised_latents = int(
            getattr(
                self,
                "ccot_superpose_num_supervised_latents",
                getattr(self.training_args, "ccot_superpose_num_supervised_latents", 0),
            )
        )
        if superpose_num_supervised_latents < 0:
            raise ValueError("ccot_superpose_num_supervised_latents must be >= 0.")
        superpose_num_supervised_latents = min(
            superpose_num_supervised_latents,
            int(superpose_aux_slots),
        )
        disable_hard_chunking_for_supervised_prefix = (
            superpose_num_supervised_latents > 0
        )
        use_uniform_chunking = requested_uniform_chunk and not disable_hard_chunking_for_supervised_prefix
        use_random_chunking = requested_random_chunk and not disable_hard_chunking_for_supervised_prefix

        if final_attention_mask is None and final_input_ids is not None:
            final_attention_mask = final_input_ids.ne(self.tokenizer.pad_token_id)
        if ref_attention_mask is None and ref_input_ids is not None:
            ref_attention_mask = ref_input_ids.ne(self.tokenizer.pad_token_id)

        device = final_input_ids.device
        use_superpose_codi_losses = bool(
            float(self.distill_loss_factor) > 0.0 or float(self.ref_loss_factor) > 0.0
        )
        if use_superpose_codi_losses:
            if (
                ref_input_ids is None
                or ref_labels is None
                or ref_attention_mask is None
                or ref_answer_position is None
                or model_answer_position is None
            ):
                raise RuntimeError(
                    "Superpose + CODI losses require ref_input_ids, ref_labels, ref_attention_mask, "
                    "ref_answer_position, and model_answer_position in the batch."
                )
        beta_init = getattr(self.training_args, "ccot_superpose_beta", 0.1)
        beta_min = getattr(self.training_args, "ccot_superpose_beta_min", 0.03)
        sr = 0.0 if step_ratio is None else float(step_ratio)
        sr = max(0.0, min(1.0, sr))
        beta = beta_init - (beta_init - beta_min) * sr

        if cot_superpose_step_token_ids is None or cot_superpose_step_token_mask is None:
            raise RuntimeError(
                "Superpose latent supervision requires cot_superpose_step_token_ids and cot_superpose_step_token_mask in the batch."
            )
        kl_target_token_ids = cot_superpose_kl_step_token_ids
        kl_target_token_mask = cot_superpose_kl_step_token_mask
        if kl_target_token_ids is None or kl_target_token_mask is None:
            kl_target_token_ids = cot_superpose_step_token_ids
            kl_target_token_mask = cot_superpose_step_token_mask
        kl_target_step_mask = cot_superpose_step_mask
        if (
            disable_hard_chunking_for_supervised_prefix
            and (requested_uniform_chunk or requested_random_chunk)
            and not bool(getattr(self, "_superpose_prefix_supervision_chunking_warned", False))
        ):
            print(
                "[SUPERPOSE-CCOT MODE] ccot_superpose_num_supervised_latents>0: "
                "disabling hard chunking to preserve no-chunking latent/step alignment."
            )
            self._superpose_prefix_supervision_chunking_warned = True
        chunk_sizes = None
        kl_has_separate_targets = kl_target_token_ids is not cot_superpose_step_token_ids
        if use_uniform_chunking or use_random_chunking:
            chunk_out = self._chunk_superpose_steps_to_latents(
                target_token_ids=cot_superpose_step_token_ids,
                target_token_mask=cot_superpose_step_token_mask,
                step_mask=cot_superpose_step_mask,
                num_latent=superpose_aux_slots,
                randomize=use_random_chunking,
                return_chunk_sizes=True,
            )
            (
                cot_superpose_step_token_ids,
                cot_superpose_step_token_mask,
                cot_superpose_step_mask,
                chunk_sizes,
            ) = chunk_out
            if kl_has_separate_targets:
                kl_target_token_ids, kl_target_token_mask, kl_target_step_mask = self._chunk_superpose_steps_to_latents(
                    target_token_ids=kl_target_token_ids,
                    target_token_mask=kl_target_token_mask,
                    step_mask=kl_target_step_mask,
                    num_latent=superpose_aux_slots,
                    randomize=False,
                    chunk_sizes=chunk_sizes,
                )
            else:
                kl_target_token_ids = cot_superpose_step_token_ids
                kl_target_token_mask = cot_superpose_step_token_mask
                kl_target_step_mask = cot_superpose_step_mask
        if superpose_num_supervised_latents > 0:
            limited_step_mask = self._limit_superpose_step_mask(
                target_token_mask=cot_superpose_step_token_mask,
                step_mask=cot_superpose_step_mask,
                max_supervised_slots=superpose_num_supervised_latents,
            )
            cot_superpose_step_mask = limited_step_mask
            kl_target_step_mask = limited_step_mask.to(
                device=limited_step_mask.device,
                dtype=kl_target_step_mask.dtype if kl_target_step_mask is not None else limited_step_mask.dtype,
            )
        superpose_target_q_feedback = bool(getattr(self.training_args, "ccot_superpose_kl_target_q_feedback", False))
        superpose_q_feedback_train_source = str(
            getattr(self, "ccot_superpose_kl_q_feedback_train_source", "target")
        ).strip().lower()
        use_generated_q_feedback_train = bool(
            use_superpose_kl
            and superpose_target_q_feedback
            and superpose_q_feedback_train_source == "generated"
        )
        if superpose_target_q_feedback and not use_superpose_kl:
            raise RuntimeError("ccot_superpose_kl_target_q_feedback=True is only supported with ccot_superpose_kl=True.")

        past_kv_with_grad, latent_prefix_pass1, last_logits, var_loss_prefix, latent_step_pred_logits = self._build_latent_prefix(
            input_ids=final_input_ids,
            attention_mask=final_attention_mask,
            step=step,
            return_latent_pred_logits=True,
            include_final_aux_slot=use_superpose_k_plus_one_slots,
            generated_q_feedback=use_generated_q_feedback_train,
            generated_q_feedback_temperature=float(getattr(self.training_args, "ccot_kl_temperature", 1.0)),
        )
        if last_logits is None:
            raise RuntimeError("Superpose CCOT requires logits from latent pass.")

        aux_loss_raw = None
        superpose_kl_bag_raw = None
        superpose_kl_raw_total = None
        superpose_kl_scalar_positional_raw = None
        use_superpose_kl_bag = bool(getattr(self.training_args, "ccot_superpose_kl_use_bag", True))
        superpose_kl_scalar_positional_weight = float(getattr(self.training_args, "ccot_superpose_kl_scalar_positional_weight", 1.0))
        superpose_kl_component_weight = float(getattr(self, "ccot_superpose_kl_weight", 1.0))
        use_superpose_kl_scalar_positional = bool(getattr(self.training_args, "ccot_superpose_kl_scalar_positional", False))
        ref_outputs = None
        ref_outputs_with_grad = None
        if use_superpose_codi_losses:
            if self.ref_loss_factor > 0.0:
                ref_outputs_with_grad = self.codi(
                    input_ids=ref_input_ids,
                    output_hidden_states=True,
                    attention_mask=ref_attention_mask,
                )
                if self.distill_loss_factor > 0.0:
                    ref_outputs = ref_outputs_with_grad
            elif self.distill_loss_factor > 0.0:
                with torch.no_grad():
                    ref_outputs = self.codi(
                        input_ids=ref_input_ids,
                        output_hidden_states=True,
                        attention_mask=ref_attention_mask,
                    )

        if (
            (not use_superpose_kl_bag)
            and (not use_superpose_kl_scalar_positional)
        ):
            raise RuntimeError(
                "ccot_superpose_kl=True but all superpose KL components are disabled "
                "(ccot_superpose_kl_use_bag=False, "
                "ccot_superpose_kl_scalar_positional=False). "
                "Enable at least one KL component."
            )
        if use_superpose_kl_bag:
            superpose_kl_bag_raw = self._kl_from_superposed_token_targets(
                student_logits=latent_step_pred_logits,
                target_token_ids=kl_target_token_ids,
                target_token_mask=kl_target_token_mask,
                step_mask=kl_target_step_mask,
                temperature=float(getattr(self.training_args, "ccot_kl_temperature", 1.0)),
            )
            superpose_kl_raw_total = superpose_kl_bag_raw
        else:
            superpose_kl_raw_total = torch.tensor(0.0, device=device, dtype=latent_step_pred_logits.dtype)
        if use_superpose_kl_scalar_positional:
            superpose_kl_scalar_positional_raw = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=latent_step_pred_logits,
                target_token_ids=kl_target_token_ids,
                target_token_mask=kl_target_token_mask,
                step_mask=kl_target_step_mask,
                temperature=float(getattr(self.training_args, "ccot_kl_temperature", 1.0)),
                mode=str(getattr(self.training_args, "ccot_superpose_kl_scalar_positional_mode", "geometric")),
            )
            superpose_kl_raw_total = superpose_kl_raw_total + superpose_kl_scalar_positional_weight * superpose_kl_scalar_positional_raw

        superpose_kl_raw_effective = superpose_kl_component_weight * superpose_kl_raw_total
        aux_loss_raw = superpose_kl_raw_effective

        # === PASS 2: answer CE ===
        past_kv_final = past_kv_with_grad
        latent_prefix_final = latent_prefix_pass1

        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((final_input_ids.size(0), self.num_latent), device=device)

        embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
        if dynamic_mask is not None:
            decoder_mask = torch.ones(
                (embds.size(0), embds.size(1)),
                dtype=torch.bool,
                device=device,
            )
            full_mask = torch.cat((final_attention_mask, dynamic_mask, decoder_mask), dim=1).bool()
        else:
            full_mask = None

        answer_outputs = self.codi(
            inputs_embeds=embds,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv_final,
            attention_mask=full_mask,
        )
        logits = answer_outputs.logits
        effective_logits = logits[:, :-1, :]
        target_ids = labels[:, 1:]
        ce_loss = self.loss_fct(effective_logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        distill_loss = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
        distill_loss_total = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
        if self.distill_loss_factor > 0.0:
            teacher_outputs = ref_outputs if ref_outputs is not None else ref_outputs_with_grad
            if teacher_outputs is None:
                raise RuntimeError("distill_loss_factor>0 in superpose mode requires reference hidden states.")
            distill_acc = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
            ref_distill_position, model_distill_position = self._distill_answer_positions(
                ref_answer_position,
                model_answer_position,
            )
            for out, ref_out in zip(answer_outputs.hidden_states, teacher_outputs.hidden_states):
                ref_selected = ref_out.gather(
                    1,
                    ref_distill_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)),
                )
                out_selected = out.gather(
                    1,
                    model_distill_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)),
                )
                distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                if self.distill_loss_div_std:
                    std = ref_selected.detach().float().std(unbiased=False).clamp_min(1e-6)
                    distill_loss_tmp = distill_loss_tmp / std
                distill_acc = distill_acc + distill_loss_tmp.to(distill_acc.dtype)
            distill_loss = distill_acc / max(len(answer_outputs.hidden_states), 1)
            distill_loss_total = distill_loss * self.distill_loss_factor
        ref_ce_loss = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
        if self.ref_loss_factor > 0.0:
            if ref_outputs_with_grad is None:
                raise RuntimeError("ref_loss_factor>0 in superpose mode requires reference logits.")
            if bool(getattr(self, "ccot_superpose_kl_train_lm_head", False)) and self.superpose_kl_teacher_head is not None:
                ref_logits = self._project_with_teacher_head(ref_outputs_with_grad.hidden_states[-1])
            else:
                ref_logits = ref_outputs_with_grad.logits
            effective_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
            ref_target_ids = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids) * self.ref_loss_factor

        if aux_loss_raw is None:
            aux_loss_raw = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)
        aux_loss_weighted = beta * aux_loss_raw

        ce_weight = 1.0
        ce_loss_weighted = ce_loss * ce_weight
        total_loss = (
            ce_loss_weighted
            + aux_loss_weighted
            + distill_loss_total
            + ref_ce_loss
        )

        if self.print_loss:
            mode_tag = "superpose-ccot-kl"
            extra_superpose_kl_bits = ""
            if superpose_kl_bag_raw is not None:
                extra_superpose_kl_bits += f", superpose_kl_bag={superpose_kl_bag_raw}"
            if superpose_kl_scalar_positional_raw is not None:
                extra_superpose_kl_bits += (
                    f", superpose_kl_scalar_pos={superpose_kl_scalar_positional_raw}"
                    f", superpose_kl_scalar_pos_w={superpose_kl_scalar_positional_weight}"
                    f", superpose_kl_scalar_pos_mode={getattr(self.training_args, 'ccot_superpose_kl_scalar_positional_mode', 'geometric')}"
                )
            if superpose_kl_raw_total is not None:
                extra_superpose_kl_bits += (
                    f", superpose_kl_w={superpose_kl_component_weight}"
                    f", superpose_kl_weighted={superpose_kl_raw_effective}"
                )
            print(
                f"{mode_tag} loss={total_loss}, ce_loss={ce_loss}, ce_weight={ce_weight}, ce_loss_weighted={ce_loss_weighted}, "
                f"distill_loss={distill_loss_total}, ref_ce_loss={ref_ce_loss}, "
                f"superpose_kl={aux_loss_raw}, beta={beta}"
                f"{extra_superpose_kl_bits}"
            )

        def _detach_metric(x):
            return x.detach() if torch.is_tensor(x) else x

        kl_raw_total_log = _detach_metric(superpose_kl_raw_total) if superpose_kl_raw_total is not None else None
        kl_raw_effective_log = _detach_metric(superpose_kl_raw_effective) if superpose_kl_raw_effective is not None else None

        outputs = {
            "loss": total_loss,
            "logits": logits,
            "ce_loss": _detach_metric(ce_loss),
            "ce_loss_weighted": _detach_metric(ce_loss_weighted),
            "ce_curriculum_weight": float(ce_weight),
            "distill_loss": _detach_metric(distill_loss_total),
            "ref_ce_loss": _detach_metric(ref_ce_loss),
        }
        if kl_raw_total_log is not None:
            outputs["kl_loss"] = kl_raw_effective_log * float(beta)
            outputs["superpose_kl_raw"] = kl_raw_total_log
            outputs["superpose_kl_weighted_raw"] = kl_raw_effective_log
            outputs["superpose_kl_weight"] = float(superpose_kl_component_weight)
            if superpose_kl_bag_raw is not None:
                outputs["superpose_kl_bag_raw"] = _detach_metric(superpose_kl_bag_raw)
            if superpose_kl_scalar_positional_raw is not None:
                scalar_pos_raw_log = _detach_metric(superpose_kl_scalar_positional_raw)
                outputs["superpose_kl_scalar_positional_raw"] = scalar_pos_raw_log
                outputs["superpose_kl_scalar_positional_weight"] = float(superpose_kl_scalar_positional_weight)
                outputs["superpose_kl_scalar_positional_loss"] = (
                    scalar_pos_raw_log * float(beta) * float(superpose_kl_scalar_positional_weight)
                )
        else:
            outputs["kl_loss"] = 0.0
        return outputs

    @torch.no_grad()
    def _try_parse_number(self, text: str) -> Optional[float]:
        """Extract the last numeric value from text; return None if not found."""
        import re as _re
        matches = _re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    @torch.no_grad()
    def _single_latent_decode(
        self,
        question: str,
        max_new_tokens: int = 128,
        superpose_target_token_ids: Optional[torch.LongTensor] = None,
        superpose_target_token_mask: Optional[torch.Tensor] = None,
        superpose_step_mask: Optional[torch.Tensor] = None,
    ):
        """
        Greedy decode one answer. Returns (answer_text, answer_logprob).
        """
        if getattr(self.training_args, "teacher_only", False):
            raise RuntimeError("generate_with_latent_sampling expects student/CCOT model (teacher_only=False).")
        device = next(self.parameters()).device
        tok = self.tokenizer

        batch = tok(question, return_tensors="pt", padding=False)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if getattr(self.training_args, "remove_eos", False):
            bot_tensor = torch.tensor([self.bot_id], dtype=torch.long, device=device).unsqueeze(0)
        else:
            bot_tensor = torch.tensor([tok.eos_token_id, self.bot_id], dtype=torch.long, device=device).unsqueeze(0)
        bot_tensor = bot_tensor.expand(input_ids.size(0), -1)
        input_ids = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

        outputs = self.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values
        latent_hidden = outputs.hidden_states[-1][:, -1, :]
        latent_embd, _ = self._prepare_latent_input(
            latent_hidden,
            step=None,
        )

        q_feedback_enabled = bool(
            getattr(self.training_args, "ccot_superpose_kl", False)
            and getattr(self.training_args, "ccot_superpose_kl_target_q_feedback", False)
        )
        q_feedback_temp = float(getattr(self.training_args, "ccot_kl_temperature", 1.0))
        if q_feedback_enabled and q_feedback_temp <= 0.0:
            raise ValueError(f"ccot_kl_temperature must be > 0 for generated q-feedback, got {q_feedback_temp}")
        last_logits = outputs.logits

        inf_latent_iterations = getattr(self.training_args, "inf_latent_iterations", self.num_latent)
        for latent_idx in range(inf_latent_iterations):
            latent_input_embd = latent_embd
            if q_feedback_enabled:
                if last_logits is None:
                    raise RuntimeError("Generated q-feedback requires logits for each latent slot.")
                logits_step = last_logits[:, -1, : self.codi.config.vocab_size]
                forced_step = self._generated_q_feedback_embedding_from_logits(
                    student_logits_step=logits_step,
                    temperature=q_feedback_temp,
                    out_dtype=latent_embd.dtype,
                )
                latent_input_embd = forced_step
                latent_input_embd = self._to_model_dtype(latent_input_embd)
            latent_out = self.codi(
                inputs_embeds=latent_input_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_out.past_key_values
            latent_hidden = latent_out.hidden_states[-1][:, -1, :]
            latent_embd, _ = self._prepare_latent_input(
                latent_hidden,
                step=None,
            )
            last_logits = latent_out.logits

        eos_tok = tok.eos_token_id if tok.eos_token_id is not None else self.pad_token_id
        if getattr(self.training_args, "remove_eos", False):
            start_tokens = torch.tensor([self.eot_id], dtype=torch.long, device=device)
        else:
            start_tokens = torch.tensor([self.eot_id, eos_tok], dtype=torch.long, device=device)
        output = self.get_embd(self.codi, self.model_name)(start_tokens).unsqueeze(0).to(device)
        output = output.expand(input_ids.size(0), -1, -1)

        generated_tokens = []
        logprob_sum = 0.0

        for _ in range(max_new_tokens):
            out = self.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : self.codi.config.vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1)
            token_logprob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            logprob_sum += float(token_logprob.mean().detach())

            token_id = next_token.item()
            generated_tokens.append(token_id)
            if tok.eos_token_id is not None and token_id == tok.eos_token_id:
                break

            output = self.get_embd(self.codi, self.model_name)(next_token).unsqueeze(1).to(device)

        decoded = tok.decode(generated_tokens, skip_special_tokens=True)
        return decoded, logprob_sum

    @torch.no_grad()
    def generate_with_latent_sampling(
        self,
        question: str,
        max_new_tokens: int = 128,
        K: Optional[int] = None,
        superpose_target_token_ids: Optional[torch.LongTensor] = None,
        superpose_target_token_mask: Optional[torch.Tensor] = None,
        superpose_step_mask: Optional[torch.Tensor] = None,
    ):
        """
        Decode answers. Returns dict with all answers, logprobs, and selected final_answer.
        """
        was_training = self.training
        self.eval()
        K = K or 1

        answers = []
        logprobs = []
        for _ in range(K):
            ans, lp = self._single_latent_decode(
                question=question,
                max_new_tokens=max_new_tokens,
                superpose_target_token_ids=superpose_target_token_ids,
                superpose_target_token_mask=superpose_target_token_mask,
                superpose_step_mask=superpose_step_mask,
            )
            answers.append(ans)
            logprobs.append(lp)

        numeric_vals = [self._try_parse_number(a) for a in answers]
        final_answer = None
        if all(v is not None for v in numeric_vals):
            counts = Counter(numeric_vals)
            final_val, _ = counts.most_common(1)[0]
            for a, v in zip(answers, numeric_vals):
                if v == final_val:
                    final_answer = a
                    break
        else:
            best_idx = int(torch.tensor(logprobs).argmax().item())
            final_answer = answers[best_idx]

        if was_training:
            self.train()

        return {
            "answers": answers,
            "logprobs": logprobs,
            "final_answer": final_answer,
        }
