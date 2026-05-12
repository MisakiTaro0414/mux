#!/usr/bin/env python3
import argparse
import json
import logging
from itertools import batched
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModel
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.utils import cached_file

import models


LOGGER = logging.getLogger("pccot.test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PCCoT checkpoint from Hugging Face or a local path.")
    parser.add_argument("--model_name_or_path", required=True, help="HF repo id or local checkpoint path.")
    parser.add_argument("--dataset_name", default="whynlp/gsm8k-aug", help="Dataset name to evaluate on.")
    parser.add_argument("--dataset_config_name", default=None, help="Optional dataset config name.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of evaluated examples.")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum number of generated answer tokens.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cuda, cuda:0, cpu, ...")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype used for model loading.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation to use when loading the model.",
    )
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache dir.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading from HF.")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="Enable low CPU memory model loading.")
    parser.add_argument("--predictions_file", default=None, help="Optional JSONL file to save predictions.")
    parser.add_argument("--num_examples_to_print", type=int, default=5, help="How many predictions to print.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return "auto"
    return getattr(torch, dtype_arg)


def resolve_attn_implementation(attn_arg: str, device: str) -> str:
    if attn_arg != "auto":
        return attn_arg
    return "flash_attention_2" if device.startswith("cuda") else "eager"


def compute_exact_match(predictions, references) -> float:
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references.")
    if not predictions:
        return 0.0
    return sum(int(pred == ref) for pred, ref in zip(predictions, references)) / len(predictions)


def find_local_pccot_args(model_name_or_path: str) -> Optional[Path]:
    candidate = Path(model_name_or_path).expanduser()
    if not candidate.exists():
        return None

    search_roots = [candidate]
    if candidate.is_file():
        search_roots.extend(candidate.parents)
    else:
        search_roots.extend(candidate.parents)

    for root in search_roots:
        args_path = root / models.PCCOT_ARGS_NAME
        if args_path.is_file():
            return args_path

    return None


def load_pccot_args(model_name_or_path: str, cache_dir: Optional[str], token: Optional[str]) -> models.PCCoTArguments:
    args_path = find_local_pccot_args(model_name_or_path)
    if args_path is None:
        args_path = cached_file(
            model_name_or_path,
            models.PCCOT_ARGS_NAME,
            cache_dir=cache_dir,
            token=token,
            _raise_exceptions_for_missing_entries=False,
        )
    if args_path is None:
        LOGGER.warning("No %s found for %s. Falling back to default PCCoT arguments.", models.PCCOT_ARGS_NAME, model_name_or_path)
        return models.PCCoTArguments()

    parser = HfArgumentParser(models.PCCoTArguments)
    (pccot_args,) = parser.parse_json_file(json_file=args_path)
    return pccot_args


def get_special_token(tokenizer, token: str) -> int:
    if token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": (token,)}, replace_additional_special_tokens=False)
    return tokenizer.convert_tokens_to_ids(token)


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    pccot_args = load_pccot_args(args.model_name_or_path, args.cache_dir, args.token)

    if pccot_args.bot_token_id is None:
        pccot_args.bot_token_id = get_special_token(tokenizer, "<pccot.bot>")
    if pccot_args.eot_token_id is None:
        pccot_args.eot_token_id = get_special_token(tokenizer, "<pccot.eot>")
    if pccot_args.latent_token_id is None:
        pccot_args.latent_token_id = get_special_token(tokenizer, "<pccot.latent>")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, device)

    is_peft = (
        cached_file(
            args.model_name_or_path,
            PEFT_CONFIG_NAME,
            cache_dir=args.cache_dir,
            token=args.token,
            _raise_exceptions_for_missing_entries=False,
        )
        is not None
    )
    if is_peft:
        model = AutoPeftModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )
        model.get_base_model().config = config
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()
    return model, tokenizer, pccot_args


def maybe_write_prediction(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    model, tokenizer, pccot_args = load_model_and_tokenizer(args, device=device)
    data_processor = models.COTDataProcessor(tokenizer=tokenizer, pccot_args=pccot_args)

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, token=args.token)
    if args.split not in raw_dataset:
        raise ValueError(f"Split {args.split!r} not found in dataset {args.dataset_name}. Available: {list(raw_dataset.keys())}")

    split_dataset = raw_dataset[args.split]
    if args.max_samples is not None:
        split_dataset = split_dataset.select(range(min(args.max_samples, len(split_dataset))))

    predictions_path = Path(args.predictions_file) if args.predictions_file else None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("", encoding="utf-8")

    predictions = []
    references = []
    progress = tqdm(
        batched(split_dataset, n=args.batch_size),
        total=(len(split_dataset) + args.batch_size - 1) // args.batch_size,
        desc=f"Evaluating {args.split}",
    )
    for batch in progress:
        questions = [item["question"] for item in batch]
        collated = data_processor.process(questions, device=device)
        decoded_tokens = model.generate(
            collated=collated,
            attention_mask=collated["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        decoded_tokens = decoded_tokens[:, collated["input_ids"].shape[1] :]
        answers = tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)
        batch_refs = [str(item["answer"]) for item in batch]

        predictions.extend(answers)
        references.extend(batch_refs)

        if predictions_path is not None:
            for item, pred, ref in zip(batch, answers, batch_refs):
                maybe_write_prediction(
                    predictions_path,
                    {
                        "question": item["question"],
                        "prediction": pred,
                        "reference": ref,
                        "match": pred == ref,
                    },
                )

        progress.set_postfix(exact_match=f"{compute_exact_match(predictions, references):.4f}")

    exact_match = compute_exact_match(predictions, references)
    for idx, (pred, ref) in enumerate(zip(predictions[: args.num_examples_to_print], references[: args.num_examples_to_print])):
        LOGGER.info("example %d prediction: %s", idx, pred)
        LOGGER.info("example %d reference : %s", idx, ref)

    result = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "split": args.split,
        "samples": len(predictions),
        "exact_match": exact_match,
        "device": device,
    }
    return result


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
