#!/usr/bin/env bash
# MUX: Evaluate a GPT-2 checkpoint
# Usage: CKPT_DIR=/path/to/checkpoint bash scripts/eval_gpt2.sh
#        DATA_NAME=svamp CKPT_DIR=/path/to/checkpoint bash scripts/eval_gpt2.sh
set -euo pipefail

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint directory}"
DATA_NAME="${DATA_NAME:-icot}"
SEED="${SEED:-11}"

python test.py \
  --data_name "$DATA_NAME" \
  --model_name_or_path gpt2 \
  --seed "$SEED" \
  --model_max_length 512 \
  --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --batch_size 128 \
  --num_latent 6 \
  --use_prj True \
  --prj_dim 768 \
  --prj_dropout 0.0 \
  --inf_latent_iterations 6 \
  --inf_num_iterations 1 \
  --remove_eos True \
  --use_lora True \
  --ckpt_dir "$CKPT_DIR" \
  --reasoning_mode codi \
  --top_k 1
