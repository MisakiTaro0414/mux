#!/usr/bin/env bash
# MUX: Probe and visualize latent tokens on GSM8K
# Usage: CKPT_DIR=/path/to/checkpoint bash scripts/probe_latent_token.sh
set -euo pipefail

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint directory}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"

python probe_latent_token.py \
  --model_name_or_path "$MODEL_NAME" \
  --ckpt_dir "$CKPT_DIR" \
  --data_name icot \
  --num_latent 1 \
  --inf_latent_iterations 6 \
  --use_prj True \
  --prj_dim 2048 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --use_lora True \
  --bf16
