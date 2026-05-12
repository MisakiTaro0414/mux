#!/usr/bin/env bash
# MUX: Train LLaMA 3.2 3B on GSM8K-AUG
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-./outputs/llama3b_gsm8k_aug}"
SEED="${SEED:-11}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B}"

mkdir -p "$SAVE_DIR"

python train.py \
  --output_dir "$SAVE_DIR" \
  --expt_name "mux_llama3b_gsm8k_aug_seed${SEED}" \
  --logging_dir "$SAVE_DIR/logs" \
  --logging_steps 10 \
  --model_name_or_path "$MODEL_NAME" \
  --data_name icot \
  --eval_data_name icot \
  --seed "$SEED" \
  --model_max_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --num_train_epochs 10 \
  --learning_rate 8e-4 \
  --max_grad_norm 2.0 \
  --use_lora True \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --save_ratio 0.1 \
  --save_strategy steps \
  --save_safetensors False \
  --save_total_limit 5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --do_train \
  --report_to tensorboard \
  --reasoning_mode codi \
  --teacher_only False \
  --ccot_only True \
  --num_latent 6 \
  --logging_strategy steps \
  --use_prj True \
  --prj_dim 3072 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --distill_loss_factor 20.0 \
  --ref_loss_factor 1.0 \
  --ccot_superpose_kl True \
  --ccot_superpose_kl_use_bag False \
  --ccot_superpose_kl_scalar_positional True \
  --ccot_superpose_kl_scalar_positional_mode geometric \
  --ccot_superpose_kl_scalar_positional_decay 0.9 \
  --ccot_superpose_random_chunk_to_latents True \
  --ccot_superpose_beta 1.0 \
  --ccot_superpose_beta_min 1.0 \
  --ccot_kl_temperature 1.0 \
  --remove_eos True \
  --inf_latent_iterations 6 \
  --inf_num_iterations 1 \
  --max_token_num 200 \
  --top_k 1 \
  --torch_compile True \
  --eval_every_save 1 \
  --exp_mode False
