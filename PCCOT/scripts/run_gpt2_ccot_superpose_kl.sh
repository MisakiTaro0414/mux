#!/bin/bash
echo "Start running GPT-2 PCCoT superpose-KL..."
WANDB_PROJECT="${WANDB_PROJECT:-mux-pccot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT WANDB_MODE
if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_ENTITY
fi
OUTPUT_ROOT="${PCCOT_OUTPUT_ROOT:-./outputs}"
mkdir -p "$OUTPUT_ROOT"

DISTILL_LOSS_FACTOR="${DISTILL_LOSS_FACTOR:-1.0}"
REF_LOSS_FACTOR="${REF_LOSS_FACTOR:-1.0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
RUN_NAME="${RUN_NAME:-pccot-gpt2-superpose-kl-6lat}"

accelerate launch run_ccot.py \
    --model_name_or_path gpt2 \
    --config_name configs/pccot_gpt2_small.json \
    --config_overrides loss_alpha=${REF_LOSS_FACTOR},loss_gamma=${DISTILL_LOSS_FACTOR},num_iterations=${NUM_ITERATIONS},use_projection=true \
    --num_latent_tokens 6 \
    --dataset_name whynlp/gsm8k-aug \
    --label_names labels cot_labels \
    --include_last_cot False \
    --ccot_superpose_kl True \
    --ccot_superpose_kl_use_bag False \
    --ccot_superpose_kl_scalar_positional True \
    --ccot_superpose_kl_scalar_positional_weight 1.0 \
    --ccot_superpose_kl_scalar_positional_mode rope \
    --ccot_superpose_kl_scalar_positional_decay 0.9 \
    --ccot_superpose_kl_scalar_positional_sin_amp 1.0 \
    --ccot_superpose_kl_scalar_positional_fourier_pairs 64 \
    --ccot_superpose_kl_scalar_positional_base 1000.0 \
    --ccot_superpose_kl_scalar_positional_freq_sampling spread \
    --ccot_superpose_kl_scalar_positional_max_positions 128 \
    --ccot_superpose_kl_teacher_distribution False \
    --ccot_superpose_kl_teacher_temperature 1.0 \
    --ccot_superpose_kl_train_lm_head False \
    --ccot_superpose_kl_train_lm_head_reinit False \
    --ccot_superpose_uniform_chunk_to_latents False \
    --ccot_superpose_random_chunk_to_latents True \
    --ccot_superpose_teacher_force False \
    --ccot_superpose_teacher_force_poscoded False \
    --ccot_superpose_beta 1.0 \
    --ccot_superpose_beta_min 1.0 \
    --ccot_superpose_num_supervised_latents 0 \
    --ccot_kl_temperature 1.0 \
    --lora_target_modules c_attn-c_fc-c_proj \
    --lora_modules_to_save "" \
    --remove_unused_columns false \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --block_size 512 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --learning_rate 3e-3 \
    --weight_decay 1e-2 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_ccot_exact_match \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_ROOT/$RUN_NAME" \
    --overwrite_output_dir True
