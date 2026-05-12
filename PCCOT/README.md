# PCCOT: Parallel Continuous Chain-of-Thought with MUX

This folder contains the code for running MUX with parallel latent token generation via Jacobi iterations, producing the MUX* results in the paper.

Based on the [PCCoT](https://github.com/pccot/pccot) framework (Wu et al., 2025), this setup generates 24 latent tokens in parallel via 3 Jacobi iterations using LLaMA 3.2 1B-Instruct.

## Environment Setup

Tested with Python 3.12 and CUDA 12.4 on NVIDIA H100 GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1
pip install transformers==4.47.1 peft==0.17.1 accelerate>=1.0 datasets>=4.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install liger-kernel sentencepiece protobuf evaluate scikit-learn stanza wandb
```

Pinned versions that matter:
- `transformers==4.47.1` — required for compatibility with the custom model configs
- `peft==0.17.1` — LoRA adapter saving/loading
- `flash-attn>=2.7` — flash attention 2 backend

## Training

**GPT-2:**
```bash
bash scripts/run_gpt2_ccot_superpose_kl.sh
```

**LLaMA 3.2 1B-Instruct:**
```bash
bash scripts/run_llama_ccot_superpose_kl.sh
```

Environment variables for customization:
```bash
WANDB_PROJECT=mux-pccot    # wandb project name
WANDB_MODE=disabled         # set to disable wandb logging
REPORT_TO=none              # set to skip all external logging
PCCOT_OUTPUT_ROOT=./outputs # output directory root
NUM_EPOCHS=10               # number of training epochs
```

## Evaluation

```bash
python test_ccot.py --config_name configs/pccot_llama3.2_1b_inst.json --ckpt_dir /path/to/checkpoint
```
