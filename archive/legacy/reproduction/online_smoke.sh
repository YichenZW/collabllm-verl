#!/bin/bash
set -euo pipefail
cd /net/scratch2/yichenzw/collabllm
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled ENABLE_COLLABLLM_LOGGING=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
port=$((20000 + SLURM_JOB_ID % 30000))
bash reproduction/gpu_check.sh
torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 -m reproduction.online_smoke
