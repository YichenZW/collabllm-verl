#!/bin/bash
set -euo pipefail
cd /net/scratch2/yichenzw/collabllm
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 ENABLE_COLLABLLM_LOGGING=0 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
export HF_HOME=/net/projects2/llm-creativity/tf_cache
bash reproduction/gpu_check.sh
exec .venv-ppo/bin/torchrun --standalone --nproc_per_node=1 -m reproduction.ppo_smoke
