#!/bin/bash
set -euo pipefail
repo=/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export ENABLE_COLLABLLM_LOGGING=0 HF_HUB_OFFLINE=1
export COLLABLLM_RUNS=/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs
export RUN_USER_DIR="$COLLABLLM_RUNS/runtime/${SLURM_JOB_ID}"
export PATH="$repo/.venv/bin:$PATH"
python -m reproduction.qwen3_resume "${1:?task required}" "${2:?source job required}"
