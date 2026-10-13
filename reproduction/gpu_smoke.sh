#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /net/scratch2/yichenzw/collabllm
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
export ENABLE_COLLABLLM_LOGGING=0
source .venv/bin/activate
bash reproduction/gpu_check.sh
python -m reproduction.gpu_smoke
