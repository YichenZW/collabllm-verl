#!/bin/bash
set -euo pipefail
cd /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 HF_HUB_OFFLINE=1
export COLLABLLM_RUNS=/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs
bash reproduction/gpu_check.sh
exec .venv/bin/python -m reproduction.qwen3_inference_smoke "$@"
