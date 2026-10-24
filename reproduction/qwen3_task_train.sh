#!/bin/bash
set -euo pipefail
cd /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
bash reproduction/qwen3_train.sh "${1:?task required}" sft
bash reproduction/qwen3_train.sh "$1" offline_dpo
