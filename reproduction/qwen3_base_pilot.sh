#!/bin/bash
set -euo pipefail
cd /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
bash reproduction/qwen3_eval.sh math-hard 8 base,proactive
bash reproduction/qwen3_eval.sh bigcodebench 8 base,proactive
