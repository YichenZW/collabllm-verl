#!/bin/bash
set -euo pipefail
cd /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
task=${1:?dataset required}
# These are the same already planned 8/32 samples. Completing them before DPO
# lets the later joint evaluation reuse the saved SFT results without new calls.
bash reproduction/qwen3_eval.sh "$task" 8 sft
bash reproduction/qwen3_eval.sh "$task" 32 sft \
    /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs/eval-8 4 generate
