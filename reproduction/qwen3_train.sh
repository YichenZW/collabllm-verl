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
task=${1:?dataset required}
mode=${2:?sft or offline_dpo required}
steps=${3:--1}
model=$(python -c 'import json; from reproduction.common import RUNS; print(json.loads((RUNS/"assets.json").read_text())["base"])')
port=$((20000 + SLURM_JOB_ID % 30000))
batch=1
gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$gpu_memory" -gt 70000 ]; then batch=2; fi
if [ "$gpu_memory" -gt 130000 ]; then batch=4; fi
accum=$((64 / batch))
max_length=1536
if [ "$task" = medium ]; then max_length=2048; fi
suffix=""
if [ "$steps" != -1 ]; then batch=1; accum=1; suffix=-smoke; fi
bash reproduction/gpu_check.sh
common=(--dataset_repo "$COLLABLLM_RUNS/data/$task-$mode-paper" --prepared_dataset
    --output_dir "$COLLABLLM_RUNS/checkpoints/$task/$mode$suffix"
    --per_device_train_batch_size "$batch" --per_device_eval_batch_size 1
    --gradient_accumulation_steps "$accum" --eval_ratio 0.0
    --logging_steps 1 --eval_steps 100000 --max_steps "$steps" --use_lora --warmup_ratio 0.1
    --seed 42 --peft_r 32 --peft_alpha 16 --peft_dropout 0.1
    --system_prompt_file /net/spaces/scratch/yichenzw-scratch2/collabllm-paper-original/collabllm/prompts/llm_assistant/system_prompt.txt)
if [ "$mode" = sft ]; then
    python -m torch.distributed.run --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
        -m scripts.train.sft "${common[@]}" --model_name "$model" \
        --num_train_epochs 3 --learning_rate 1e-5 --max_seq_length "$max_length" --assistant_only_loss
elif [ "$mode" = offline_dpo ]; then
    model="$COLLABLLM_RUNS/checkpoints/$task/sft$suffix"
    gap=0.02
    if [ "$task" = math-hard ]; then gap=0.05; fi
    python -m torch.distributed.run --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
        -m scripts.train.offline_dpo "${common[@]}" --model_name "$model" \
        --num_train_epochs 8 --learning_rate 5e-6 --max_new_tokens "$max_length" \
        --max_prompt_length 2048 --minimum_gap "$gap" --truncation_mode keep_end
else
    echo "Unsupported training mode: $mode" >&2
    exit 1
fi
