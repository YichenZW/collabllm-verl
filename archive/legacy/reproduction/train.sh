#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /net/scratch2/yichenzw/collabllm
source .venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export ENABLE_COLLABLLM_LOGGING=0
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
task=${1:?dataset required}
mode=${2:?sft or offline_dpo required}
steps=${3:--1}
run_root=/net/scratch2/yichenzw/collabllm-runs
model=$(python -c 'import json; print(json.load(open("/net/scratch2/yichenzw/collabllm-runs/assets.json"))["base"])')
port=$((20000 + SLURM_JOB_ID % 30000))
batch=1
accum=64
if [ "$mode" = offline_dpo ]; then
    batch=2
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$gpu_memory" -gt 130000 ]; then batch=4; fi
    accum=$((64 / batch))
fi
max_length=1536
if [ "$task" = medium ]; then max_length=2048; fi
if [ "$steps" != -1 ]; then accum=1; fi
suffix=""
if [ "$steps" != -1 ]; then suffix=-smoke; fi
bash reproduction/gpu_check.sh
common=(--dataset_repo "$run_root/data/$task-$mode-paper" --prepared_dataset
    --output_dir "$run_root/checkpoints/$task/$mode$suffix"
    --per_device_train_batch_size "$batch" --per_device_eval_batch_size 1
    --gradient_accumulation_steps "$accum" --eval_ratio 0.0
    --logging_steps 1 --eval_steps 100000 --max_steps "$steps" --use_lora --warmup_ratio 0.1
    --system_prompt_file /net/scratch2/yichenzw/collabllm-paper-original/collabllm/prompts/llm_assistant/system_prompt.txt)
if [ "$mode" = sft ]; then
    torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 -m scripts.train.sft "${common[@]}" \
        --model_name "$model" --num_train_epochs 3 --learning_rate 1e-5 \
        --max_seq_length "$max_length" --assistant_only_loss
elif [ "$mode" = offline_dpo ]; then
    model="$run_root/checkpoints/$task/sft$suffix"
    gap=0.02
    if [ "$task" = math-hard ]; then gap=0.05; fi
    torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 -m scripts.train.offline_dpo "${common[@]}" \
        --model_name "$model" --num_train_epochs 8 --learning_rate 5e-6 \
        --max_new_tokens "$max_length" --max_prompt_length 2048 --minimum_gap "$gap" --truncation_mode keep_end
else
    echo "Unsupported training mode: $mode" >&2
    exit 1
fi
