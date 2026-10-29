#!/bin/bash
set -euo pipefail
repo=/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export ENABLE_COLLABLLM_LOGGING=0 VLLM_USE_V1=0 HF_HUB_OFFLINE=1
export COLLABLLM_RUNS=/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs
export RUN_USER_DIR="$COLLABLLM_RUNS/runtime/${SLURM_JOB_ID}"
export PATH="$repo/.venv/bin:$PATH"
task=${1:?dataset required}
limit=${2:?sample count required}
variants=${3:-base,proactive,sft,offline_dpo}
seed_root=${4:-$COLLABLLM_RUNS/no-seed}
workers=${5:-4}
phase=${6:-all}
port=$((20000 + SLURM_JOB_ID % 30000))
IFS=',' read -ra modes <<< "$variants"
adapters=()
for variant in "${modes[@]}"; do
    case "$variant" in
        sft|offline_dpo|online_dpo|ppo)
            checkpoint="$COLLABLLM_RUNS/checkpoints/$task/$variant"
            test -f "$checkpoint/adapter_model.safetensors"
            test -f "$checkpoint/train_results.json"
            adapters+=(--adapter "$task-$variant=$checkpoint")
            ;;
    esac
done
bash reproduction/gpu_check.sh
python -c 'import torch; assert torch.cuda.is_available(); print(torch.zeros(1, device="cuda"))'
python -c 'from reproduction.budget import rpc; print(rpc("health", {}))'
python -m reproduction.serve --port "$port" "${adapters[@]}" > "$COLLABLLM_RUNS/logs/server-$task-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
python - "$port" "$server_pid" <<'PY'
import os, sys, time, urllib.request
port, pid = map(int, sys.argv[1:])
for _ in range(360):
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
            break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError('vLLM did not become healthy')
PY
for variant in "${modes[@]}"; do
    model=base
    case "$variant" in sft|offline_dpo|online_dpo|ppo) model="$task-$variant" ;; esac
    seed_args=()
    if [ -f "$seed_root/$task/$variant/manifest.json" ]; then
        seed_args=(--seed-results "$seed_root/$task/$variant")
    fi
    python -m reproduction.evaluate --dataset "$task" --variant "$variant" \
        --model "$model" --local-url "http://127.0.0.1:$port/v1" --limit "$limit" \
        --workers "$workers" --phase "$phase" \
        --output "$COLLABLLM_RUNS/eval-$limit/$task/$variant" "${seed_args[@]}"
done
