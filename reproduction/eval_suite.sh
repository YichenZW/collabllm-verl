#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /net/scratch2/yichenzw/collabllm
source .venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export ENABLE_COLLABLLM_LOGGING=0 VLLM_USE_V1=0
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
task=${1:?dataset required}
limit=${2:?sample count required}
variants=${3:-base,proactive,released,sft,offline_dpo}
run_root=/net/scratch2/yichenzw/collabllm-runs
output_root=${4:-$run_root/eval-$limit}
seed_root=${5:-$run_root/pilot}
workers=${6:-12}
port=$((20000 + SLURM_JOB_ID % 30000))
IFS=',' read -ra modes <<< "$variants"
adapters=()
for variant in "${modes[@]}"; do
    if [ -n "${5:-}" ]; then
        test -f "$seed_root/$task/$variant/manifest.json"
    fi
    if [ "$variant" = sft ] || [ "$variant" = offline_dpo ]; then
        checkpoint="$run_root/checkpoints/$task/$variant"
        test -f "$checkpoint/adapter_model.safetensors"
        test -f "$checkpoint/train_results.json"
        adapters+=(--adapter "$task-$variant=$checkpoint")
    fi
done
bash reproduction/gpu_check.sh
python -m reproduction.serve --port "$port" "${adapters[@]}" > "$run_root/logs/server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
python - "$port" "$server_pid" <<'PY'
import os, sys, time, urllib.request
port, pid = map(int, sys.argv[1:])
for _ in range(180):
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
    if [ "$variant" = released ]; then model="$task"; fi
    if [ "$variant" = sft ] || [ "$variant" = offline_dpo ]; then model="$task-$variant"; fi
    seed_args=()
    if [ -f "$seed_root/$task/$variant/manifest.json" ]; then
        seed_args=(--seed-results "$seed_root/$task/$variant")
    fi
    python -m reproduction.evaluate --dataset "$task" --variant "$variant" \
        --model "$model" --local-url "http://127.0.0.1:$port/v1" --limit "$limit" \
        --workers "$workers" --output "$output_root/$task/$variant" "${seed_args[@]}"
done
python -m reproduction.report
