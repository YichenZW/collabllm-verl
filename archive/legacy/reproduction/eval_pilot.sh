#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /net/scratch2/yichenzw/collabllm
source .venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export ENABLE_COLLABLLM_LOGGING=0
export VLLM_USE_V1=0
export RUN_USER_DIR=/net/scratch2/yichenzw/collabllm-runs/runtime/${SLURM_JOB_ID}
run_root=/net/scratch2/yichenzw/collabllm-runs
port=$((20000 + SLURM_JOB_ID % 30000))
bash reproduction/gpu_check.sh
python -m reproduction.serve --port "$port" > "$run_root/logs/server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
python - "$port" "$server_pid" <<'PY'
import os, sys, time, urllib.request
port, pid = int(sys.argv[1]), int(sys.argv[2])
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
for task in math-hard medium bigcodebench; do
    for variant in base proactive released; do
        model=base
        if [ "$variant" = released ]; then model="$task"; fi
        python -m reproduction.evaluate --dataset "$task" --variant "$variant" \
            --model "$model" --local-url "http://127.0.0.1:$port/v1" \
            --limit 5 --workers 4 --output "$run_root/pilot/$task/$variant"
    done
done
python -m reproduction.budget
