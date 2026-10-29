#!/bin/bash
set -euo pipefail
repo=/net/spaces/scratch/yichenzw-scratch2/collabllm-llama-eval-200-200-100
run_root=/net/spaces/scratch/yichenzw-scratch2/collabllm-llama-eval-200-200-100-runs
old_root=/net/spaces/scratch/yichenzw-scratch2/collabllm-runs
cd "$repo"
export COLLABLLM_RUNS="$run_root" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export ENABLE_COLLABLLM_LOGGING=0 VLLM_USE_V1=0
export RUN_USER_DIR="$run_root/runtime/$SLURM_JOB_ID"
export LLAMA_EVAL_PORT=$((20000 + SLURM_JOB_ID % 30000))
mkdir -p "$RUN_USER_DIR"
bash reproduction/gpu_check.sh
adapters=()
for task in bigcodebench medium; do
    for variant in sft offline_dpo; do
        adapters+=(--adapter "$task-$variant=$old_root/checkpoints/$task/$variant")
    done
    published=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["adapter"])' "$run_root/assets.json" "$task")
    adapters+=(--adapter "$task-released=$published")
done
.venv/bin/python -m reproduction.serve --port "$LLAMA_EVAL_PORT" --max-model-len 16384 "${adapters[@]}" > "$run_root/logs/smoke-server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
.venv/bin/python - "$LLAMA_EVAL_PORT" "$server_pid" <<'PY'
import os,sys,time,urllib.request
port,pid=map(int,sys.argv[1:])
for _ in range(360):
    os.kill(pid,0)
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2).close()
        break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError('vLLM startup did not finish within 30 minutes')
PY
.venv/bin/python -m reproduction.llama_eval_smoke
