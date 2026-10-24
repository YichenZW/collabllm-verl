#!/bin/bash
set -euo pipefail
cd /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b
bash reproduction/qwen3_train.sh math-hard sft 2
bash reproduction/qwen3_train.sh math-hard offline_dpo 2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json, math
from safetensors import safe_open
from reproduction.common import RUNS, atomic_json
records = {}
for mode in ['sft', 'offline_dpo']:
    directory = RUNS/'checkpoints/math-hard'/f'{mode}-smoke'
    state = json.loads((directory/'trainer_state.json').read_text())
    metrics = json.loads((directory/'train_results.json').read_text())
    assert state['global_step'] == 2 and math.isfinite(metrics['train_loss'])
    assert (directory/'adapter_model.safetensors').stat().st_size > 0
    records[mode] = dict(global_step=state['global_step'], metrics=metrics)
before = RUNS/'checkpoints/math-hard/sft-smoke/adapter_model.safetensors'
after = RUNS/'checkpoints/math-hard/offline_dpo-smoke/adapter_model.safetensors'
with safe_open(before, framework='pt') as first, safe_open(after, framework='pt') as second:
    changed = sum(not first.get_tensor(k).equal(second.get_tensor(k)) for k in first.keys())
assert changed > 0, 'DPO did not update adapter weights'
records['changed_dpo_adapter_tensors'] = changed
atomic_json(RUNS/'training_smoke.json', records)
print('QWEN3_TRAINING_SMOKE_PASS', records)
PY
