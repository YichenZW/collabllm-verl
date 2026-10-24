This branch runs the CollabLLM reproduction with `Qwen/Qwen3-4B` in non-thinking
mode. The model revision is `1cfa9a7208912126459214e8b04321603b3df60c`.
It does not load the Llama-specific public adapters.

The experiment workspace is `/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b`.
Artifacts are under `/net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs`.
`plan.json`, `assets.json`, `slurm_jobs.json`, per-run manifests, and the API ledger
record the protocol and execution state. Historical Llama results and concurrent
experiments use separate directories and ledgers.

The existing pinned main, PPO, and code-checking environments are reused through
local symlinks. No packages in those shared environments are modified. A private
experiment-specific `.env` supplies API credentials and is never committed.

SFT uses the same 450 training conversations and 50 validation conversations per
task, three epochs, learning rate 1e-5, LoRA rank 32 / alpha 16 / dropout 0.1, and
effective batch size 64. Offline DPO uses the unchanged archived preference data,
eight epochs, learning rate 5e-6, and the base policy with LoRA disabled as reference.
Training lengths remain 1536 for math/code and 2048 for writing. Math and code run
first; writing follows those task experiments. The prior BF16/modern-TRL protocol
differences from the archived paper code remain documented in RESULTS.zh-CN.md.

Qwen-specific changes are native ChatML assistant masking, retaining Qwen's pad
token separately from EOS, and removing the native EOS-plus-newline suffix before
TRL appends the completion EOS. The local model view forces `enable_thinking=false`
in the official template. Original cached model weights and tokenizer files are
not edited. vLLM uses its default generation config rather than inheriting Qwen's
recommended sampling settings, while the evaluation passes the original task
temperature, top-p, output-length, and random-seed settings explicitly.

New authorized API budgets are $100 OpenAI and $20 Anthropic. The coordinator
enforces independent $99 and $19 conservative caps, including reservations for
unknown/failed calls. All probes, simulated users, and judges use this guard.
An unavailable coordinator fails closed. No allowance from another experiment is
reused. The user selected retaining Claude scoring, prioritizing math/code paired
comparisons, then using remaining budget for expansion and writing. Full Table 1
test sizes and full PPO/Online-DPO training must not be claimed unless completed.

Run CPU validation before releasing the GPU smoke job:

```bash
.venv/bin/python -m pytest -q reproduction/test_budget.py reproduction/test_budget_service.py reproduction/test_qwen3_budget.py
.venv/bin/python -m reproduction.qwen3_preflight
```

GPU entry points, run only inside an allocation:

```bash
bash reproduction/qwen3_smoke.sh
bash reproduction/qwen3_task_train.sh math-hard
bash reproduction/qwen3_task_train.sh bigcodebench
bash reproduction/qwen3_eval.sh math-hard 8
bash reproduction/qwen3_eval.sh bigcodebench 8
```

Writing and expansion use the same scripts. Evaluation expansions accept a prior
`eval-N` directory as the fourth argument and reuse samples only when model,
protocol, task inputs, and judge identities match. An initial OpenAI credential
returned HTTP 401. After the user updated the private configuration, GPT-4o-mini,
GPT-4o, and Claude Sonnet 4.5 all passed metered availability probes.

Refresh the machine-readable run status without any API calls:

```bash
.venv/bin/python -m reproduction.qwen3_status
```
