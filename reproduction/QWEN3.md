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

The initial baseline pilot used a 16,384-token serving capacity. New servers use
Qwen's native 40,960-token capacity, so long writing conversations are not rejected
by an artificial smaller limit. This changes capacity, not the task output limits
or prompts; successfully completed pilot samples did not hit the smaller capacity.
Each new server saves its exact command and context limit alongside its logs.

New authorized API budgets are $100 OpenAI and $20 Anthropic. The coordinator
enforces independent $99 and $19 conservative caps, including reservations for
unknown/failed calls. All probes, simulated users, and judges use this guard.
An unavailable coordinator fails closed. No allowance from another experiment is
reused. The user selected retaining Claude scoring, prioritizing math/code paired
comparisons, then using remaining budget for expansion and writing. Full Table 1
test sizes and full PPO/Online-DPO training must not be claimed unless completed.
The user explicitly reconfirmed spending remaining Claude credit on larger
math/code comparisons, then writing, rather than reduced PPO/Online-DPO runs.
`online_cost_audit.json` records the archived launchers' nominal reward rollout
counts and the limits of that estimate. The final epoch is evaluated for our
training runs; published configurations also reference intermediate checkpoints
whose complete selection history is unavailable.

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

Larger evaluations can separate GPU generation from Claude Message Batch scoring.
The batch requests use the identical model snapshot, prompts, temperature and
maximum output length as the synchronous pilot, at the provider's 50% batch rate.
Transport is recorded per response. Each request is reserved before submission;
an unknown submission keeps all reservations and requires inspection. A single
controller owns the batch manifests. Refreshes are idempotent, and local scoring
refuses to fall back to additional paid API calls.

```bash
# Within a GPU allocation, first generate a paired fixed-prefix expansion:
bash reproduction/qwen3_eval.sh math-hard 32 base,proactive,sft,offline_dpo /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs/eval-8 4 generate
# Submit and later refresh from a CPU process; do not run controllers concurrently:
.venv/bin/python -m reproduction.qwen3_batch --roots /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs/eval-32 --submit
.venv/bin/python -m reproduction.qwen3_batch
.venv/bin/python -m reproduction.qwen3_finalize --roots /net/spaces/scratch/yichenzw-scratch2/collabllm-qwen3-4b-runs/eval-32
```

Submit immutable copies of shell launchers, so later edits cannot alter a running
Bash script. Initial job 2237770 failed from such a live edit before evaluation;
2237776 failed during CUDA initialization on r001, also before any paid evaluation.
The replacement uses GPU types already validated with this pinned environment.
