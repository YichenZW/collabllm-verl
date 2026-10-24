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
Server and evaluation processes also record the code commit and source hashes
at process start; the submission's commit alone does not freeze Python imports.
Initial inference used an A100 80 GB. A subsequent A40 check also passed native
40,960-token serving, real SFT/DPO adapters, and four concurrent 18,021-token
prompts. Its profiled KV cache supports 4.99 simultaneous native-length requests.
New inference jobs therefore accept one A40/A100/H100; each runtime records its
hardware and configuration. Writing training still requests A100/H100 for its
longer sequences; automatic microbatch sizing retains effective batch size 64.

The cluster has a 12-hour job limit. Conditional DPO resume jobs run after the
original allocation ends. They skip models whose full schedule and final artifacts
already exist; otherwise they restore the latest complete checkpoint, including
optimizer, scheduler and RNG state. All saved training arguments, including the
original microbatch and accumulation sizes, are preserved even on another GPU
type. Incomplete checkpoint writes are ignored. Evaluation depends on the full
training/resume chain. `resumes/` records any continuation; resumed train loss and
runtime describe the resumed segment, while `trainer_state.json` retains the
global step and history; the resumed Trainer loss is not a reconstructed mean
over all original and resumed updates. A separate GPU smoke check verifies checkpoint loading
and subsequent optimizer/adapter updates without making external API calls.

Live job state and ownership are checked with `scontrol`; `sacct` is only the
fallback for jobs no longer retained by the controller. Accounting can lag at
allocation startup. The first math handoff safely refused this stale state,
left the source running, and was requeued after the state check was corrected.

An existing pending resume can also perform a controlled GPU handoff after a
complete epoch checkpoint exists. A request in `RUNS/handoffs/` names the exact
task, source job and replacement job. Once allocated, the replacement verifies
job ownership, CUDA execution, adapter contents, optimizer step, scheduler step,
and saved Python/NumPy/CPU/CUDA RNG states before cancelling only that source
job. It waits for the source to become terminal before writing any training
output. Until a replacement GPU is actually ready, the original training keeps
running. Training resumes from the latest complete checkpoint with the original
eight epochs, microbatch, accumulation and all other training arguments; work
after that checkpoint may be recomputed. Handoff and resume audits retain both
job identities and checkpoint provenance. CPU threads follow the new allocation.

The full code reference preflight exposed two environment problems beyond the
initial 32 samples: BigCodeBench/363 could hang when the upstream process guard
blocked cleanup of multiprocessing.Pool workers; BigCodeBench/1005's ZIP host
returned HTTP 403 for Python's default User-Agent. The experiment-local checker
allows cleanup only within the test's own descendant process tree, limits default
Pool size to two CPUs, and uses a browser default User-Agent only for the original
getsamplefiles.com host. Other hosts and explicit candidate headers are preserved.
Original prompts, reference code, URLs and assertions remain unchanged. Both real
references passed isolated checks and intentionally wrong implementations failed.
The installed shared package is untouched. Every code result records the checker
source hashes; stale local scores and canonical caches are recomputed without
additional model/judge API calls. Timed-out checks terminate their own isolated
process group, including children, rather than leaving a blocked Pool behind.

Full validation also exposed a removed dependency: librosa 0.10.1 imports
`pkg_resources`, which is absent from the shared environment's setuptools84.
Install `setuptools==80.9.0` into the private `RUNS/code_python_overlay` using
`uv pip install --python .venv-code/bin/python --target /absolute/RUNS/code_python_overlay --no-deps setuptools==80.9.0`.
The checker prepends this overlay and records its package metadata and
implementation hashes; no shared environment package is changed.

Two unchanged reference solutions fail their own assertions under the pinned
official dependencies: BigCodeBench/111 masks missing calendar cells in a heatmap,
whereas its test expects zeros; BigCodeBench/276 returns SciPy's documented NaN
for constant/singleton skewness and kurtosis, whereas its tests forbid NaN.
`code_reference_issues.json` binds these findings to exact dataset metadata and
specific assertion failures. Separate diagnostic candidates pass the unchanged
tests, and intentionally wrong candidates fail (`code_reference_diagnostics.json`).
Reference failures stay recorded as failures. The preflight permits only these
reviewed signatures; unknown errors still stop evaluation. All candidate code is
scored against every original assertion, and all selected tasks remain in primary
results. A separate reference-pass subset sensitivity analysis is reported;
these findings do not determine the evaluation sample size.
See [SciPy skew](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.skew.html),
[SciPy kurtosis](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.kurtosis.html),
[NumPy masked-list behavior](https://numpy.org/doc/1.23/reference/generated/numpy.ma.MaskedArray.tolist.html),
and [seaborn heatmap masking](https://seaborn.pydata.org/generated/seaborn.heatmap.html).

Authorized API budgets for this Qwen experiment are $100 OpenAI and $40 Anthropic.
The user added another $20 Anthropic credit after the initial $20 authorization;
the authorization history is recorded in `plan.json`. The coordinator
enforces independent $99 and $39 conservative caps, including reservations for
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

An optional bounded worker finishes one fixed evaluation stage while its GPU jobs
run. It submits reserved batches, collects results, finalizes local scores, and
refreshes reports. It exits only after all requested methods finish or on an
error/timeout; it does not expand samples or restart training. Do not run manual
batch controllers concurrently with this worker. Its owner, progress and log are
recorded under the run directory.

```bash
.venv/bin/python -m reproduction.qwen3_batch_worker --limit 32 --tasks math-hard bigcodebench
```

Once a larger stage is recorded in `plan.json`, `--extra-roots` can score its
generated conversations concurrently. For example, add the absolute `eval-200`
directory for the approved full math expansion. Worker completion still refers
to the primary `--limit` stage; any extra-stage requests remain in the durable
batch manifests for the next stage. SIGTERM requests a clean stop after the
current operation finishes, without interrupting a reserved API submission.

Expansion decisions are recorded before generating additional samples. The
current approved targets are math200 and code100 for all four offline methods.
Code100 uses the actual new24-sample batch costs from completed Base/Proactive32,
the higher original pilot projection, and a 50% margin. Its planning calculation
also includes outstanding trained32 evaluations, full math200, and a $10 writing
allowance (`code100_cost_decision.json`). This is an estimate, subject to the
provider hard caps. Larger code targets await all four methods' measured costs.
Only fully scored seed stages are copied, avoiding duplicate judge requests.
The worker can watch both `eval-200` and `eval-100` as extra roots while the
trained32 comparisons finish.

Submit immutable copies of shell launchers, so later edits cannot alter a running
Bash script. Initial job 2237770 failed from such a live edit before evaluation;
2237776 failed during CUDA initialization on r001, also before any paid evaluation.
The replacement uses GPU types already validated with this pinned environment.

Completed SFT checkpoints can be evaluated before full DPO training finishes:
`qwen3_sft_eval.sh` runs the already planned 8-sample synchronous pilot followed
by 32-sample generation using that pilot as its seed. Subsequent joint SFT/DPO
stages depend on this stage and reuse its saved SFT results; sample counts and
API work are unchanged. This only changes scheduling, not the protocol.

A scoring audit found that the historical Claude API wrapper prepends the exact
string `You are a helpful assistant designed to output in JSON format.` to every
string prompt with `json_object=True`. The first Qwen evaluations omitted that
wrapper instruction, although the task prompt bodies were unchanged. This is
corrected by `judge_protocol.py` and verified against the archived callable.
The archived GPT wrapper only prepends its instruction when the prompt contains
no `json`; all three simulator prompts already contain it, so saved conversations
and training remain valid. Judge temperature 0 and output limit 4096 are unchanged.

All earlier scores are superseded, archived in `scoring_archive/omitted-json-prefix-v0`,
and re-evaluated for every method with `d573e49-claude-json-prefix-v1`. Old in-flight
batches are still settled and retained, but cannot publish to current sample
fields. Exact scoring request hashes deduplicate identical dialogues copied into
8/32/100/200-sample stages. API costs from both protocols count against the same
budget. Original code execution still receives the extractor output directly;
no selective fence stripping, escape repair, or reference-based edits are applied
to improve scores. The wrapper correction does not guarantee syntactically valid
code extraction from the replacement Claude model.
