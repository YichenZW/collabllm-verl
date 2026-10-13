# CollabLLM Table 1 reproduction

Workspace: `/net/scratch2/yichenzw/collabllm`; branch `reproduce/table1`.
Upstream main: `7378aab`. Paper-era checkout: `../collabllm-paper-original`,
commit `d573e49` (before the June 2025 refactor). Artifacts and Slurm logs live
in `/net/scratch2/yichenzw/collabllm-runs`, outside the source tree.

## Scope and budget

The user authorized Slurm experiments with **at most USD 40 in API credits**.
`reproduction.budget` uses one authenticated CPU coordinator to write a JSON
ledger. GPU workers reserve a conservative request upper bound through RPC
before dispatch, then settle from provider usage. Its cap is USD 38, including a
USD 0.10 allowance for preflight probes. Unavailable accounting fails closed.
Timeouts keep their reservations; provider SDK retries are disabled. Accounting
RPC retries are idempotent and never repeat a provider call. No cache discounts
are assumed. All new paid calls must use this guard. GPU compute is separate.

Inspect costs with `.venv/bin/python -m reproduction.budget`.
Do not delete or reset `../collabllm-runs/api_budget.json`. Start the coordinator
before paid experiments (one CPU, 512 MB, no GPU):

```bash
sbatch --parsable --partition=general,clab --nodes=1 --cpus-per-task=1 \
  --mem=512M --time=12:00:00 --job-name=collabllm-budget \
  --output=/net/scratch2/yichenzw/collabllm-runs/logs/budget-service-%j.log \
  reproduction/budget_service.sh
.venv/bin/python -c 'from reproduction.budget import rpc; print(rpc("health", {}))'
```

Wait for `BUDGET_COORDINATOR_READY` in the Slurm log before the health check.

The service holds `budget-coordinator.lock` for its lifetime. Its private token
is mode 0600; RPC sends reservation bounds and usage, never provider keys or
conversation text. Stop its recorded Slurm job only after paid workers finish.
Never steal an active coordinator lease. If a service is forcibly killed,
verify its Slurm job is terminal before recovering a stale lease.

Historical accounting was first migrated from SQLite to atomic directory locks.
Expanded cross-node evaluation also exposed directory-lock contention. All
three paid workers were stopped before handing the unchanged 382-record ledger
to this coordinator. `api_budget.before_rpc.json` and `budget_handover.json`
preserve the previous ledger, checksum, amounts, and terminal worker states.

The coordinator then exposed a storage bottleneck: a handler on `j001-ds`
blocked in Lustre's `osc_io_fsync_end`, delaying queued accounting requests.
After stopping the three evaluations and gracefully draining the coordinator,
all 492 records ($4.662409 including reservations) were moved byte-for-byte to
`/home/yichenzw/.local/state/collabllm-table1-20261013/api_budget.json`.
The artifact path `../collabllm-runs/api_budget.json` is now a symlink to that
ledger. The code resolves it before atomic writes, preserving both the symlink
and fsync durability on the destination filesystem. The destination directory
is mode 0700. Home NFS fsync measured 0.8–1.3 ms on the same node.
`api_budget.before_home_storage.json` and `budget_storage_handover.json` retain
the exact previous bytes, checksum, cost, and terminal-job audit. Uncertain
calls retain their full reservations; the migration does not reclaim charges.

The budgeted evaluation targets 200 math examples and 10 writing/code examples
per variant: Base, Proactive Base, released adapter, our SFT, and our Offline DPO.
The first five pilot examples are reused without additional API calls. These
counts were chosen from measured pilot costs; the global guard still controls
every request if a later example is more expensive. PPO and full Online DPO
training are outside this budgeted run. The public adapter is a separate row.

Simulation, judging, and answer extraction validate the fields they actually
consume, in addition to JSON syntax. A malformed response uses the existing
three-attempt limit; every attempt is metered, and invalid replies are saved in
the run's `invalid_api_responses.jsonl`. This fixes a real writing SFT failure
where a JSON object omitted `response`. Completed samples and conversation
turns are reused when resuming the failed sample.

Rates (USD / million input, output tokens), checked 2026-10-13:

| Model | Input | Output | Source |
|---|---:|---:|---|
| GPT-4o-mini | 0.15 | 0.60 | [OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-mini) |
| GPT-4o | 2.50 | 10.00 | [OpenAI](https://developers.openai.com/api/docs/models/gpt-4o) |
| Claude Sonnet 4.5 | 3.00 | 15.00 | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |

## Environment and data

```bash
cd /net/scratch2/yichenzw/collabllm
uv venv --python 3.11 .venv
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv pip install \
  --python .venv/bin/python -r reproduction/requirements.in -c requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
ENABLE_COLLABLLM_LOGGING=0 .venv/bin/python -m pytest reproduction -q
.venv/bin/python -m reproduction.prepare
.venv/bin/python -m reproduction.prepare_training
```

The private `.env` contains `OPENAI_API_KEY`; it is gitignored and mode 0600.
Anthropic credentials come from the inherited environment. Never print them.

Code execution uses a separate Python 3.10 environment so benchmark libraries
do not conflict with vLLM's dependencies:

```bash
uv venv --python 3.10 .venv-code
uv pip install --python .venv-code/bin/python -r reproduction/requirements-code.txt
uv pip install --python .venv-code/bin/python --no-deps bigcodebench==0.2.5
.venv/bin/python -m reproduction.validate_code --limit 50
```

Only BigCodeBench's local evaluator is used. Its optional generation/cloud
backends are intentionally absent from this environment, so `uv pip check`
reports those 19 missing package-level requirements. The primary environment
passes dependency checks, and all 50 first canonical code solutions pass in the
evaluation environment. Each scored code sample also checks its canonical
solution and aborts on an environment failure rather than assigning a model
failure. Subprocess environments omit API credentials. Code test subprocesses
run one at a time per evaluation job to avoid CPU contention changing benchmark
timeout outcomes; model generation and paid API requests remain concurrent.

Training uses the archived, original 500-conversation datasets (2303 writing,
2527 math, 2627 code preference rows), translated into the current loader schema.
No paid synthetic-data regeneration is necessary for SFT or offline DPO.
For full training, `prepare_training` selects the exact historical chosen rows,
retains original preference ordering, and holds out 50 examples using seed 42.
SFT therefore trains on 450 conversations per task; DPO trains on 1767 writing,
2105 math, and 2372 code pairs after the original reward-gap filtering.

## Evaluation protocol

The latest upstream implementation materially differs from the original.
`reproduction.prepare` recovers the pre-refactor sampling, using random seed 42:
chronological Medium split; MATH-Hard official test split; BigCodeBench v0.1.2
with 45% train and 55% test, then sampling 100/200/600 examples respectively.
`reproduction.evaluate` uses original user, judge, extraction and proactive
prompts. It saves every turn and raw judge output and resumes partial samples.

| Setting | Writing | Math | Code |
|---|---|---|---|
| User simulator in original scripts | GPT-4o | GPT-4o-mini | GPT-4o |
| Maximum user/assistant pairs | 8 | 6 | 6 |
| Assistant temperature | 0.8 | 0.6 | 0.6 |
| Assistant max output tokens | 2048 | 1536 | 1536 |
| Top-p | 0.9 | 0.9 | 0.9 |

Known differences and interpretation limits:

* Paper prose says GPT-4o-mini for user simulation, whereas original scripts use
  GPT-4o for writing/code. The primary run follows the original script and
  records the model in every manifest. A mini-only variant must be labelled.
* Claude 3.5 Sonnet snapshots `20240620` and `20241022` return HTTP 404.
  Claude Sonnet 4.5 `20250929` was successfully probed and substitutes as judge.
  This prevents an exact Table 1 numerical reproduction.
* Original BLEU operates on Python strings, hence character n-grams. Preserve
  it for comparability, rather than silently changing to word BLEU.
* Original ITR rescales 1–3 judge scores with `(score - 2.5) * 2`, which can be
  negative. Save this value, the raw score, and a separate `score/3` diagnostic.
  The latter is not a replacement for the reported historical metric.
* Public adapters' model cards do not identify their training stage. Label them
  `released`, not `Online DPO`, until provenance can be verified.
* Proactive Base explicitly renders the paper's Appendix C.2 prompt. The
  archived local-HF generation branch ignores `prompt_method` (despite the
  launch configuration requesting it); this harness follows the paper's
  intended proactive baseline and makes the prompt effective.
* Pilot sample counts are not the full Table 1 evaluation. Reports must state n.
* The archived code generator uses an 80% training split, while its evaluator
  changes this to 45%. Mapping the saved conversation indices through the
  generator's split gives 212 overlapping source tasks among the 600 historical
  test samples. The first 10 sampled test tasks have zero overlap with all 500
  archived training conversations. `reproduction.audit_code_split` records the
  task-level mapping in `code_split_audit.json`. This reconstruction does not
  establish the undocumented public adapters' training provenance.

## Slurm entry points

Use explicit Bash scripts: Slurm's `--wrap` defaults to `/bin/sh` on this cluster,
which rejects `set -o pipefail`.

* `reproduction/gpu_smoke.sh`: real 8B base/adapter inference, API user simulation,
  and upstream multiturn reward computation.
* `reproduction/online_smoke.sh`: two actual Online DPO optimizer steps using a
  four-conversation archived math dataset (two prompts processed), with three sampled futures and four new
  messages per reward. This diagnostic uses GPT-4o-mini judging, 256 output
  tokens, SFT initialization, and the modern TRL trainer's frozen initial-policy
  reference. It verifies finite rewards/loss and changed adapter weights; it is
  separate from paper-scale Online DPO and is never entered in Table 1 results.
* `reproduction/eval_pilot.sh`: one local vLLM GPU server and five paired examples
  per task/model (base, proactive, official released adapter).
* `reproduction/train.sh DATASET sft|offline_dpo [MAX_STEPS]`: upstream trainers
  on original data, LoRA r=32/alpha=16/dropout=0.1, BF16, no quantization;
  SFT assistant-only loss, LR 1e-5/3 epochs; DPO LR 5e-6/8 epochs; batch 64.
  A positive MAX_STEPS writes a distinct smoke checkpoint.

Full training uses historical sequence limits (1536 math/code, 2048 writing),
cosine scheduling with 10% warmup, and a final validation pass. Unlike the
historical script's validation on every optimizer step, validation runs at the
end; no validation-based model selection is applied. DPO uses TRL 0.18.1's
`keep_end` combined-sequence truncation, preserving response tokens when a long
prompt would otherwise consume the entire limit. The old TRL version handled
truncation differently. The prepared-data DPO path renders the historical chat
template and lets TRL append exactly one EOS terminator.
The archived model loader quantizes its frozen backbone to NF4; these runs use
BF16 without quantization. As in the archived PEFT DPO path, the policy starts
from the saved SFT adapter and the reference forward pass disables the adapter
(the reference is the base model). This is recorded explicitly rather than
assuming an SFT reference or silently changing the objective.
The paper's SFT LR is used; the historical math checkpoint path mentions 2e-5
although its launch script and the paper specify 1e-5.
The first full SFT runs (jobs 2147668/2147669/2147670) set the Trainer seed to 42
after creating their LoRA adapters, so initial LoRA randomness was not fixed.
Their saved checkpoints remain sufficient to repeat evaluation. Subsequent
launches explicitly set the seed before adapter creation. This initial-run
difference must be retained when reporting those checkpoints.

Long jobs should specify `--partition=general,clab`, GPU resources, time limit,
and `--output=/net/scratch2/yichenzw/collabllm-runs/logs/NAME-%j.log`.
On 2026-10-13, k003 exposed its L40S to `nvidia-smi` but both jobs 2147782 and
2147784 failed CUDA initialization before evaluation (`No CUDA GPUs are
available`). Subsequent evaluation jobs exclude this node with `--exclude=k003`.
Record job ID, GPU/node, final state and exit code. Do not count an enqueued or
running job as a completed experiment.

## Initial verification

* API preflight: GPT-4o-mini and GPT-4o HTTP 200; replacement Claude judge HTTP 200.
* Slurm job 2147617: NVIDIA A40 on g004, `COMPLETED`, `0:0`.
* Initial job 2147616 failed on the shell mismatch; fixed with explicit Bash.
* Eight regression checks passed; after replacing SQLite, the three budget
  tests passed again. Two Slurm nodes also concurrently exercised the new lock:
  exactly two reservations fit a 25000 microUSD test cap (total 20510).
* Job 2147639: SFT and offline-DPO two-step training passed, `COMPLETED`, `0:0`.
* Job 2147676: real A100-80GB base/adapter inference, user simulation, metrics,
  and three future MR samples passed, `COMPLETED`, `0:0`.
* Job 2147738: real H200 two-step Online DPO passed, `COMPLETED`, `0:0`.
  Mean loss 0.723679, finite gradients and rewards, and all 448 adapter tensors
  changed from the SFT initialization. See `online_smoke.json` and the saved
  preference rewards/checkpoints. This is an optimization diagnostic, not a
  Table 1 Online DPO measurement.
* Job 2147664: all 45 pilot samples completed with zero evaluation errors,
  `COMPLETED`, `0:0`, in 57m46s. A model answer failing a code unit test remains
  an observed zero pass rate; it is distinct from an evaluation runtime error.
* The additional concurrent report writer check passed (eight writers, 160
  atomic updates); binary score reports now retain Wilson 95% intervals.
* The coordinator passed local concurrency, duplicate-RPC, invalid-usage,
  authentication, and cap tests. Two real Slurm nodes submitted 64 reservations
  to an isolated test ledger: exactly two fit its 25000 microUSD cap, totaling
  20510 microUSD. These tests made no provider API calls.
  A tiny metered GPT-4o-mini request from a GPU node also passed the complete
  reserve/provider/settle path. The full updated regression suite passed all 11
  checks (`logs/tests-coordinator.log`).
* After the durable-storage and response-schema fixes, the complete suite
  passed all 18 tests in 247.41 seconds (`logs/tests-schema-storage.log`), with
  one Pydantic deprecation warning and no failures. This includes atomic
  updates through the ledger symlink and malformed provider-response fields.
* Expanded evaluation attempts 2147702/2147728/2147729 hit accounting lock
  timeouts and were cancelled for the coordinator handover. Saved conversations
  are resumed; failed attempt summaries and every cost reservation are retained.
  A lock timeout could occur before dispatch or during settlement after a paid
  response, so uncertain calls retain their full reservations.
* Earlier smoke jobs: 2147629 encountered legitimate simulator early termination
  in a trivial test; 2147667 was cancelled to replace stalled SQLite accounting.
  These attempts and all conservative cost reservations remain in the audit.
* Full code SFT: 3 epochs, train loss 0.61399, held-out loss 0.60575. Job 2147670
  saved its model successfully, then its live-edited Bash launcher failed during
  the stage transition. Subsequent stages use an immutable script snapshot;
  no completed SFT work is discarded. See `slurm_jobs.json` for actual job states.
  The same launcher transition affected 2147668 and 2147669 after successful
  SFT completion. Math held-out loss is 0.66396; writing is 0.99515.
* All 200 math / 100 writing / 600 code test rows were reconstructed offline
  from the pinned dataset revisions and matched their saved SHA256 checksums.
  `asset_versions.json` locks the exact model and dataset revisions.

## Outputs

Expanded writing and code evaluation now has 10 completed examples per model,
with zero evaluation errors in all eight groups:

| Model | Writing character BLEU (%) | Code pass rate (%) |
|---|---:|---:|
| Base | 31.57 | 10.0 |
| Proactive Base | 35.41 | 20.0 |
| Released adapter | 37.65 | 20.0 |
| Our SFT | 33.38 | 20.0 |

These are small-sample results with the substituted judge, not the paper's
full-test numbers. Code job 2147801 completed with exit 0:0. Writing job 2147800
saved all three initial groups and nine SFT samples before a missing API field
failed the last sample; after the schema fix, resume job 2147828 completed that
sample and exited 0:0. The original failure remains in `failed_attempts` and
the Slurm audit. The 200-example math evaluation and full Offline DPO training
are still running; their completion must be verified separately.

`python -m reproduction.report` refreshes `../collabllm-runs/report.json`,
`results.csv` (all measured runs), and `comparison.csv` (largest completed run
per task/model/protocol). Reference Table 1 values are stored separately from
measurements. Every summary states sample count, failures, means, and SEMs.
Reports also include Wilson 95% intervals for binary accuracy/pass rate, so an
all-fail five-example pilot is not presented as a zero-uncertainty estimate.
`python -m reproduction.record_jobs` refreshes Slurm state/exit-code provenance.
Full training checkpoints contain args, trainer state, training and held-out
metrics, tokenizer, and LoRA weights.
`MPLCONFIGDIR=../collabllm-runs/mpl-cache .venv-code/bin/python reproduction/plot_training.py`
exports the measured SFT curves as PNG, SVG, and CSV in the artifact `figures`
directory. The final small minibatch in each epoch can produce visible loss spikes.
After refreshing the report,
`MPLCONFIGDIR=../collabllm-runs/mpl-cache .venv-code/bin/python reproduction/plot_comparison.py`
exports `figures/table1_comparison.png`, an SVG, and the underlying JSON data.
The plot labels each measured sample count, leaves unevaluated models blank,
and shows published values separately with the changed judge and historical
ITR scale explicitly noted. Binary scores use Wilson 95% intervals; BLEU uses
one SEM. Partial runs remain available in `results.csv` but do not replace
completed runs in this comparison.

Reference: [paper](https://arxiv.org/pdf/2502.00640),
[upstream](https://github.com/Wuyxin/collabllm).
