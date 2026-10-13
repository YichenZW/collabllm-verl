# CollabLLM Table 1 reproduction

Workspace: `/net/scratch2/yichenzw/collabllm`; branch `reproduce/table1`.
Upstream main: `7378aab`. Paper-era checkout: `../collabllm-paper-original`,
commit `d573e49` (before the June 2025 refactor). Artifacts and Slurm logs live
in `/net/scratch2/yichenzw/collabllm-runs`, outside the source tree.

## Scope and budget

The user authorized Slurm experiments with **at most USD 40 in API credits**.
`reproduction.budget` maintains one JSON ledger across processes, reserves a
conservative request upper bound before dispatch, and settles from provider
usage. Its cap is USD 38, including a USD 0.10 allowance for preflight probes.
Timeouts keep their reservations; SDK retries are disabled. No cache discounts
are assumed. All new paid calls must use this guard. GPU compute is separate.

Inspect costs with `.venv/bin/python -m reproduction.budget`.
Do not delete or reset `../collabllm-runs/api_budget.json` or its lock directory.
The initial SQLite ledger is retained for audit; after its workers stop, migrate
it once with `python -m reproduction.budget --migrate-sqlite`. Migration preserves
every charge and reservation. Atomic directory locks avoid the long cross-node
SQLite lock stalls observed on this cluster's shared filesystem.

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
failure. Subprocess environments omit API credentials.

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

## Slurm entry points

Use explicit Bash scripts: Slurm's `--wrap` defaults to `/bin/sh` on this cluster,
which rejects `set -o pipefail`.

* `reproduction/gpu_smoke.sh`: real 8B base/adapter inference, API user simulation,
  and upstream multiturn reward computation.
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
The paper's SFT LR is used; the historical math checkpoint path mentions 2e-5
although its launch script and the paper specify 1e-5.
The first full SFT runs (jobs 2147668/2147669/2147670) set the Trainer seed to 42
after creating their LoRA adapters, so initial LoRA randomness was not fixed.
Their saved checkpoints remain sufficient to repeat evaluation. Subsequent
launches explicitly set the seed before adapter creation. This initial-run
difference must be retained when reporting those checkpoints.

Long jobs should specify `--partition=general,clab`, GPU resources, time limit,
and `--output=/net/scratch2/yichenzw/collabllm-runs/logs/NAME-%j.log`.
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

`python -m reproduction.report` refreshes `../collabllm-runs/report.json`,
`results.csv` (all measured runs), and `comparison.csv` (largest completed run
per task/model/protocol). Reference Table 1 values are stored separately from
measurements. Every summary states sample count, failures, means, and SEMs.
`python -m reproduction.record_jobs` refreshes Slurm state/exit-code provenance.
Full training checkpoints contain args, trainer state, training and held-out
metrics, tokenizer, and LoRA weights.
`MPLCONFIGDIR=../collabllm-runs/mpl-cache .venv-code/bin/python reproduction/plot_training.py`
exports the measured SFT curves as PNG, SVG, and CSV in the artifact `figures`
directory. The final small minibatch in each epoch can produce visible loss spikes.

Reference: [paper](https://arxiv.org/pdf/2502.00640),
[upstream](https://github.com/Wuyxin/collabllm).
