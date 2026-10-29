# CollabLLM reproduction source

This repository preserves the Git history of
[Wuyxin/collabllm](https://github.com/Wuyxin/collabllm) and our changes for
Llama-3.1-8B-Instruct and Qwen3-4B experiments. The upstream MIT license and
attribution are retained. The paper/report source lives separately in
[human-llm-collab-rl-overleaf](https://github.com/YichenZW/human-llm-collab-rl-overleaf).

## Source revisions and branches

Our initial checkout was upstream `7378aab4efb1168557e72699cc51aebb5b8bc656`.
The paper-era implementation and prompts are pinned separately at
`d573e4949abdc301cb2b630525c593cc432e94b8`, tagged `paper-original-d573e49`.

| Branch | Preserved commit | Purpose |
| --- | --- | --- |
| `reproduce/table1` | `1b80789befb3d35072a6b4eaf17212616b6621dc` | Original Llama training and initial evaluation |
| `reproduce/qwen3-4b-table1` | `04b9e63591daf51181f85c518ce0aa536fbbfdb5` | Qwen3-4B training, evaluation, and later online-training preparation |
| `reproduce/llama-eval-200-200-100` | `05e3153833889f12c4978ae2b874a289e37763c3` | Expanded Llama evaluation on the matched cohort |
| `reproduce/release` | See this branch's Git history | Publication entry point based on the expanded Llama branch |

The machine-readable revision map is
[`reproduction/source_manifest.json`](reproduction/source_manifest.json).
These are experiment snapshots, not a unified cross-model runner. Use the
appropriate branch for each experiment. Individual run manifests record the
source revision and hashes used at execution time; the branch tips also contain
later audit and reporting changes.

The matched comparison uses 200 math, 200 code, and 100 writing examples per
method. Llama has Base, Proactive Base, SFT, Offline DPO, and the authors' released
checkpoint (2,500 evaluations); Qwen has the first four methods (2,000 matched
evaluations). Qwen's earlier 225-example code cohort is recorded separately.
Full PPO and Online DPO experiments remain deferred; smoke tests and preparation
code do not constitute completed training results.

## Code map

| Area | Entry points |
| --- | --- |
| SFT and preference optimization | `scripts/train/sft.py`, `scripts/train/offline_dpo.py` |
| Historical Llama preparation and training | `reproduction/prepare.py`, `reproduction/prepare_training.py`, `reproduction/train.sh` |
| Qwen preparation, training, and resume | `reproduction/qwen3_preflight.py`, `reproduction/qwen3_submit.py`, `reproduction/qwen3_train.sh`, `reproduction/qwen3_resume.py` |
| Conversation generation and evaluation | `reproduction/evaluate.py`, `reproduction/serve.py` |
| Expanded Llama evaluation | `reproduction/llama_eval_prepare.py`, `reproduction/llama_eval_generate.sh`, `reproduction/llama_eval_worker.py`, `reproduction/llama_eval_report.py` |
| Judge protocol and batch scoring | `reproduction/judge_protocol.py`, `reproduction/qwen3_batch.py` |
| API accounting and reservations | `reproduction/budget.py`, `reproduction/budget_service.py` |
| Code execution and reference checks | `reproduction/code_executor.py`, `reproduction/code_runtime.py`, `reproduction/code_reference_policy.py` |

## Environments and layout

The main training/evaluation environment uses Python 3.11. Version pins are in
`reproduction/requirements.in` and `reproduction/requirements.lock.txt`.
BigCodeBench uses a separate Python 3.10 environment with
`reproduction/requirements-code.txt` and its lock file. The historical PPO smoke
environment has its own requirements and lock file. Installation steps and
experiment details are in [`reproduction/README.md`](reproduction/README.md) and
[`reproduction/QWEN3.md`](reproduction/QWEN3.md); read them on the corresponding
experiment branch because defaults differ.

Archived prompts are loaded from a sibling directory named
`collabllm-paper-original`. From a clone with the paper tag available, create it
with:

```bash
git worktree add --detach ../collabllm-paper-original paper-original-d573e49
```

The launch scripts retain the original Slurm paths and resource settings. Review
those paths before using a different cluster. `COLLABLLM_RUNS` overrides the
Python default run directory, but does not replace hard-coded shell paths.
The expanded Llama preparation script also expects historical Llama and Qwen
artifact directories; it is a continuation of those experiments, not a fresh
download-and-train command.

API credentials belong in an ignored local `.env`. The repository does not
include trained weights, environment directories, or external run directories.
The final evaluation scripts require run-specific budget authorization files;
launching a new experiment needs new run configuration and API funding.

## Local regression checks

With the main environment installed and the archived worktree present, the
following checks use fixtures and a local accounting server, without GPU or
paid API calls:

```bash
ENABLE_COLLABLLM_LOGGING=0 .venv/bin/python -m pytest -q \
  reproduction/test_budget.py reproduction/test_budget_service.py \
  reproduction/test_evaluation_recovery.py reproduction/test_judge_protocol.py \
  reproduction/test_code_reference_policy.py reproduction/test_response_schema.py \
  reproduction/test_llama_eval_budget.py
```

The release branch corrects an inherited Qwen test fixture to use the expanded
Llama batch scorer's model and generation-completion marker. The preserved
experiment branches and production evaluation behavior are unchanged.
