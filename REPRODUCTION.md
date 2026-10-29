# Historical reproduction index

New experiments use veRL; see [the workflow entry point](workflows/verl/README.md).
The results below were obtained with the historical implementation and must
not be described as veRL results.

| Preserved branch | Commit | Scope |
| --- | --- | --- |
| `reproduce/table1` | `1b80789befb3d35072a6b4eaf17212616b6621dc` | Original Llama training and initial evaluation |
| `reproduce/qwen3-4b-table1` | `04b9e63591daf51181f85c518ce0aa536fbbfdb5` | Qwen training, evaluation, and later online-training preparation |
| `reproduce/llama-eval-200-200-100` | `05e3153833889f12c4978ae2b874a289e37763c3` | Expanded Llama evaluation |

The matched comparison uses 200 math, 200 code, and 100 writing examples per
method: five Llama methods (2,500 evaluations) and four Qwen methods (2,000).
Qwen's earlier 225-example code cohort is recorded separately. Full PPO and
Online DPO experiments remain deferred in this reproduction.

Detailed historical settings, environment pins, source maps, and reports are
preserved in [archive/legacy/REPRODUCTION.md](archive/legacy/REPRODUCTION.md),
[the original reproduction guide](archive/legacy/reproduction/README.md), and
[the Qwen guide](archive/legacy/reproduction/QWEN3.md).

The complete pre-cleanup layout is tagged `legacy-pre-verl-2026-10-28` at
`98fb181bb8975e638a3e13925d38ca4cb3453479`. The experiment branches retain their
original trees; no experiment commits were rewritten. See
[archive/README.md](archive/README.md) for restoring an isolated historical
checkout. [docs/source_manifest.json](docs/source_manifest.json) records the
upstream, paper-era, experiment, and archive revisions.
