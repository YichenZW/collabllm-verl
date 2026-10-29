# veRL workflow

All future training and rollout entry points in this project belong here and
use veRL. This directory currently documents the transition; it does not yet
contain validated experiment launchers.

## Upstream implementation

- [veRL core](https://github.com/verl-project/verl)
- [CollabLLM recipe](https://github.com/verl-project/verl-recipe/tree/main/collabllm)
- [Recipe version requirements](https://github.com/verl-project/verl-recipe/blob/main/collabllm/REQUIRED_VERL.txt)

The upstream recipe provides data preparation, SFT, and collaborative RL
examples. The current recipe repository records the compatible core and recipe
revisions in `REQUIRED_VERL.txt`. Pin both revisions for an actual experiment;
the links above are references, not an environment lock for this project.

An existing local veRL checkout was found at
`/net/spaces/scratch/yichenzw-scratch2/collabllm-verl`, on
`codex/collabllm-local`, with base commit
`f9c855f7cf04d603c9546bc01776c74806a879c1`. It has uncommitted local changes.
This cleanup does not copy or alter that checkout; its base commit alone does
not capture those changes or establish a reproducible runtime pin.

## Experiment migration

| Area | Required work before running from this branch |
| --- | --- |
| Base / Proactive | Implement veRL rollout and the intended system prompts |
| SFT | Port data formatting, masking, splits, and training settings |
| Offline DPO / Online DPO / PPO | Validate the corresponding objective in the selected veRL version and add the needed adapters/configuration |
| Evaluation | Port the matched cohort, user simulator, judge protocol, code checks, and score aggregation |
| Execution | Configure Slurm resources, checkpoint/resume behavior, and API budget accounting |

The inspected local upstream RL example uses
`verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo`. The trainer module's
name does not establish that a run reproduces the paper's PPO or Online DPO
objective. Each method needs an explicit configuration and validation; archived
TRL commands are not a fallback execution path.

Useful migration references are the retained `collabllm/` and `examples/`
libraries, plus the historical judge, budget, code-execution, and audit modules
under `archive/legacy/reproduction/`. Those archived modules contain legacy
path and model assumptions. Port their behavior into the veRL workflow instead
of importing the archived package through `PYTHONPATH`.

The existing results retain their historical backend label. This cleanup does
not run training, allocate GPUs, or make paid API calls.
