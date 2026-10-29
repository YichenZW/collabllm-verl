# Legacy experiment archive

This directory is a reference archive of the pre-veRL reproduction workflow.
Its source snapshot is `98fb181bb8975e638a3e13925d38ca4cb3453479`, tagged
`legacy-pre-verl-2026-10-28`.

| Archived path | Contents |
| --- | --- |
| `legacy/scripts/train/` | Historical TRL SFT, Offline DPO, Online DPO, and PPO trainers |
| `legacy/scripts/engine/` | Historical dataset generation and inference commands |
| `legacy/reproduction/` | Experiment-specific preparation, serving, evaluation, batch scoring, Slurm launch/resume/monitor scripts, tests, reports, and environment pins |
| `legacy/requirements.txt`, `legacy/pyproject.toml` | Pre-cleanup dependency specifications |
| `legacy/notebook_tutorials/` | Original tutorials using the previous workflow |
| `legacy/README.md`, `legacy/REPRODUCTION.md` | Previous project entry points |
| `legacy/transparency_document.pdf` | Upstream transparency document |

Moved files retain their original contents, including internal imports, relative
paths, and historical cluster paths. The relocated tree is for inspection, not
an executable installation. Historical documentation describes the original
layout and may link to components that remain at the current repository root.

For an audit requiring the complete original layout, restore it to a separate
worktree from the repository root:

```bash
git worktree add --detach ../collabllm-legacy-audit legacy-pre-verl-2026-10-28
```

That command only restores source. Historical environments, model weights, run
artifacts, and cluster paths are still external requirements. The original
experiment branches can likewise be checked out at their pinned commits.
No script in this archive is an entry point for new experiments.

[`SNAPSHOT.json`](SNAPSHOT.json) records the move map and source revision.
The dataset/prompt/reward components remain at the repository root, with the
existing shared-component regression tests copied to `tests/`. Default pytest
discovery and Python package discovery exclude this archive.
