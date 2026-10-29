# Human–LLM collaboration with veRL

Future experiments in this project use **veRL** as the training and rollout
backend. This repository retains the history of
[Wuyxin/collabllm](https://github.com/Wuyxin/collabllm) and the completed
Llama-3.1-8B / Qwen3-4B reproduction code.

Start with [`workflows/verl/README.md`](workflows/verl/README.md) for the upstream
recipe and migration status. This cleanup establishes the repository layout;
the experiment-specific veRL launchers have not yet been ported or validated in
this branch.

| Path | Purpose |
| --- | --- |
| `workflows/verl/` | Entry point for future veRL configurations and launchers |
| `collabllm/` | Retained dataset, prompt, simulator, reward, and parsing components |
| `examples/` | Retained task datasets and metrics |
| `tests/` | Existing offline regression checks for the retained components |
| `archive/legacy/` | Historical TRL trainers, experiment runners, environments, notebooks, and reports |
| `docs/source_manifest.json` | Pinned experiment revisions and archive provenance |

The retained Python components are migration references and reusable building
blocks; retaining them does not mean that veRL already uses their behavior.
veRL and its recipe manage the new execution environment. The root
`pyproject.toml` describes the retained CollabLLM package, not a complete veRL
environment; historical lock files belong to the archive.
The standalone TRL trainer dependency has also been removed from the active
package metadata; its original specification is preserved in the archive.

## Historical experiments

See [REPRODUCTION.md](REPRODUCTION.md) for preserved branches, cohort sizes, and
the archive recovery tag. Old launch commands and cluster-specific paths are
kept under [archive/](archive/README.md) for provenance. Default test discovery
excludes that archive.

The paper source is maintained separately in
[human-llm-collab-rl-overleaf](https://github.com/YichenZW/human-llm-collab-rl-overleaf).

## Offline checks

With the retained package dependencies and pytest available, run:

```bash
ENABLE_COLLABLLM_LOGGING=0 python -m pytest -q
```

These tests use local fixtures and mocked model calls. No GPU or paid API is
needed. The cleanup preserves the upstream [MIT license](LICENSE) and
attribution to [CollabLLM](https://arxiv.org/abs/2502.00640).
