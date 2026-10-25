"""Create a separate evaluation cohort from audited historical conversations.

No network calls. Historical artifacts are read only. Old judge fields are not
reused because their Claude JSON instruction was missing. Cached dialogues are
checked against both the original split and the completed Qwen comparison split.
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from reproduction.common import ROOT, RUNS, BASE, TASKS, atomic_json, digest
from reproduction.judge_protocol import VERSION

OLD = ROOT.parent / "collabllm-runs"
QWEN = ROOT.parent / "collabllm-qwen3-4b-runs"
VARIANTS = ["base", "proactive", "sft", "offline_dpo", "released"]
TARGETS = {"math-hard": 200, "bigcodebench": 200, "medium": 100}
SOURCE_LIMITS = {"math-hard": 200, "bigcodebench": 20, "medium": 18}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in ["data", "logs", "launches", "claude_batches"]:
        (RUNS / name).mkdir(exist_ok=True)
    for name in [".venv", ".venv-code"]:
        path = ROOT / name
        if not path.exists():
            path.symlink_to(ROOT.parent / "collabllm" / name, target_is_directory=True)
    env = ROOT / ".env"
    if not env.exists():
        env.symlink_to(ROOT.parent / "collabllm-qwen3-4b" / ".env")
    overlay = RUNS / "code_python_overlay"
    if not overlay.exists():
        overlay.symlink_to(QWEN / "code_python_overlay", target_is_directory=True)
    old_assets = json.loads((OLD / "assets.json").read_text())
    assets = dict(old_assets, model_id=BASE, revision=Path(old_assets["base"]).name,
                  enable_thinking=None, generation_config="auto", max_model_len=16384)
    atomic_json(RUNS / "assets.json", assets)
    groups, sources, adapters = [], [], []
    for task, target in TARGETS.items():
        original_data = OLD / "data" / f"{task}-test.json"
        rows = json.loads(original_data.read_text())
        comparison_rows = json.loads((QWEN / "data" / f"{task}-test.json").read_text())
        assert rows[:target] == comparison_rows[:target], f"Split mismatch: {task}"
        path = RUNS / "data" / original_data.name
        if not path.exists():
            path.symlink_to(original_data)
        for variant in VARIANTS:
            previous_n = 10 if variant == "released" and task != "math-hard" else SOURCE_LIMITS[task]
            source = OLD / f"eval-{previous_n}" / task / variant
            destination = RUNS / f"eval-{target}" / task / variant
            previous = json.loads((source / "manifest.json").read_text())
            assert previous["test_sha256"] == digest(rows[:previous_n])
            assert previous["dataset"] == task and previous["variant"] == variant
            assert previous["judge"] == "claude-sonnet-4-5-20250929"
            assert previous["user_model"] == TASKS[task]["user"]
            manifest = dict(previous, limit=target, workers=4, output=str(destination),
                test_sha256=digest(rows[:target]), underlying_model=BASE,
                model_revision=assets["revision"], enable_thinking=None,
                generation_config="auto", scoring_protocol=VERSION)
            manifest_path = destination / "manifest.json"
            if manifest_path.exists():
                assert json.loads(manifest_path.read_text()) == manifest
            else:
                atomic_json(manifest_path, manifest)
            copied = 0
            for row in rows[:previous_n]:
                old_path = source / "samples" / f"{row['index']:04d}.json"
                old_bytes = old_path.read_bytes()
                old = json.loads(old_bytes)
                assert old["completed"] and old["input"] == row
                chat = old["chat"]
                assert chat and (len([m for m in chat if m["role"] != "system"]) >= 2 * TASKS[task]["turns"]
                                 or "[[TERMINATE CHAT]]" in chat[-1]["content"])
                provenance = dict(path=str(old_path), sha256=hashlib.sha256(old_bytes).hexdigest(),
                    manifest_path=str(source / "manifest.json"), manifest_sha256=sha(source / "manifest.json"),
                    dialogue_sha256=digest(dict(input=row, chat=chat)),
                    original_generation_source=str(ROOT.parent / "collabllm/reproduction/evaluate.py"),
                    scoring_reused=False)
                new = dict(index=old["index"], input=row, chat=chat, generated=True,
                           completed=False, legacy_generation=provenance)
                new_path = destination / "samples" / old_path.name
                if new_path.exists():
                    existing = json.loads(new_path.read_text())
                    assert existing["input"] == row and existing["chat"] == chat
                    assert existing["legacy_generation"] == provenance
                else:
                    atomic_json(new_path, new)
                assert old_path.read_bytes() == old_bytes
                sources.append(provenance)
                copied += 1
            summary_path = destination / "summary.json"
            if not summary_path.exists():
                atomic_json(summary_path, dict(expected=target, completed=0, generated=copied,
                    failed=0, errors=[], scoring_protocol=VERSION, status="prepared_no_paid_calls"))
            groups.append(dict(dataset=task, variant=variant, target=target, reused_dialogues=copied,
                               new_dialogues=target-copied, directory=str(destination)))
            if variant in ["sft", "offline_dpo", "released"]:
                checkpoint = Path(old_assets[task]["adapter"]) if variant == "released" else OLD / "checkpoints" / task / variant
                assert (checkpoint / "adapter_model.safetensors").is_file()
                cfg = json.loads((checkpoint / "adapter_config.json").read_text())
                assert cfg["r"] == 32
                adapters.append(dict(dataset=task, variant=variant, path=str(checkpoint),
                    config_sha256=sha(checkpoint / "adapter_config.json"),
                    bytes=(checkpoint / "adapter_model.safetensors").stat().st_size,
                    stage="unknown_author_release" if variant == "released" else variant))
    code = json.loads((QWEN / "code200_cost_observation.json").read_text())["records"]
    writing = json.loads((QWEN / "writing100_cost_observation.json").read_text())["records"]
    main_openai = sum(x["generation"]["openai_per_sample_usd"] * 180 for x in code)
    main_openai += sum(x["generation"]["openai_per_sample_usd"] * 82 for x in writing.values())
    extra_openai = sum(x["generation"]["openai_per_sample_usd"] for x in code) / 4 * 190
    extra_openai += sum(x["generation"]["openai_per_sample_usd"] for x in writing.values()) / 4 * 90
    claude_components = {}
    for task, target in TARGETS.items():
        means = {}
        for kind in (["judge"] if task == "math-hard" else ["judge", "extract"]):
            tokens, count = 0, 0
            for variant in VARIANTS[:4]:
                usage_path = OLD / f"eval-{SOURCE_LIMITS[task]}" / task / variant / "api_usage.jsonl"
                for line in usage_path.read_text().splitlines():
                    call = json.loads(line)
                    if call["kind"] != kind:
                        continue
                    u = call["usage"]
                    tokens += (u["input_tokens"] * 1.5 + u["output_tokens"] * 7.5) / 1e6
                    count += 1
            means[kind] = tokens / count
        claude_components[task] = dict(observed_mean_per_sample_usd=sum(means.values()),
            projected_five_methods_usd=sum(means.values()) * target * 5)
    projection = dict(openai_usd=main_openai+extra_openai,
        anthropic_batch_usd=sum(v["projected_five_methods_usd"] for v in claude_components.values()),
        method="OpenAI proxy: recent Qwen code25 / writing92 generation; Claude: Llama old actual tokens repriced as batch. Estimates, not guarantees.",
        claude_components=claude_components)
    plan = dict(scope="llama-eval-200-200-100", status="prepared_awaiting_budget_authorization",
        updated=datetime.now(timezone.utc).isoformat(), model=BASE,
        variants=VARIANTS, targets=TARGETS, groups=groups, adapters=adapters,
        total_dialogues=sum(g["target"] for g in groups),
        reused_dialogues=sum(g["reused_dialogues"] for g in groups),
        new_dialogues=sum(g["new_dialogues"] for g in groups),
        claude_requests=4000, projection=projection,
        scoring_protocol=VERSION, online_training="deferred_by_user",
        comparison="Qwen fixed first 200 math / 200 code / 100 writing; published adapter is separate, not assigned an inferred training stage.",
        historical_generation="Keep original Llama vLLM auto generation config, 16384 context, deterministic per-turn seed; old source and new source recorded separately.",
        original_results_modified=False, paid_requests_dispatched=0)
    atomic_json(RUNS / "plan.json", plan)
    atomic_json(RUNS / "seed_audit.json", dict(sources=sources, original_dialogues_changed=0,
                                              input_prefix_matches_qwen=True))
    auth = RUNS / "budget_authorization.json"
    if not auth.exists():
        atomic_json(auth, dict(scope=plan["scope"], status="pending", provider_cap_micro={"openai":0,"anthropic":0}))
    print(json.dumps({k:plan[k] for k in ["status","total_dialogues","reused_dialogues","new_dialogues","projection"]}, indent=2))


if __name__ == "__main__":
    main()
