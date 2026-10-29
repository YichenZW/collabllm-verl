"""Archive the omitted-wrapper scores while retaining every generated dialogue."""
import json
import shutil
from datetime import datetime, timezone

from reproduction.common import RUNS, atomic_json, digest
from reproduction.judge_protocol import VERSION, LEGACY_VERSION


def main():
    if (RUNS / "batch-worker.lock").exists() or (RUNS / "claude_batches/controller.lock").exists():
        raise RuntimeError("Stop the live scoring controller cleanly before migration")
    archive = RUNS / "scoring_archive" / LEGACY_VERSION
    def snapshot(path):
        destination = archive / path.relative_to(RUNS)
        if path.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        return destination

    for name in ["report.json", "RESULTS.zh-CN.md", "results.csv", "comparison.csv",
                 "statistical_analysis.json", "plan.json"]:
        snapshot(RUNS / name)
    changed = []
    cleared = ["judge", "extraction", "metrics", "code_result", "code_harness",
               "code_reference_issue", "batch_api", "score_requests", "score_sources"]
    for path in sorted(RUNS.glob("eval-*/*/*/manifest.json")):
        manifest = json.loads(path.read_text())
        snapshot(path)
        manifest["scoring_protocol"] = VERSION
        atomic_json(path, manifest)
        summary_path = path.parent / "summary.json"
        snapshot(summary_path)
        old_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        states = []
        for sample_path in sorted((path.parent / "samples").glob("*.json")):
            state = json.loads(sample_path.read_text())
            before = digest(dict(input=state["input"], chat=state["chat"]))
            if state.get("scoring_protocol") != VERSION and any(key in state for key in cleared):
                saved = snapshot(sample_path)
                for key in cleared:
                    state.pop(key, None)
                state["completed"] = False
                state["superseded_scoring"] = dict(protocol=LEGACY_VERSION, archive=str(saved))
                atomic_json(sample_path, state)
                after = json.loads(sample_path.read_text())
                if digest(dict(input=after["input"], chat=after["chat"])) != before:
                    raise RuntimeError("Migration changed a dialogue or input")
                changed.append(dict(sample=str(sample_path), dialogue_sha256=before))
            states.append(state)
        usage_path = path.parent / "api_usage.jsonl"
        if usage_path.exists():
            snapshot(usage_path)
            calls = [json.loads(line) for line in usage_path.read_text().splitlines()]
            for call in calls:
                if "claude" in call["model"]:
                    call.setdefault("scoring_protocol", LEGACY_VERSION)
            usage_path.write_text("".join(json.dumps(call) + "\n" for call in calls))
        if summary_path.exists():
            remaining = [s for s in states if s.get("completed")]
            if remaining:
                raise RuntimeError("Inspect partially migrated completed results before continuing")
            atomic_json(summary_path, dict(expected=old_summary["expected"], completed=0,
                generated=sum(bool(s.get("generated")) for s in states),
                failed=old_summary.get("failed", 0), errors=old_summary.get("errors", []),
                scoring_protocol=VERSION, scoring_status="awaiting_corrected_scoring"))
    for name in ["math_base200_completed_audit.json", "code_base100_completed_audit.json"]:
        path = RUNS / name
        if path.exists():
            saved = snapshot(path)
            atomic_json(path, dict(status="superseded_scoring_protocol", archive=str(saved),
                                  reason="Omitted archived Claude JSON instruction; re-scoring required"))
    path = RUNS / "plan.json"
    plan = json.loads(path.read_text())
    plan["evaluation"]["scoring_protocol"] = VERSION
    plan["evaluation"]["wrapper_correction"] = dict(previous=LEGACY_VERSION,
        conversation_regeneration=False, old_scores_archived=str(archive),
        new_scores_required_for_all_methods=True, costs_count_against_existing_budget=True)
    atomic_json(path, plan)
    record = dict(status="migrated", protocol=VERSION, archived_protocol=LEGACY_VERSION,
                  finished=datetime.now(timezone.utc).isoformat(), changed_scoring_samples=changed,
                  dialogues_changed=0, archive=str(archive))
    atomic_json(RUNS / "scoring_migration.json", record)
    print(json.dumps(dict(status=record["status"], changed=len(changed), dialogues_changed=0)))


if __name__ == "__main__":
    main()
