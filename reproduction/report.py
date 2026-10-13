"""Collect completed measured results; never fill missing cells with paper values."""
import csv
import json
from reproduction.common import ROOT, RUNS, atomic_json
from reproduction.budget import status

def main():
    rows = []
    paper = json.loads((ROOT / "reproduction/paper_table1.json").read_text())
    names = {"base": "Base", "proactive": "Proactive Base", "sft": "SFT",
             "offline_dpo": "Offline DPO", "online_dpo": "Online DPO", "ppo": "PPO"}
    for path in sorted(RUNS.glob("**/summary.json")):
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest, summary = json.loads(manifest_path.read_text()), json.loads(path.read_text())
        metrics = summary.get("metrics", {})
        if not metrics:
            continue
        task = manifest["dataset"]
        task_metric = {"math-hard": "accuracy", "medium": "bleu", "bigcodebench": "pass_rate"}[task]
        row = dict(run=str(path.parent.relative_to(RUNS)), task=task, variant=manifest["variant"],
              n=summary["completed"], expected=summary["expected"], failed=summary["failed"],
              task_score_percent=100*metrics[task_metric]["mean"],
              task_score_sem_percent=100*metrics[task_metric]["sem"] if metrics[task_metric]["sem"] is not None else None,
              assistant_tokens_k=metrics["tokens_k"]["mean"], itr_percent=100*metrics["itr"]["mean"],
              user=manifest["user_model"], judge=manifest["judge"],
              paper_task_score_percent=None, paper_assistant_tokens_k=None, paper_itr_percent=None)
        if manifest["variant"] in names:
            ref = paper["rows"][names[manifest["variant"]]][paper["task_order"].index(task)]
            row.update(paper_task_score_percent=ref[0], paper_assistant_tokens_k=ref[1], paper_itr_percent=ref[2])
        rows.append(row)
    training = []
    for args_path in sorted((RUNS / "checkpoints").glob("*/*/args.json")):
        folder = args_path.parent
        record = dict(run=str(folder.relative_to(RUNS)), args=json.loads(args_path.read_text()))
        for name in ["train_results", "eval_results", "trainer_state"]:
            path = folder / f"{name}.json"
            if path.exists():
                record[name] = json.loads(path.read_text())
        record["adapter_saved"] = (folder / "adapter_model.safetensors").exists()
        training.append(record)
    primary = {}
    for row in rows:
        if row["n"] != row["expected"] or row["failed"]:
            continue
        key = row["task"], row["variant"], row["user"], row["judge"]
        if key not in primary or row["n"] > primary[key]["n"]:
            primary[key] = row
    atomic_json(RUNS / "report.json", dict(api_budget=status(), measured_results=rows,
       primary_results=list(primary.values()), training=training,
       caveats=["Claude 3.5 retired; judge substituted with Claude Sonnet 4.5",
                "Released adapter training stage undocumented; do not equate with Online DPO",
                "Sample counts are explicit; pilot metrics are not full-test metrics"]))
    if rows:
        with (RUNS / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if primary:
        with (RUNS / "comparison.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(primary.values())
    print(json.dumps(dict(results=len(rows), training=len(training), budget=status()), indent=2))

if __name__ == "__main__":
    main()
