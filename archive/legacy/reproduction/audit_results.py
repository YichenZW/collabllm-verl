"""Audit saved experimental evidence offline; never makes paid API calls."""
import argparse
import datetime
import getpass
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import tiktoken
from nltk.translate.bleu_score import sentence_bleu

from reproduction.budget import status
from reproduction.common import ROOT, RUNS, atomic_json, digest
from reproduction.record_jobs import IDS


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Metric mismatch: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--writing-n", type=int, default=18)
    parser.add_argument("--code-n", type=int, default=20)
    parser.add_argument("--math-n", type=int, default=200)
    parser.add_argument("--require-idle", action="store_true")
    args = parser.parse_args()
    targets = {"medium": args.writing_n, "bigcodebench": args.code_n, "math-hard": args.math_n}
    report = json.loads((RUNS / "report.json").read_text())
    rows = {(row["task"], row["variant"]): row for row in report["primary_results"]}
    expected_keys = {(task, variant) for task in targets
                     for variant in ["base", "proactive", "sft", "offline_dpo", "released"]}
    if set(rows) != expected_keys or len(rows) != len(report["primary_results"]):
        raise ValueError("Missing, unexpected, or duplicate primary evaluation protocols")
    enc = tiktoken.get_encoding("cl100k_base")
    groups = []
    reuse_files = 0
    reused_samples = set()
    for (task, variant), row in sorted(rows.items()):
        n = 10 if variant == "released" and task != "math-hard" else targets[task]
        if row["n"] != row["expected"] or row["n"] != n or row["failed"]:
            raise ValueError(f"Incomplete planned group: {task}/{variant}")
        folder = RUNS / row["run"]
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        inputs = json.loads((RUNS / "data" / f"{task}-test.json").read_text())[:n]
        expected_inputs = {record["index"]: record for record in inputs}
        if len(inputs) != n or summary["completed"] != n or summary["expected"] != n:
            raise ValueError("Saved counts do not match the planned sample count")
        if manifest["test_sha256"] != digest(inputs) or summary["errors"] or summary["failed"]:
            raise ValueError("Input digest or evaluation error audit failed")
        samples = {int(path.stem): json.loads(path.read_text())
                   for path in (folder / "samples").glob("*.json")}
        if len(samples) != n or set(samples) != set(expected_inputs):
            raise ValueError("Sample identities differ from the planned prefix")
        for index, sample in samples.items():
            if not sample["completed"] or sample["index"] != index or sample["input"] != expected_inputs[index]:
                raise ValueError("Incomplete sample or changed input")
            if not all(math.isfinite(value) for value in sample["metrics"].values()):
                raise ValueError("Nonfinite metric")
            raw_itr = float(sample["judge"]["interactivity"]["score"])
            if not 1 <= raw_itr <= 3:
                raise ValueError("Raw interactivity outside the original scale")
            close(sample["metrics"]["itr"], 2 * (raw_itr - 2.5))
            close(sample["metrics"]["itr_div3"], raw_itr / 3)
            text = "".join(message["content"] + " " for message in sample["chat"]
                           if message["role"] == "assistant")
            close(sample["metrics"]["tokens_k"], len(enc.encode(text, disallowed_special=())) / 1000)
            if task == "math-hard":
                if sample["metrics"]["accuracy"] not in (0, 1):
                    raise ValueError("Invalid binary accuracy")
                close(sample["metrics"]["accuracy"], float(sample["judge"]["accuracy"]["score"]))
            elif task == "medium":
                close(sample["metrics"]["bleu"], sentence_bleu(
                    [sample["input"]["completion"]], sample["extraction"]["final_answer"]))
            else:
                close(sample["metrics"]["pass_rate"], float(sample["code_result"][0] == "pass"))
        for name, metric in summary["metrics"].items():
            values = [sample["metrics"][name] for sample in samples.values()]
            close(metric["mean"], statistics.mean(values))
            close(metric["sem"], statistics.stdev(values) / math.sqrt(n))
        task_metric = {"medium": "bleu", "math-hard": "accuracy", "bigcodebench": "pass_rate"}[task]
        close(row["task_score_percent"], 100 * summary["metrics"][task_metric]["mean"])
        close(row["itr_percent"], 100 * summary["metrics"]["itr"]["mean"])
        close(row["assistant_tokens_k"], summary["metrics"]["tokens_k"]["mean"])
        # Verify reuse byte-for-byte, including all original saved responses.
        seed_roots = ["eval-10"] if task != "math-hard" and variant != "released" else []
        if task == "medium" and n > 15 and variant != "released":
            seed_roots.append("eval-15")
        for seed_root in seed_roots:
            for seed in (RUNS / seed_root / task / variant / "samples").glob("*.json"):
                if seed.read_bytes() != (folder / "samples" / seed.name).read_bytes():
                    raise ValueError(f"Reused conversation changed: {seed}")
                reuse_files += 1
                reused_samples.add((task, variant, int(seed.stem)))
        groups.append(dict(task=task, variant=variant, n=n, run=row["run"], passed=True))
        print(task, variant, n, "PASS", flush=True)
    training = []
    for task in targets:
        for method, epochs in [("sft", 3), ("offline_dpo", 8)]:
            folder = RUNS / "checkpoints" / task / method
            state = json.loads((folder / "trainer_state.json").read_text())
            train = json.loads((folder / "train_results.json").read_text())
            evaluation = json.loads((folder / "eval_results.json").read_text())
            close(state["epoch"], epochs)
            expected_steps = 24 if method == "sft" else {"math-hard": 264, "medium": 224, "bigcodebench": 304}[task]
            if state["global_step"] != expected_steps:
                raise ValueError("Full training step count differs from the prepared data and schedule")
            if not math.isfinite(train["train_loss"]) or not math.isfinite(evaluation["eval_loss"]):
                raise ValueError("Nonfinite training metrics")
            if not (folder / "adapter_model.safetensors").is_file():
                raise ValueError("Missing trained adapter")
            training.append(dict(task=task, method=method, epochs=epochs,
                                 global_step=state["global_step"], passed=True))
    budget = status()
    if budget["committed_usd"] > 38 or budget["cap_usd"] != 38:
        raise ValueError("Budget constraint violated")
    active = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"], text=True)
    active_ours = [line for line in active.splitlines() if line.strip() in set(map(str, IDS))]
    if args.require_idle and (active_ours or (RUNS / "budget-coordinator.lock").exists()
                             or (RUNS / "budget_service.json").exists()):
        raise ValueError("Experiment resources or accounting lease remain active")
    result = dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        total_primary_samples=sum(group["n"] for group in groups), groups=groups,
        reused_file_comparisons_verified=reuse_files, unique_reused_samples=len(reused_samples),
        training=training, budget=budget,
        ledger_sha256=hashlib.sha256((RUNS / "api_budget.json").read_bytes()).hexdigest(),
        active_experiment_jobs=active_ours, resource_release_checked=args.require_idle,
        scope="Saved-result, full SFT/DPO training-state, budget and resource audit; not proof of identical paper protocol",
        limitations=report["caveats"])
    atomic_json(RUNS / "completion_audit.json", result)
    print(json.dumps(dict(total=result["total_primary_samples"], budget=budget["committed_usd"],
                          resource_release_checked=args.require_idle), indent=2), flush=True)


if __name__ == "__main__":
    main()
