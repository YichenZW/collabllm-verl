"""Audit actual Qwen training/evaluation artifacts without making model API calls.

The default mode records progress and integrity errors. --final additionally
requires the explicit final coverage decision and every requested artifact.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from nltk.translate.bleu_score import sentence_bleu

from reproduction.budget import status
from reproduction.code_executor import harness_identity
from reproduction.code_reference_policy import reviewed_reference_issue
from reproduction.common import RUNS, TASKS, atomic_json, digest
from reproduction.judge_protocol import VERSION, load_score, score_params

VARIANTS = ("base", "proactive", "sft", "offline_dpo")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label)


def sample_metrics(task, sample, manifest, encoder):
    require(sample.get("scoring_protocol") == VERSION, "Obsolete scoring protocol")
    for kind, field in [("judge", "judge")] + ([] if task == "math-hard" else [("extract", "extraction")]):
        params = score_params(task, sample, kind, manifest["judge"])
        cached = load_score(params)
        require(cached is not None, f"Missing exact paid scoring evidence: {kind}")
        require(sample.get("score_requests", {}).get(kind) == digest(params), f"Wrong request identity: {kind}")
        require(sample[field] == cached["payload"], f"Scored payload differs from cache: {kind}")
        require(bool(cached["metadata"].get("usage")), f"Missing usage evidence: {kind}")
    raw = float(sample["judge"]["interactivity"]["score"])
    require(1 <= raw <= 3, "Invalid interactivity value")
    metrics = dict(itr=(raw - 2.5) * 2, itr_div3=raw / 3,
        tokens_k=len(encoder.encode("".join(m["content"] + " " for m in sample["chat"]
            if m["role"] == "assistant"), disallowed_special=())) / 1000)
    if task == "math-hard":
        metrics["accuracy"] = float(sample["judge"]["accuracy"]["score"])
        require(metrics["accuracy"] in [0, 1], "Invalid accuracy")
    elif task == "medium":
        metrics["bleu"] = float(sentence_bleu([sample["input"]["completion"]],
                                               sample["extraction"]["final_answer"]))
    else:
        require(sample.get("code_harness") == harness_identity(), "Stale code execution environment")
        require(isinstance(sample.get("code_result"), list) and len(sample["code_result"]) >= 2,
                "Missing code execution result")
        metrics["pass_rate"] = float(sample["code_result"][0] == "pass")
    require(set(sample["metrics"]) == set(metrics), "Unexpected metric fields")
    for key, value in metrics.items():
        same_number(sample["metrics"][key], value, f"Metric differs from saved source: {key}")
    return metrics


def evaluation_audit(task, variant, limit, assets, encoder):
    directory = RUNS / f"eval-{limit}" / task / variant
    if not (directory / "manifest.json").exists():
        return dict(status="pending", expected=limit, completed=0)
    manifest = json.loads((directory / "manifest.json").read_text())
    inputs = json.loads((RUNS / "data" / f"{task}-test.json").read_text())[:limit]
    expected = dict(dataset=task, variant=variant, limit=limit, seed=42, protocol="d573e49",
        model="base" if variant in ["base", "proactive"] else f"{task}-{variant}",
        underlying_model=assets["model_id"], model_revision=assets["revision"],
        enable_thinking=False, generation_config="vllm", user_model=TASKS[task]["user"],
        judge="claude-sonnet-4-5-20250929", scoring_protocol=VERSION, test_sha256=digest(inputs))
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    require(len(inputs) == limit, "Requested coverage exceeds available data")
    sample_paths = list((directory / "samples").glob("*.json"))
    require({p.name for p in sample_paths} <= {f"{r['index']:04d}.json" for r in inputs},
            "Unexpected samples outside fixed test prefix")
    rows, fingerprints = [], []
    canonical = None
    if task == "bigcodebench" and (directory / "code_validation_manifest.json").exists():
        canonical = Path(json.loads((directory / "code_validation_manifest.json").read_text())["cache"])
    for row in inputs:
        path = directory / "samples" / f"{row['index']:04d}.json"
        if not path.exists():
            continue
        sample = json.loads(path.read_text())
        require(sample["input"] == row and sample["index"] == row["index"], "Changed test input")
        if not sample.get("completed"):
            continue
        rows.append(sample_metrics(task, sample, manifest, encoder))
        fingerprints.append(digest(sample))
        if task == "bigcodebench":
            require(canonical is not None, "Missing reference validation provenance")
            gold_path = canonical / path.name
            require(gold_path.exists(), "Missing canonical validation result")
            gold = json.loads(gold_path.read_text())
            require(gold.get("harness") == harness_identity(), "Stale reference validation")
            issue = reviewed_reference_issue(row["metadata"], gold)
            require(bool(gold["pass_rate"]) or issue is not None, "Unreviewed canonical failure")
            require(sample.get("code_reference_issue") == issue, "Wrong reference issue annotation")
    result = dict(status="pending", expected=limit, completed=len(rows), path=str(directory),
                  test_sha256=digest(inputs), scored_samples_sha256=digest(fingerprints))
    if len(rows) != limit:
        return result
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("completed") != limit:
        return dict(result, pending="Waiting for final summary publication")
    require(summary.get("scoring_protocol") == VERSION and summary["expected"] == limit
            and summary["failed"] == 0, "Invalid completed summary")
    for key in rows[0]:
        values = [r[key] for r in rows]
        same_number(summary["metrics"][key]["mean"], float(np.mean(values)), f"Summary mean: {key}")
        if limit > 1:
            same_number(summary["metrics"][key]["sem"], float(np.std(values, ddof=1) / np.sqrt(limit)),
                        f"Summary uncertainty: {key}")
    return dict(result, status="complete", metrics=summary["metrics"])


def training_audit(task, mode, assets):
    directory = RUNS / "checkpoints" / task / mode
    required = ["args.json", "trainer_state.json", "train_results.json", "eval_results.json",
                "adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        return dict(status="pending", missing=missing)
    args = json.loads((directory / "args.json").read_text())
    epochs = 3 if mode == "sft" else 8
    expected = dict(num_train_epochs=epochs, learning_rate=1e-5 if mode == "sft" else 5e-6,
        max_steps=-1, peft_r=32, peft_alpha=16, peft_dropout=0.1, seed=42, use_4bit=False,
        prepared_dataset=True, dataset_repo=str(RUNS / "data" / f"{task}-{mode}-paper"),
        model_name=assets["base"] if mode == "sft" else str(directory.parent / "sft"))
    expected["max_seq_length" if mode == "sft" else "max_new_tokens"] = TASKS[task]["max_tokens"]
    if mode == "sft":
        expected["assistant_only_loss"] = True
    else:
        expected.update(minimum_gap=TASKS[task]["gap"], max_prompt_length=2048, truncation_mode="keep_end")
    for key, value in expected.items():
        require(args.get(key) == value, f"Training setting mismatch: {key}")
    require(args["per_device_train_batch_size"] * args["gradient_accumulation_steps"] == 64,
            "Wrong effective training batch")
    state = json.loads((directory / "trainer_state.json").read_text())
    require(state["global_step"] == state["max_steps"] and state["epoch"] >= epochs,
            "Final artifacts do not cover the full training schedule")
    adapter = json.loads((directory / "adapter_config.json").read_text())
    require(adapter["r"] == 32 and adapter["lora_alpha"] == 16 and adapter["lora_dropout"] == 0.1,
            "Wrong saved LoRA configuration")
    require(set(adapter["target_modules"]) == set(args["target_modules"].split(",")),
            "Wrong saved adapter target modules")
    return dict(status="complete", epochs=state["epoch"], steps=state["global_step"],
        files_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in required})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    plan = json.loads((RUNS / "plan.json").read_text())
    assets = json.loads((RUNS / "assets.json").read_text())
    require(assets["model_id"] == plan["model"] == "Qwen/Qwen3-4B"
            and assets["revision"] == plan["revision"]
            and assets["enable_thinking"] is False, "Base model identity differs from plan")
    coverage = plan["evaluation"].get("final_coverage")
    targets = {task: row["samples_per_variant"] for task, row in coverage.items()} if coverage else {
        "math-hard": 200, "bigcodebench": 100, "medium": 8}
    require(set(targets) == set(TASKS), "Final coverage must address all three tasks")
    for task, limit in targets.items():
        require(type(limit) is int and 0 < limit <= TASKS[task]["n"], f"Invalid coverage: {task}")
        if coverage:
            require(bool(coverage[task].get("decision_basis")), f"Missing coverage rationale: {task}")
    encoder = tiktoken.get_encoding("cl100k_base")
    evidence = dict(updated=datetime.now(timezone.utc).isoformat(), final_mode=args.final,
        final_coverage=coverage, targets=targets, training={}, evaluation={}, errors=[], pending=[])
    if not coverage:
        evidence["pending"].append("Final sample coverage and budget decision have not been recorded")
    data_path = RUNS / "data_fidelity_audit.json"
    if not data_path.exists():
        evidence["pending"].append("Data fidelity audit is pending")
    else:
        evidence["data_fidelity"] = json.loads(data_path.read_text())
        if evidence["data_fidelity"]["status"] == "failed":
            evidence["errors"].append("Data fidelity audit failed")
        elif evidence["data_fidelity"]["status"] != "passed_with_documented_representation_difference":
            evidence["pending"].append("Data fidelity audit has not passed")
        else:
            for task in TASKS:
                current = json.loads((RUNS / "data" / f"{task}-test.json").read_text())
                require(digest(current) == evidence["data_fidelity"]["tasks"][task]["test"]["sha256"],
                        f"Dataset changed since fidelity audit: {task}")
    for task in TASKS:
        for mode in ["sft", "offline_dpo"]:
            key = f"{task}/{mode}"
            try:
                result = training_audit(task, mode, assets)
                evidence["training"][key] = result
                if result["status"] != "complete":
                    evidence["pending"].append(f"Training: {key}")
            except (ValueError, KeyError) as exc:
                evidence["errors"].append(f"Training {key}: {exc}")
        for variant in VARIANTS:
            key = f"{task}/{variant}"
            try:
                result = evaluation_audit(task, variant, targets[task], assets, encoder)
                evidence["evaluation"][key] = result
                if result["status"] != "complete":
                    evidence["pending"].append(f"Evaluation: {key}")
            except (ValueError, KeyError) as exc:
                evidence["errors"].append(f"Evaluation {key}: {exc}")
    evidence["budget"] = status()
    for provider, budget in evidence["budget"]["providers"].items():
        require(budget["committed_usd"] <= budget["cap_usd"] <= plan["budget"]["authorized_usd"][provider],
                f"Budget cap mismatch: {provider}")
    evidence["outstanding_batches"] = [str(path) for path in (RUNS / "claude_batches").glob("batch-*.json")
        if json.loads(path.read_text()).get("state") != "ended"]
    if evidence["outstanding_batches"]:
        evidence["pending"].append("Submitted Claude batches have not all been settled")
    if len(evidence["evaluation"]) == 12 and all(
            row["status"] == "complete" for row in evidence["evaluation"].values()):
        report = json.loads((RUNS / "report.json").read_text())
        rows = {(row["task"], row["variant"]): row for row in report["primary_results"]}
        if set(rows) != {(task, variant) for task in TASKS for variant in VARIANTS}:
            evidence["pending"].append("Final paired report has not been published")
        else:
            for (task, variant), row in rows.items():
                require(row["n"] == row["expected"] == targets[task] and row["failed"] == 0,
                        f"Report coverage mismatch: {task}/{variant}")
                metrics = evidence["evaluation"][f"{task}/{variant}"]["metrics"]
                score = {"math-hard": "accuracy", "bigcodebench": "pass_rate", "medium": "bleu"}[task]
                for key, expected in [("task_score_percent", 100 * metrics[score]["mean"]),
                                      ("itr_percent", 100 * metrics["itr"]["mean"]),
                                      ("assistant_tokens_k", metrics["tokens_k"]["mean"])]:
                    same_number(row[key], expected, f"Report value mismatch: {task}/{variant}/{key}")
    evidence["status"] = "failed" if evidence["errors"] else "incomplete" if evidence["pending"] else "complete"
    atomic_json(RUNS / ("final_audit.json" if args.final else "progress_audit.json"), evidence)
    print(json.dumps(evidence, indent=2))
    if evidence["errors"] or (args.final and evidence["pending"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
