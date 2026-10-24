"""Compare every prepared row with the archived paper's preprocessing functions.

Read-only for datasets and source repositories; writes only the audit evidence.
Run on a CPU Slurm allocation with the pinned Hugging Face cache available.
"""
import ast
import copy
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset, load_from_disk

from reproduction.common import LEGACY, ROOT, RUNS, TASKS, atomic_json, digest, legacy_prompt


def archived_functions(path, names, namespace, class_name=None):
    """Execute the archived function bodies without importing API/model modules."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"Missing archived functions in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def training_audit(task):
    source = LEGACY / "collabllm/datasets/__init__.py"
    raw = list(load_from_disk(str(LEGACY / "data" / f"interactive-{task}"))["train"])
    # The archived SFT function also carries an unused `qa` annotation. Stub only
    # that annotation; prompts, choices, ordering, filtering and splits execute
    # the actual archived functions on in-memory copies of all released rows.
    annotation_rows = [None] * (max(row["idx"] for row in raw) + 1)
    namespace = dict(copy=copy, random=random, re=re, np=np, List=List,
        Dataset=Dataset, DatasetDict=DatasetDict, interleave_datasets=interleave_datasets,
        SYSTEM_PROMPT=legacy_prompt("llm_assistant/system_prompt.txt"),
        datasets_info={task: {"task": TASKS[task]["task"]}},
        load_dataset=lambda *a, **kw: DatasetDict(train=Dataset.from_list(raw)),
        load_single_turn_dataset=lambda *a, **kw: {"train": {"chat": annotation_rows}})
    archived_functions(source, ["add_sys_prompt", "load_dpo_dataset", "filter_dpo",
                               "split_train_dev_datasets"], namespace)
    namespace["load_multiturn_dataset"] = lambda name, is_dpo, **kw: namespace["load_dpo_dataset"](name, **kw)
    result = {}
    for mode in ["sft", "offline_dpo"]:
        train, evaluation = namespace["split_train_dev_datasets"](
            f"collabllm/collabllm-{task}", is_dpo=True, to_sft_dataset=mode == "sft",
            minimum_gap=TASKS[task]["gap"], n_eval_per_dataset=50, add_system_prompt=True,
            return_eval_as_dict=False, seed=42)
        path = RUNS / "data" / f"{task}-{mode}-paper"
        prepared = load_from_disk(str(path))
        checked = {}
        for split, oracle in [("train", train), ("eval", evaluation)]:
            expected = [dict(messages=row["chat"]) if mode == "sft" else
                        {key: row[key] for key in ["prompt", "chosen", "rejected"]}
                        for row in oracle]
            actual = list(prepared[split])
            if actual != expected:
                mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
                raise AssertionError(f"{task}/{mode}/{split}: row mismatch at {mismatch}; "
                                     f"counts {len(actual)}/{len(expected)}")
            checked[split] = dict(rows=len(actual), sha256=digest(actual), all_rows_equal=True)
        result[mode] = dict(source_preference_rows=len(raw), path=str(path),
                           resolved_path=str(path.resolve()), splits=checked)
    return result


def test_audit(task, versions):
    configuration = {
        "math-hard": ("lighteval/MATH-Hard", "math_hard.py", "MATH", {}),
        "bigcodebench": ("bigcode/bigcodebench", "bigcodebench.py", "BigCodeBench", {"split": "v0.1.2"}),
        "medium": ("Kamaljp/medium_articles", "medium.py", "Medium", {}),
    }
    repo, filename, class_name, kwargs = configuration[task]
    raw = load_dataset(repo, revision=versions["datasets"][repo], **kwargs)
    # Medium's archived method assigns Python lists into its input mapping.
    if task == "medium":
        raw = dict(raw)
    encoder = tiktoken.get_encoding("cl100k_base")
    namespace = dict(json=json, random=random, tqdm=lambda rows: rows,
        num_tokens_from_string=lambda text: len(encoder.encode(text, disallowed_special=())))
    source = LEGACY / "collabllm/datasets" / filename
    preprocess = archived_functions(source, ["preprocess"], namespace, class_name)["preprocess"]
    processed = preprocess(SimpleNamespace(train_ratio=0.45 if task == "bigcodebench" else 0.02,
                                          test_ratio=0.005, max_tokens=512), raw)
    # ChatDataset.to_hf_dataset preserves the original order within each split.
    test = [row for row in processed if row["metadata"]["split"] == "test"]
    indices = random.Random(42).sample(range(len(test)), min(TASKS[task]["n"], len(test)))
    path = RUNS / "data" / f"{task}-test.json"
    actual = json.loads(path.read_text())
    if len(actual) != len(indices):
        raise AssertionError(f"{task}: test count mismatch")
    representation_differences = 0
    for position, (row, idx) in enumerate(zip(actual, indices)):
        oracle = test[idx]
        if row["index"] != idx or row["prompt"] != oracle["chat"][0]["content"]:
            raise AssertionError(f"{task}: input/index mismatch at position {position}")
        completion = oracle["chat"][1]["content"]
        if task == "bigcodebench":
            if json.loads(row["completion"]) != json.loads(completion):
                raise AssertionError(f"{task}: ground truth content mismatch at position {position}")
            representation_differences += row["completion"] != completion
            if row["metadata"] != json.loads(completion):
                raise AssertionError(f"{task}: checker metadata mismatch at position {position}")
        else:
            if row["completion"] != completion:
                raise AssertionError(f"{task}: answer mismatch at position {position}")
            for key, value in row["metadata"].items():
                if oracle["metadata"][key] != value:
                    raise AssertionError(f"{task}: metadata mismatch at position {position}/{key}")
    return dict(rows=len(actual), available_test_rows=len(test), sha256=digest(actual),
        dataset=repo, revision=versions["datasets"][repo], path=str(path),
        resolved_path=str(path.resolve()), all_indices_questions_and_answer_content_equal=True,
        ground_truth_serialization_differences=representation_differences,
        representation_note=("Prepared BigCodeBench ground truth serializes the same JSON keys in a "
            "different insertion order. Exact tests/reference code/entry points are unchanged. "
            "This is a prompt serialization deviation, not byte-identical prompt fidelity."
            if representation_differences else None))


def main():
    versions = json.loads((ROOT / "reproduction/asset_versions.json").read_text())
    sources = ["collabllm/datasets/__init__.py", "collabllm/datasets/dataset.py",
               "collabllm/datasets/math_hard.py", "collabllm/datasets/bigcodebench.py",
               "collabllm/datasets/medium.py", "scripts/eval_multiturn.py"]
    evidence = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
        paper_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LEGACY, text=True).strip(),
        archived_source_sha256={p: hashlib.sha256((LEGACY / p).read_bytes()).hexdigest() for p in sources},
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_note="Actual archived preprocessing/filter/split functions; unused SFT qa annotation stubbed",
        tasks={})
    output = RUNS / "data_fidelity_audit.json"
    try:
        for task in ["math-hard", "bigcodebench", "medium"]:
            evidence["tasks"][task] = dict(training=training_audit(task), test=test_audit(task, versions))
            atomic_json(output, evidence)
            print(json.dumps(dict(task=task, audit=evidence["tasks"][task])), flush=True)
        evidence["status"] = "passed_with_documented_representation_difference"
    except Exception as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        evidence["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, evidence)


if __name__ == "__main__":
    main()
