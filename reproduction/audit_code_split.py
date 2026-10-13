"""Audit historical code task overlap without changing the evaluation split.

The archived generator used BigCodeBench's default 80% training split. The
archived evaluator separately changed that ratio to 45%. Map archived idx
values through the generator's split to recover their source task identifiers.
This is a source-code reconstruction, not a claim about undocumented public
adapter training provenance.
"""
import json
import random

from datasets import load_from_disk

from reproduction.common import LEGACY, RUNS, atomic_json
from reproduction.prepare import pinned_dataset


def main():
    raw = pinned_dataset("bigcode/bigcodebench", split="v0.1.2")
    n_train = int(len(raw) * 0.8)
    splits = ["train"] * n_train + ["test"] * (len(raw) - n_train)
    random.Random(42).shuffle(splits)
    train_ids = [row["task_id"] for row, split in zip(raw, splits) if split == "train"]
    archived = load_from_disk(str(LEGACY / "data/interactive-bigcodebench"))["train"]
    archived_ids = {train_ids[idx] for idx in archived["idx"]}
    evaluation = json.loads((RUNS / "data/bigcodebench-test.json").read_text())
    matches = [dict(index=row["index"], task_id=row["metadata"]["task_id"],
                    seen_in_archived_conversations=row["metadata"]["task_id"] in archived_ids)
               for row in evaluation]
    counts = {str(n): sum(row["seen_in_archived_conversations"] for row in matches[:n])
              for n in [5, 10, 20, 50, 600]}
    result = dict(benchmark_tasks=len(raw), original_train_tasks=len(train_ids),
                  archived_conversation_tasks=len(archived_ids), overlap_counts=counts,
                  method="Archived idx mapped through default train_ratio=0.8, seed42; eval train_ratio=0.45",
                  limitation="Reconstructed from archived source; public adapter provenance remains unknown",
                  samples=matches)
    atomic_json(RUNS / "code_split_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
