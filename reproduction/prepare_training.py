"""Recover the historical chosen rows and 50-example validation splits."""
import random
from datasets import Dataset, DatasetDict, load_from_disk
from reproduction.common import LEGACY, RUNS, TASKS, legacy_prompt, atomic_json, digest

def split(rows):
    held_out = random.Random(42).sample(range(len(rows)), min(50, len(rows)))
    train = [i for i in range(len(rows)) if i not in set(held_out)]
    ds = Dataset.from_list(rows)
    return DatasetDict(train=ds.select(train), eval=ds.select(held_out))

def main():
    manifest = {}
    system = [dict(role="system", content=legacy_prompt("llm_assistant/system_prompt.txt"))]
    for task, cfg in TASKS.items():
        raw = list(load_from_disk(str(LEGACY / "data" / f"interactive-{task}"))["train"])
        conversations = {}
        for row in raw:
            conversations.setdefault(row["idx"], []).append(row)
        sft = []
        for idx in sorted(conversations):
            last = max(conversations[idx], key=lambda row: len(row["prompt"]))
            sft.append(dict(messages=system + last["prompt"] +
                            [dict(role="assistant", content=last["chosen"])]))
        dpo = [dict(prompt=system + row["prompt"], chosen=row["chosen"], rejected=row["rejected"])
               for row in raw if row["chosen_eval"]["reward"] - row["rejected_eval"]["reward"] >= cfg["gap"]]
        for mode, rows in [("sft", sft), ("offline_dpo", dpo)]:
            ds = split(rows)
            path = RUNS / "data" / f"{task}-{mode}-paper"
            if not path.exists():
                ds.save_to_disk(str(path))
            manifest[f"{task}/{mode}"] = dict(path=str(path), sha256=digest(rows),
                train=len(ds["train"]), eval=len(ds["eval"]), gap=cfg["gap"] if mode == "offline_dpo" else None)
    atomic_json(RUNS / "training_data.json", manifest)
    print(manifest, flush=True)

if __name__ == "__main__":
    main()
