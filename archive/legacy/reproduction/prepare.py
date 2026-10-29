"""Download official assets and recover deterministic pre-refactor test splits."""
import argparse
import json
import random
import time
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download
import tiktoken

from reproduction.common import BASE, RUNS, TASKS, LEGACY, atomic_json, digest

VERSIONS = json.loads(Path(__file__).with_name("asset_versions.json").read_text())

def pinned_dataset(repo, **kwargs):
    return load_dataset(repo, revision=VERSIONS["datasets"][repo], **kwargs)

def test_rows(name):
    if name == "math-hard":
        raw = pinned_dataset("lighteval/MATH-Hard")["test"]
        rows = [dict(prompt=r["problem"], completion=r["solution"],
                     metadata={"level": r.get("level"), "type": r.get("type")}) for r in raw]
    elif name == "medium":
        raw = pinned_dataset("Kamaljp/medium_articles")["train"]
        rows = sorted((r for r in raw if r.get("timestamp") is not None),
                      key=lambda r: r["timestamp"])
        n_test = int(len(rows) * 0.005)
        enc = tiktoken.get_encoding("cl100k_base")
        rows = [dict(prompt="Please write an article in less than 500 words about: " + r["title"],
                     completion=r["text"], metadata={k: r.get(k) for k in
                     ["url", "authors", "timestamp", "tags"]})
                for r in rows[-n_test:]
                if len(enc.encode(r["text"], disallowed_special=())) <= 512]
    else:
        raw = pinned_dataset("bigcode/bigcodebench", split="v0.1.2")
        n_train = int(len(raw) * 0.45)
        splits = ["train"] * n_train + ["test"] * (len(raw) - n_train)
        random.Random(42).shuffle(splits)
        rows = []
        for r, split in zip(raw, splits):
            if split != "test":
                continue
            gt = {k: r[k] for k in ["task_id", "test", "entry_point"]}
            gt.update(dataset="bigcodebench", answer=r["code_prompt"] + r["canonical_solution"])
            rows.append(dict(prompt=r["instruct_prompt"], completion=json.dumps(gt), metadata=gt))
    indices = random.Random(42).sample(range(len(rows)), min(TASKS[name]["n"], len(rows)))
    return [{"index": i, **rows[i]} for i in indices]

def download(repo, **kwargs):
    for attempt in range(4):
        try:
            return snapshot_download(repo, revision=VERSIONS["models"][repo], **kwargs)
        except Exception:
            if attempt == 3:
                raise
            print(f"Retrying interrupted download: {repo} (attempt {attempt+2})", flush=True)
            time.sleep(3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets-only", action="store_true")
    args = p.parse_args()
    manifest = {"base": download(BASE, allow_patterns=["*.json", "*.safetensors", "*.jinja"])}
    for name in ["math-hard", "medium", "bigcodebench"]:
        cfg = TASKS[name]
        repo = f"collabllm/CollabLLM-{cfg['adapter']}-Llama-3.1-8B-Instruct"
        manifest[name] = {"adapter": download(repo)}
        if not args.assets_only:
            # These are the exact 2303/2527/2627 preference rows in Table 5.
            old = load_from_disk(str(LEGACY / "data" / f"interactive-{name}"))["train"]
            flat = []
            for r in old:
                for choice in ["chosen", "rejected"]:
                    flat.append(dict(prompt=r["prompt"], completion=r[choice], conv_id=r["idx"],
                        score=r[choice + "_eval"]["reward"],
                        single_turn_prompt="", single_turn_completion="",
                        single_turn_metadata={"source": "paper-era supervised data; goal not needed for SFT/DPO"}))
            train_path = RUNS / "data" / f"{name}-train"
            if not train_path.exists():
                Dataset.from_list(flat).save_to_disk(str(train_path))
            manifest[name].update(train_path=str(train_path), preference_rows=len(old),
                                  conversations=len(set(old["idx"])))
            rows = test_rows(name)
            target = RUNS / "data" / f"{name}-test.json"
            atomic_json(target, rows)
            manifest[name].update(test_file=str(target), n=len(rows), sha256=digest(rows))
            print(name, "test n=", len(rows), flush=True)
        atomic_json(RUNS / "assets.json", manifest)
    atomic_json(RUNS / "assets.json", manifest)
    print("Assets saved:", RUNS / "assets.json", flush=True)

if __name__ == "__main__":
    main()
