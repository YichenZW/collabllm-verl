"""Exact archived JSON prompt wrappers and reusable paid scoring results."""
import json

from reproduction.common import RUNS, TASKS, atomic_json, chat_text, digest, legacy_prompt

VERSION = "d573e49-claude-json-prefix-v1"
LEGACY_VERSION = "omitted-json-prefix-v0"
CLAUDE_PREFIX = "You are a helpful assistant designed to output in JSON format."
GPT_PREFIX = "You are a helpful assistant designed to output JSON. "


def wrap_json_prompt(prompt, model):
    if "claude" in model:
        return CLAUDE_PREFIX + prompt
    return GPT_PREFIX + prompt if "json" not in prompt.lower() else prompt


def score_params(task, sample, kind, model):
    row = sample["input"]
    if kind == "judge":
        prompt = legacy_prompt(f"llm_judge/{TASKS[task]['task']}.txt").format(
            chat_history="", chat=chat_text(sample["chat"]), question=row["prompt"],
            answer=row["completion"])
    elif kind == "extract":
        prompt = legacy_prompt("funtional/extract_answer.txt").format(chat=chat_text(sample["chat"]))
    else:
        raise ValueError("Unknown scoring request kind")
    return dict(model=model, max_tokens=4096, temperature=0,
                messages=[dict(role="user", content=wrap_json_prompt(prompt, model))])


def cache_path(key):
    return RUNS / "judge_cache" / VERSION / f"{key}.json"


def load_score(params):
    key = digest(params)
    path = cache_path(key)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["request_key"] != key or record["params"] != params or record["scoring_protocol"] != VERSION:
        raise ValueError("Scoring cache identity mismatch")
    return record


def save_score(params, payload, metadata):
    key = digest(params)
    record = dict(request_key=key, scoring_protocol=VERSION, params=params,
                  payload=payload, metadata=metadata)
    previous = load_score(params)
    if previous is not None and previous != record:
        raise RuntimeError("A different scorer already saved this exact request")
    atomic_json(cache_path(key), record)
    return record


def apply_score(sample, kind, record):
    field = "judge" if kind == "judge" else "extraction"
    if field in sample and (sample.get("scoring_protocol") != VERSION or
                           sample.get("score_requests", {}).get(kind) != record["request_key"] or
                           sample[field] != record["payload"]):
        raise RuntimeError("Existing scoring uses a different protocol; archive it before reuse")
    sample[field] = record["payload"]
    sample["scoring_protocol"] = VERSION
    sample.setdefault("score_requests", {})[kind] = record["request_key"]
    sample.setdefault("score_sources", {})[kind] = record["metadata"]


def reuse_cached(roots):
    """Copy exact paid requests across seed/expanded stages without new calls."""
    reused = 0
    for root in roots:
        for manifest_path in sorted(root.glob("*/*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("scoring_protocol") != VERSION:
                continue
            task = manifest["dataset"]
            for path in sorted((manifest_path.parent / "samples").glob("*.json")):
                sample = json.loads(path.read_text())
                if not sample.get("generated") or sample.get("completed"):
                    continue
                changed = False
                for kind, field in [("judge", "judge"), ("extract", "extraction")]:
                    if field in sample or (kind == "extract" and task == "math-hard"):
                        continue
                    record = load_score(score_params(task, sample, kind, manifest["judge"]))
                    if record is not None:
                        apply_score(sample, kind, record)
                        changed = True
                        reused += 1
                if changed:
                    atomic_json(path, sample)
    return reused
