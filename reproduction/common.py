"""Experiment paths and immutable paper settings (no credentials in manifests)."""
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.getenv("COLLABLLM_RUNS", str(ROOT.parent / "collabllm-qwen3-4b-runs")))
LEGACY = ROOT.parent / "collabllm-paper-original"
PAPER_COMMIT = "d573e49"
BASE = "Qwen/Qwen3-4B"
TASKS = {
    "medium": dict(task="document-editing", n=100, turns=8, temperature=0.8,
                   max_tokens=2048, adapter="writing", user="gpt-4o", gap=0.02),
    "math-hard": dict(task="question-answering", n=200, turns=6, temperature=0.6,
                      max_tokens=1536, adapter="math", user="gpt-4o-mini", gap=0.05),
    "bigcodebench": dict(task="code-generation", n=600, turns=6, temperature=0.6,
                         max_tokens=1536, adapter="code", user="gpt-4o", gap=0.02),
}

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

def source_identity(*relative_paths):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return dict(commit_at_process_start=commit, files={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in relative_paths})

def legacy_prompt(relative_path):
    return (LEGACY / "collabllm/prompts" / relative_path).read_text()

def chat_text(messages):
    return "\n".join(f"**{m['role']}**: {m['content']}\n" for m in messages
                     if m["role"] != "system")
