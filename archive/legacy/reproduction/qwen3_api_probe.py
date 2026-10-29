"""Minimal metered availability probes. Never print API credentials."""
import argparse
import json
import os

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

from reproduction.budget import metered_call, status
from reproduction.common import ROOT, RUNS, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"])
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    api = OpenAI(base_url=os.getenv("OPENAI_API_BASE") or None, timeout=60, max_retries=0)
    claude = anthropic.Anthropic(timeout=60, max_retries=0)
    path = RUNS / "api_probe.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    for model in args.models:
        fn = claude.messages.create if model.startswith("claude") else api.chat.completions.create
        try:
            response = metered_call(fn, model=model, max_tokens=16,
                messages=[{"role": "user", "content": "Reply with OK."}], temperature=0)
            records[model] = dict(ok=True, resolved_model=response.model, usage=response.usage.model_dump())
        except Exception as exc:
            records[model] = dict(ok=False, error_type=type(exc).__name__, status_code=getattr(exc,"status_code",None))
        atomic_json(RUNS / "api_probe.json", records)
        print(model, records[model], flush=True)
    print(status(), flush=True)
    if not all(r["ok"] for r in records.values()):
        raise RuntimeError("At least one configured API model is unavailable")


if __name__ == "__main__":
    main()
