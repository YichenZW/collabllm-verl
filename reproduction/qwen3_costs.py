"""Cost-only sizing evidence from the synchronous pilot, without reading scores."""
import json
from collections import defaultdict

from reproduction.budget import rates, status
from reproduction.common import RUNS, atomic_json


def main():
    records = []
    for manifest in sorted((RUNS / "eval-8").glob("*/*/manifest.json")):
        protocol = json.loads(manifest.read_text())
        summary_path = manifest.parent / "summary.json"
        usage_path = manifest.parent / "api_usage.jsonl"
        if not summary_path.exists() or not usage_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary["completed"] != summary["expected"] or summary["failed"]:
            continue
        costs = defaultdict(float)
        calls = defaultdict(int)
        for line in usage_path.read_text().splitlines():
            call = json.loads(line)
            usage = call["usage"]
            model = call["model"] + ("@batch" if call.get("transport") == "batch" else "")
            inp, out = rates(model)
            nin = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            nin += (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
            nout = usage.get("output_tokens", usage.get("completion_tokens", 0))
            costs[call["kind"]] += (nin * inp + nout * out) / 1e6
            calls[call["kind"]] += 1
        n = summary["completed"]
        claude = costs["judge"] + costs["extract"]
        records.append(dict(task=protocol["dataset"], variant=protocol["variant"], n=n,
            calls=dict(calls), usd=dict(costs), openai_per_sample_usd=costs["user"] / n,
            claude_synchronous_per_sample_usd=claude / n,
            claude_batch_projected_per_sample_usd=claude / (2 * n)))
    result = dict(pilot_costs=records, budget=status(),
        caveat="Pilot means are projections, not guarantees; input/output lengths vary. Advance reservation caps still govern every request.")
    atomic_json(RUNS / "cost_pilot.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
