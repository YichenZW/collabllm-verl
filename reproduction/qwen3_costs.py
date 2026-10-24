"""Cost-only pilot sizing from actual transport prices, without reading scores."""
import json
from collections import defaultdict

from reproduction.budget import rates, status
from reproduction.common import RUNS, atomic_json


def summarize_usage(lines, n):
    costs = defaultdict(float)
    calls = defaultdict(int)
    claude_synchronous_equivalent = claude_batch_projection = 0.0
    for line in lines:
        call = json.loads(line)
        usage = call["usage"]
        is_batch = call.get("transport") == "batch"
        model = call["model"] + ("@batch" if is_batch else "")
        inp, out = rates(model)
        nin = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        nin += (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
        nout = usage.get("output_tokens", usage.get("completion_tokens", 0))
        actual = (nin * inp + nout * out) / 1e6
        costs[call["kind"]] += actual
        calls[call["kind"]] += 1
        if call["kind"] in ["judge", "extract"]:
            # Batch usage already has its discount. Never halve that cost twice.
            claude_synchronous_equivalent += actual * (2 if is_batch else 1)
            claude_batch_projection += actual * (1 if is_batch else 0.5)
    return dict(calls=dict(calls), usd=dict(costs), openai_per_sample_usd=costs["user"] / n,
        claude_observed_per_sample_usd=(costs["judge"] + costs["extract"]) / n,
        claude_synchronous_per_sample_usd=claude_synchronous_equivalent / n,
        claude_batch_projected_per_sample_usd=claude_batch_projection / n)


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
        n = summary["completed"]
        records.append(dict(task=protocol["dataset"], variant=protocol["variant"], n=n,
            **summarize_usage(usage_path.read_text().splitlines(), n)))
    result = dict(pilot_costs=records, budget=status(),
        caveat="Pilot means are projections, not guarantees; input/output lengths vary. Synchronous-equivalent and batch-projected costs account for each call's actual transport; observed cost is reported separately. Advance reservation caps still govern every request.")
    atomic_json(RUNS / "cost_pilot.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
