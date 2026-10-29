"""Cost-only pilot sizing from actual transport prices, without reading scores."""
import json
from collections import defaultdict

from reproduction.budget import rates, status
from reproduction.common import RUNS, atomic_json
from reproduction.judge_protocol import VERSION, load_score, score_params


def scoring_costs_from_samples(directory):
    """Measure current scoring even when a paid result was reused from another stage.

    This is a per-sample expansion estimate, not an invoice total: overlapping
    stages refer to the same cache entries. The ledger alone totals all spending.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if manifest.get("scoring_protocol") != VERSION or summary.get("scoring_protocol") != VERSION:
        raise ValueError("Cost sizing requires the corrected scoring protocol")
    if summary["completed"] != summary["expected"] or summary["failed"]:
        raise ValueError("Cost sizing requires a completed evaluation stage")
    task, judge = manifest["dataset"], manifest["judge"]
    lines, keys = [], []
    samples = list((directory / "samples").glob("*.json"))
    if len(samples) != summary["expected"]:
        raise ValueError("Unexpected sample count for cost sizing")
    for path in samples:
        sample = json.loads(path.read_text())
        if not sample.get("completed") or sample.get("scoring_protocol") != VERSION:
            raise ValueError("Sample scoring is not current and complete")
        for kind in (["judge"] if task == "math-hard" else ["judge", "extract"]):
            cached = load_score(score_params(task, sample, kind, judge))
            if cached is None or sample.get("score_requests", {}).get(kind) != cached["request_key"]:
                raise ValueError("Missing current scoring cost provenance")
            metadata = cached["metadata"]
            lines.append(json.dumps(dict(kind=kind, model=judge, usage=metadata["usage"],
                                         transport=metadata["transport"])))
            keys.append(cached["request_key"])
    result = summarize_usage(lines, summary["completed"])
    return dict(n=summary["completed"], scoring_protocol=VERSION, request_keys=keys,
                **{k: v for k, v in result.items() if k != "openai_per_sample_usd"})


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
        if protocol.get("scoring_protocol") != VERSION or summary.get("scoring_protocol") != VERSION:
            continue
        n = summary["completed"]
        historical = summarize_usage(usage_path.read_text().splitlines(), n)
        current = scoring_costs_from_samples(manifest.parent)
        records.append(dict(task=protocol["dataset"], variant=protocol["variant"],
            openai_per_sample_usd=historical["openai_per_sample_usd"],
            usage_history_usd=historical["usd"], **current))
    result = dict(pilot_costs=records, budget=status(),
        caveat="Pilot means are projections, not guarantees; input/output lengths vary. Current scoring costs come from exact cached requests, including cross-stage reuse, and exclude superseded scoring from projections. Do not sum overlapping stages as invoice totals; the ledger includes all old and new charges. Transport discounts are applied once. Advance reservation caps govern every request.")
    atomic_json(RUNS / "cost_pilot.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
