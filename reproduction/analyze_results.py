"""Offline paired math outcomes and code split sensitivity; no new API calls."""
import json
import math
from collections import Counter

from reproduction.common import RUNS, atomic_json
from reproduction.report import wilson_interval


def completed_samples(row):
    samples = {}
    for path in (RUNS / row["run"] / "samples").glob("*.json"):
        sample = json.loads(path.read_text())
        if sample.get("completed"):
            samples[sample["index"]] = sample
    if len(samples) != row["n"]:
        raise ValueError(f"Summary/sample count mismatch: {row['run']}")
    return samples


def main():
    report = json.loads((RUNS / "report.json").read_text())
    primary = {}
    for row in report["primary_results"]:
        key = row["task"], row["variant"]
        if key in primary:
            raise ValueError("Multiple protocols; analyze them separately")
        primary[key] = row
    math_pairs = []
    for a, b in [("base", "sft"), ("base", "offline_dpo"),
                 ("sft", "offline_dpo"), ("proactive", "offline_dpo")]:
        if not all(("math-hard", v) in primary for v in [a, b]):
            continue
        left = completed_samples(primary["math-hard", a])
        right = completed_samples(primary["math-hard", b])
        indices = sorted(left.keys() & right.keys())
        if any(left[i]["input"] != right[i]["input"] for i in indices):
            raise ValueError("Paired inputs differ")
        pairs = Counter((int(left[i]["metrics"]["accuracy"]),
                         int(right[i]["metrics"]["accuracy"])) for i in indices)
        lost, gained = pairs[1, 0], pairs[0, 1]
        discordant = lost + gained
        p = min(1.0, 2 * sum(math.comb(discordant, k)
                for k in range(min(lost, gained) + 1)) / 2 ** discordant)
        math_pairs.append(dict(a=a, b=b, n=len(indices), a_only_correct=lost,
            b_only_correct=gained, both_correct=pairs[1, 1], neither_correct=pairs[0, 0],
            difference_percentage_points=100 * (gained - lost) / len(indices),
            mcnemar_exact_two_sided_p=p))
    audit = json.loads((RUNS / "code_split_audit.json").read_text())
    seen = {r["task_id"]: r["seen_in_archived_conversations"] for r in audit["samples"]}
    code = []
    for (task, variant), row in primary.items():
        if task != "bigcodebench":
            continue
        samples = completed_samples(row)
        clean = [s for s in samples.values()
                 if not seen[s["input"]["metadata"]["task_id"]]]
        n = len(clean)
        passed = sum(s["metrics"]["pass_rate"] for s in clean)
        interval = wilson_interval(passed / n, n) if n else (None, None)
        code.append(dict(variant=variant, run=row["run"], total_n=len(samples),
            reconstructed_overlap_n=len(samples) - n, nonoverlap_n=n,
            nonoverlap_passed=passed, nonoverlap_pass_rate_percent=100 * passed / n if n else None,
            nonoverlap_wilson95_percent=[100*x if x is not None else None for x in interval]))
    result = dict(math_paired=math_pairs, code_nonoverlap_sensitivity=code,
        limitations=["One sampled conversation per task/model; no multiple-comparison adjustment",
                     "Code overlap reconstructed from archived source, not public adapter provenance",
                     "Non-overlap code rows are a sensitivity analysis, not a replacement paper split",
                     "Judge/prompt differences still prevent an identical-protocol reproduction"])
    atomic_json(RUNS / "statistical_analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
