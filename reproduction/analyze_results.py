"""Offline paired outcomes and code split sensitivity; no new API calls."""
import json
import math
from collections import Counter
from itertools import combinations

from reproduction.common import RUNS, atomic_json
from reproduction.report import wilson_interval


def completed_samples(row):
    samples = {}
    for path in (RUNS / row["run"] / "samples").glob("*.json"):
        sample = json.loads(path.read_text())
        if sample.get("completed"):
            if sample["index"] in samples:
                raise ValueError(f"Duplicate completed sample index: {row['run']}")
            samples[sample["index"]] = sample
    if len(samples) != row["n"]:
        raise ValueError(f"Summary/sample count mismatch: {row['run']}")
    return samples


def paired_comparisons(primary, task, metric, contrasts):
    """Use the full paired sample set and adjust within this contrast family."""
    results, cached = [], {}
    for a, b in contrasts:
        if not all((task, variant) in primary for variant in (a, b)):
            continue
        for variant in (a, b):
            if variant not in cached:
                cached[variant] = completed_samples(primary[task, variant])
        left, right = cached[a], cached[b]
        if not left or left.keys() != right.keys():
            raise ValueError("Paired sample indices differ or are empty")
        indices = sorted(left)
        if any(left[i]["input"] != right[i]["input"] for i in indices):
            raise ValueError("Paired inputs differ")
        outcomes = [(left[i]["metrics"][metric], right[i]["metrics"][metric])
                    for i in indices]
        if any(value not in (0, 1) for pair in outcomes for value in pair):
            raise ValueError("Paired exact tests require binary outcomes")
        pairs = Counter(outcomes)
        lost, gained = pairs[1, 0], pairs[0, 1]
        discordant = lost + gained
        p = min(1.0, 2 * sum(math.comb(discordant, k)
                for k in range(min(lost, gained) + 1)) / 2 ** discordant)
        results.append(dict(a=a, b=b, n=len(indices),
            a_run=primary[task, a]["run"], b_run=primary[task, b]["run"],
            a_only_correct=lost, b_only_correct=gained,
            both_correct=pairs[1, 1], neither_correct=pairs[0, 0],
            difference_percentage_points=100 * (gained - lost) / len(indices),
            mcnemar_exact_two_sided_p=p))
    adjusted = 0.0
    for rank, row in enumerate(sorted(results, key=lambda r: r["mcnemar_exact_two_sided_p"])):
        adjusted = min(1.0, max(adjusted,
            (len(results) - rank) * row["mcnemar_exact_two_sided_p"]))
        row["mcnemar_holm_adjusted_p"] = adjusted
        row["holm_family_size"] = len(results)
    return results


def main():
    report = json.loads((RUNS / "report.json").read_text())
    primary = {}
    for row in report["primary_results"]:
        key = row["task"], row["variant"]
        if key in primary:
            raise ValueError("Multiple protocols; analyze them separately")
        primary[key] = row
    math_pairs = paired_comparisons(primary, "math-hard", "accuracy", [
        ("base", "sft"), ("base", "offline_dpo"),
        ("sft", "offline_dpo"), ("proactive", "offline_dpo")])
    code_pairs = paired_comparisons(primary, "bigcodebench", "pass_rate",
        combinations(("base", "proactive", "sft", "offline_dpo"), 2))
    audit = json.loads((RUNS / "code_split_audit.json").read_text())
    seen = {r["task_id"]: r["seen_in_archived_conversations"] for r in audit["samples"]}
    code = []
    reference_sensitivity = []
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
        flagged = [s for s in samples.values() if s.get("code_reference_issue")]
        reference_clean = [s for s in samples.values() if not s.get("code_reference_issue")]
        n = len(reference_clean)
        passed = sum(s["metrics"]["pass_rate"] for s in reference_clean)
        interval = wilson_interval(passed / n, n) if n else (None, None)
        reference_sensitivity.append(dict(variant=variant, run=row["run"], total_n=len(samples),
            flagged_task_ids=[s["input"]["metadata"]["task_id"] for s in flagged],
            reference_pass_subset_n=n, reference_pass_subset_passed=passed,
            reference_pass_subset_percent=100 * passed / n if n else None,
            reference_pass_subset_wilson95_percent=[100*x if x is not None else None for x in interval]))
    result = dict(math_paired=math_pairs, code_paired=code_pairs,
        multiple_comparison_adjustment=dict(method="Holm", families=dict(
            math_paired=len(math_pairs), code_paired=len(code_pairs))),
        code_nonoverlap_sensitivity=code,
        code_reference_issue_sensitivity=reference_sensitivity,
        limitations=["One sampled conversation per task/model; uncertainty does not include variation between reruns",
                     "Exact McNemar tests use identical task inputs; Holm adjustment is separate for the listed math and code contrast families",
                     "Code overlap reconstructed from archived source, not public adapter provenance",
                     "Non-overlap code rows are a sensitivity analysis, not a replacement paper split",
                     "Primary scores retain all original tests, including audited reference inconsistencies; reference-pass subsets are sensitivity analyses only",
                     "Judge/prompt differences still prevent an identical-protocol reproduction"])
    atomic_json(RUNS / "statistical_analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
