import json

import pytest

from reproduction import qwen3_report as report
from reproduction.judge_protocol import VERSION, LEGACY_VERSION


def group(root, n, incomplete=None, different_input=None):
    rows = []
    for variant in report.VARIANTS:
        run = f"eval-{n}/math-hard/{variant}"
        directory = root / run
        directory.mkdir(parents=True)
        protocol = dict(test_sha256="other" if variant == different_input else f"prefix-{n}",
            underlying_model="Qwen/Qwen3-4B", model_revision="pinned", enable_thinking=False,
            generation_config="vllm", seed=42, protocol="paper", scoring_protocol=VERSION)
        (directory / "manifest.json").write_text(json.dumps(protocol))
        rows.append(dict(run=run, task="math-hard", variant=variant, n=n - (variant == incomplete),
                         expected=n, failed=int(variant == incomplete), user="user", judge="judge",
                         scoring_protocol=VERSION))
    return rows


def test_incomplete_larger_run_does_not_replace_paired_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    rows = group(tmp_path, 8) + group(tmp_path, 32, incomplete="sft")
    selected = report.paired_results(rows)
    assert len(selected) == 4 and {r["n"] for r in selected} == {8}


def test_different_test_inputs_reject_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="different inputs"):
        report.paired_results(group(tmp_path, 8, different_input="offline_dpo"))


def test_larger_obsolete_scoring_cannot_replace_current_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    current = group(tmp_path, 8)
    obsolete = group(tmp_path, 32)
    for row in obsolete:
        row["scoring_protocol"] = LEGACY_VERSION
    selected = report.paired_results(current + obsolete)
    assert len(selected) == 4 and {row["n"] for row in selected} == {8}
