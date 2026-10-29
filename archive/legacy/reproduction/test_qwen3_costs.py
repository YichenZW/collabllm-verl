"""A batch-scored writing pilot must not receive a second batch discount."""
import json
import pytest

from reproduction.qwen3_costs import summarize_usage
from reproduction.qwen3_costs import scoring_costs_from_samples
from reproduction import judge_protocol


def test_expansion_sizing_uses_current_reused_scores_not_obsolete_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_protocol, "RUNS", tmp_path)
    directory = tmp_path / "eval-2/math-hard/base"
    (directory / "samples").mkdir(parents=True)
    model = "claude-sonnet-4-5-20250929"
    (directory / "manifest.json").write_text(json.dumps(dict(dataset="math-hard", judge=model,
        scoring_protocol=judge_protocol.VERSION)))
    (directory / "summary.json").write_text(json.dumps(dict(expected=2, completed=2, failed=0,
        scoring_protocol=judge_protocol.VERSION)))
    (directory / "api_usage.jsonl").write_text('obsolete and unrelated scoring charges\n')
    for i, transport in enumerate(["batch", "synchronous"]):
        sample = dict(completed=True, input=dict(prompt=str(i), completion="answer"), chat=[])
        params = judge_protocol.score_params("math-hard", sample, "judge", model)
        cached = judge_protocol.save_score(params, dict(interactivity=dict(score=3), accuracy=dict(score=1)),
            dict(transport=transport, usage=dict(input_tokens=512*(i+1), output_tokens=64*(i+1))))
        judge_protocol.apply_score(sample, "judge", cached)
        (directory / "samples" / f"{i}.json").write_text(json.dumps(sample))
    costs = scoring_costs_from_samples(directory)
    assert costs["n"] == 2 and costs["calls"]["judge"] == 2
    assert costs["claude_observed_per_sample_usd"] == pytest.approx(0.00312)
    assert costs["claude_batch_projected_per_sample_usd"] == pytest.approx(0.001872)


def test_mixed_transport_projection_and_actual_cost():
    common = dict(model="claude-sonnet-4-5-20250929", kind="judge",
                  usage=dict(input_tokens=1000, output_tokens=100))
    synchronous = summarize_usage([json.dumps(common)], 1)
    batch = summarize_usage([json.dumps(dict(common, transport="batch"))], 1)
    mixed = summarize_usage([json.dumps(common), json.dumps(dict(common, transport="batch"))], 2)
    assert synchronous["claude_observed_per_sample_usd"] == pytest.approx(0.0045)
    assert batch["claude_observed_per_sample_usd"] == pytest.approx(0.00225)
    assert mixed["claude_observed_per_sample_usd"] == pytest.approx(0.003375)
    for result in [synchronous, batch, mixed]:
        assert result["claude_batch_projected_per_sample_usd"] == pytest.approx(0.00225)
        assert result["claude_synchronous_per_sample_usd"] == pytest.approx(0.0045)
