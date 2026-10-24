"""A batch-scored writing pilot must not receive a second batch discount."""
import json

import pytest

from reproduction.qwen3_costs import summarize_usage


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
