import copy

import pytest
import tiktoken

from reproduction import judge_protocol, qwen3_audit
from reproduction.common import digest


def paid_math_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_protocol, "RUNS", tmp_path)
    sample = dict(input=dict(prompt="What is one plus one?", completion="2"),
        chat=[dict(role="user", content="What is one plus one?"),
              dict(role="assistant", content="2")], scoring_protocol=judge_protocol.VERSION)
    manifest = dict(judge="claude-sonnet-4-5-20250929")
    params = judge_protocol.score_params("math-hard", sample, "judge", manifest["judge"])
    payload = dict(interactivity=dict(score=2), accuracy=dict(score=1))
    record = judge_protocol.save_score(params, payload,
        dict(usage=dict(input_tokens=1000, output_tokens=100), transport="batch"))
    judge_protocol.apply_score(sample, "judge", record)
    encoder = tiktoken.get_encoding("cl100k_base")
    sample["metrics"] = dict(accuracy=1.0, itr=-1.0, itr_div3=2/3,
                            tokens_k=len(encoder.encode("2 ")) / 1000)
    return sample, manifest, encoder


def test_actual_score_requires_exact_paid_request_and_original_metrics(tmp_path, monkeypatch):
    sample, manifest, encoder = paid_math_sample(tmp_path, monkeypatch)
    assert qwen3_audit.sample_metrics("math-hard", sample, manifest, encoder) == sample["metrics"]
    corruptions = [
        lambda row: row.update(scoring_protocol=judge_protocol.LEGACY_VERSION),
        lambda row: row["score_requests"].update(judge=digest("another dialogue")),
        lambda row: row["judge"]["accuracy"].update(score=0),
        lambda row: row["metrics"].update(accuracy=0),
        lambda row: row["metrics"].update(itr=2/3),
        lambda row: row["chat"][-1].update(content="different answer"),
    ]
    for corrupt in corruptions:
        changed = copy.deepcopy(sample)
        corrupt(changed)
        with pytest.raises(ValueError):
            qwen3_audit.sample_metrics("math-hard", changed, manifest, encoder)
