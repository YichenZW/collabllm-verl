import json
import sys
from types import SimpleNamespace as NS

import pytest

from reproduction import evaluate
from reproduction import judge_protocol


def test_generation_and_precomputed_scoring_do_not_call_paid_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RUNS", tmp_path)
    monkeypatch.setattr(judge_protocol, "RUNS", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "assets.json").write_text(json.dumps(dict(model_id="Qwen/Qwen3-4B", revision="test", enable_thinking=False)))
    row = dict(index=1, prompt="2+2?", completion="4", metadata={})
    (tmp_path / "data/math-hard-test.json").write_text(json.dumps([row]))
    calls = []
    def user(**kwargs):
        content = "What is 2+2?" if len(calls) == 1 else "[[TERMINATE CHAT]]"
        return NS(model="gpt-4o-mini", choices=[NS(message=NS(content=json.dumps(dict(response=content))))],
            usage=NS(model_dump=lambda: dict(prompt_tokens=10, completion_tokens=10)))
    def local(**kwargs):
        return NS(choices=[NS(message=NS(content="4"))])
    def forbidden(**kwargs):
        raise AssertionError("Paid judge called during generation or precomputed scoring")
    monkeypatch.setattr(evaluate, "OpenAI", lambda **kwargs: NS(chat=NS(completions=NS(
        create=local if kwargs.get("api_key") == "local" else user))))
    monkeypatch.setattr(evaluate.anthropic, "Anthropic", lambda **kwargs: NS(messages=NS(create=forbidden)))
    def metered(fn, **kwargs):
        calls.append(kwargs["model"])
        return fn(**kwargs)
    monkeypatch.setattr(evaluate, "metered_call", metered)
    out = tmp_path / "eval/math-hard/base"
    argv = ["evaluate", "--dataset", "math-hard", "--variant", "base", "--model", "base",
            "--workers", "1", "--limit", "1", "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--phase", "generate"])
    evaluate.main()
    sample = out / "samples/0001.json"
    state = json.loads(sample.read_text())
    assert state["generated"] and not state["completed"] and len(state["chat"]) == 3
    assert calls == ["gpt-4o-mini", "gpt-4o-mini"]
    monkeypatch.setattr(sys, "argv", argv + ["--phase", "score"])
    with pytest.raises(RuntimeError, match="samples failed"):
        evaluate.main()
    assert len(calls) == 2
    state["judge"] = dict(interactivity=dict(score=3), accuracy=dict(score=1))
    sample.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="samples failed"):
        evaluate.main()
    assert len(calls) == 2  # Obsolete fields neither count as scores nor trigger paid fallback.
    payload = state.pop("judge")
    params = judge_protocol.score_params("math-hard", state, "judge", "claude-sonnet-4-5-20250929")
    cached = judge_protocol.save_score(params, payload, dict(transport="batch", usage={}))
    judge_protocol.apply_score(state, "judge", cached)
    sample.write_text(json.dumps(state))
    evaluate.main()
    assert len(calls) == 2
    assert json.loads(sample.read_text())["completed"]
    assert json.loads((out / "summary.json").read_text())["completed"] == 1
