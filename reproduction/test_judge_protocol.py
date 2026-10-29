import ast
import json
from types import SimpleNamespace as NS
from typing import Any, Union

from reproduction import judge_protocol as protocol, qwen3_batch as batch
from reproduction.common import BASE, LEGACY, TASKS, digest, legacy_prompt


def test_archived_claude_wrapper_is_exact_request_oracle():
    source = LEGACY / "collabllm/utils/api_lib/claude.py"
    definition = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "complete_text_claude")
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return NS(to_dict=lambda: {"content": [{"text": '{"score": 1}'}]})
    namespace = dict(Any=Any, Union=Union, extract_json=json.loads,
                     anthropic_client=NS(messages=NS(create=create)))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    text = "You are a conversation evaluator. Return JSON."
    namespace["complete_text_claude"](text, model=batch.MODEL, json_object=True,
                                     max_new_tokens=4096, temperature=0, max_retry=1)
    assert captured[0]["messages"] == [{"role": "user", "content": protocol.wrap_json_prompt(text, batch.MODEL)}]
    assert captured[0]["messages"][0]["content"] == protocol.CLAUDE_PREFIX + text


def test_simulator_prompts_do_not_trigger_archived_gpt_prefix():
    for config in TASKS.values():
        prompt = legacy_prompt(f"user_simulator_cot/{config['task']}.txt").format(
            question="test", answer="test", chat_history="")
        assert "json" in prompt.lower()
        assert protocol.wrap_json_prompt(prompt, config["user"]) == prompt


def test_same_dialogue_in_two_stages_is_only_charged_once(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "RUNS", tmp_path)
    roots = [tmp_path / "eval-32", tmp_path / "eval-200"]
    sample = dict(generated=True, completed=False, input=dict(prompt="question", completion="answer"),
                  chat=[dict(role="user", content="question"), dict(role="assistant", content="answer")])
    for root in roots:
        directory = root / "math-hard/base"
        (directory / "samples").mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(dict(dataset="math-hard", judge=batch.MODEL,
            underlying_model=BASE, scoring_protocol=protocol.VERSION)))
        (directory / "samples/0001.json").write_text(json.dumps(sample))
    assert list(batch.candidates(roots, [])) == []
    for root in roots:
        (root / "math-hard/base/generation_complete.json").write_text(json.dumps(dict(status="complete")))
    requests = list(batch.candidates(roots, []))
    assert len(requests) == 1
    assert requests[0]["params"]["messages"][0]["content"].startswith(protocol.CLAUDE_PREFIX)
    active = [dict(scoring_protocol=protocol.VERSION,
                   items=[dict(requests[0], status="submitted")])]
    assert list(batch.candidates(roots, active)) == []
    payload = dict(interactivity=dict(score=3), accuracy=dict(score=1))
    protocol.save_score(requests[0]["params"], payload, dict(source="test-paid-request"))
    assert protocol.reuse_cached(roots) == 2
    assert list(batch.candidates(roots, [])) == []
    for root in roots:
        state = json.loads((root / "math-hard/base/samples/0001.json").read_text())
        assert state["judge"] == payload
        assert state["score_requests"]["judge"] == digest(requests[0]["params"])
        assert state["input"] == sample["input"] and state["chat"] == sample["chat"]
