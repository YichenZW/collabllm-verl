import json
from types import SimpleNamespace as NS

import pytest

from reproduction import budget, qwen3_batch as batch
from reproduction import judge_protocol


def isolate(monkeypatch, path):
    monkeypatch.setattr(judge_protocol, "RUNS", path.parent / "runs")
    reserve, settle = budget.reserve, budget.settle
    monkeypatch.setattr(budget, "reserve", lambda model, messages, maximum:
        reserve(model, messages, maximum, path))
    monkeypatch.setattr(budget, "settle", lambda ident, model, usage:
        settle(ident, model, usage, path))


def item(tmp_path):
    sample = tmp_path / "eval/math-hard/base/samples/0001.json"
    sample.parent.mkdir(parents=True)
    sample.write_text(json.dumps(dict(generated=True, chat=[], input={})))
    return dict(sample=str(sample), kind="judge", field="judge", dataset="math-hard", custom_id="sample-1",
        params=dict(model=batch.MODEL, max_tokens=4096, temperature=0,
                    messages=[dict(role="user", content="Evaluate this conversation.")]))


def test_batch_is_reserved_before_dispatch_and_settled_once(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    isolate(monkeypatch, ledger)
    candidate = item(tmp_path)
    manifest = tmp_path / "batch.json"
    provider = NS(id="batch-1", processing_status="ended", model_dump=lambda **_: dict(id="batch-1"))
    def create(requests):
        calls = [r for r in budget.read_ledger(ledger).values() if r["status"] == "reserved"]
        assert len(calls) == len(requests) == 1
        assert requests[0]["params"]["model"] == batch.MODEL
        assert json.loads(manifest.read_text())["state"] == "submitting"
        return provider
    payload = dict(interactivity=dict(score=3), accuracy=dict(score=1))
    message = NS(model=batch.MODEL, usage=NS(model_dump=lambda: dict(input_tokens=512, output_tokens=64)),
                 content=[NS(type="text", text=json.dumps(payload))])
    result = NS(custom_id="sample-1", result=NS(type="succeeded", message=message),
                model_dump=lambda **_: dict(custom_id="sample-1", result="test-success"))
    client = NS(messages=NS(batches=NS(create=create, retrieve=lambda _: provider, results=lambda _: [result])))
    batch.submit(client, [candidate], manifest)
    batch.refresh(client, manifest)
    batch.refresh(client, manifest)
    rows = [r for r in budget.read_ledger(ledger).values() if r["model"] != "probe allowance"]
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert rows[0]["micro"] == 1248  # 512 * $1.5/M + 64 * $7.5/M
    sample = json.loads(__import__('pathlib').Path(candidate["sample"]).read_text())
    assert sample["judge"] == payload
    usage_file = __import__('pathlib').Path(candidate["sample"]).parent.parent / "api_usage.jsonl"
    assert len(usage_file.read_text().splitlines()) == 1


def test_unknown_batch_submission_keeps_reservations(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    isolate(monkeypatch, ledger)
    def timeout(**_):
        raise TimeoutError("Unknown dispatch status")
    client = NS(messages=NS(batches=NS(create=timeout)))
    manifest = tmp_path / "batch.json"
    with pytest.raises(TimeoutError):
        batch.submit(client, [item(tmp_path)], manifest)
    assert json.loads(manifest.read_text())["state"] == "submitting"
    assert any(r["status"] == "reserved" and r["micro"] > 0 for r in budget.read_ledger(ledger).values())
    with pytest.raises(RuntimeError, match="Unconfirmed batch"):
        batch.refresh(client, manifest)


def test_old_batch_completion_never_overwrites_corrected_score(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    isolate(monkeypatch, ledger)
    candidate = item(tmp_path)
    manifest = tmp_path / "batch.json"
    provider = NS(id="old-batch", processing_status="ended", model_dump=lambda **_: dict(id="old-batch"))
    payload = dict(interactivity=dict(score=3), accuracy=dict(score=1))
    message = NS(model=batch.MODEL, usage=NS(model_dump=lambda: dict(input_tokens=512, output_tokens=64)),
                 content=[NS(type="text", text=json.dumps(payload))])
    result = NS(custom_id="sample-1", result=NS(type="succeeded", message=message),
                model_dump=lambda **_: dict(custom_id="sample-1", result="old-success"))
    client = NS(messages=NS(batches=NS(create=lambda **_: provider, retrieve=lambda _: provider,
                                     results=lambda _: [result])))
    batch.submit(client, [candidate], manifest)
    record = json.loads(manifest.read_text());record.pop("scoring_protocol")
    manifest.write_text(json.dumps(record))
    from pathlib import Path
    path = Path(candidate["sample"])
    state = json.loads(path.read_text())
    state.update(judge=dict(interactivity=dict(score=2), accuracy=dict(score=0)),
                 scoring_protocol=judge_protocol.VERSION)
    path.write_text(json.dumps(state))
    batch.refresh(client, manifest)
    batch.refresh(client, manifest)
    assert json.loads(path.read_text()) == state
    assert json.loads(manifest.read_text())["items"][0]["parsed_payload"] == payload
    assert len((path.parent.parent / "api_usage.jsonl").read_text().splitlines()) == 1
