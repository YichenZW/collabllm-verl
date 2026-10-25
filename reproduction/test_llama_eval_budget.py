"""Check that a new evaluation stage cannot inherit a past API allowance."""
import json
import pytest

from reproduction import budget


def test_missing_or_pending_authorization_blocks_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "RUNS", tmp_path)
    assert budget.authorized_caps() == {"openai": 0, "anthropic": 0}
    (tmp_path / "budget_authorization.json").write_text(json.dumps(dict(
        scope="llama-eval-200-200-100", status="pending",
        provider_cap_micro={"openai": 100000000, "anthropic": 100000000})))
    assert budget.authorized_caps() == {"openai": 0, "anthropic": 0}
    monkeypatch.setattr(budget, "CAP_MICRO", 0)
    monkeypatch.setattr(budget, "PROVIDER_CAP_MICRO", budget.authorized_caps())
    called = []
    original = budget.reserve
    monkeypatch.setattr(budget, "reserve", lambda model, messages, max_tokens:
        original(model, messages, max_tokens, path=tmp_path / "ledger.json"))
    with pytest.raises(budget.BudgetExceeded):
        budget.metered_call(lambda **kwargs: called.append(kwargs), model="gpt-4o",
            messages=[dict(role="user", content="hello")], max_tokens=16)
    assert not called


def test_provider_cap_stops_calls_with_total_budget_remaining(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "CAP_MICRO", 200000)
    monkeypatch.setattr(budget, "PROVIDER_CAP_MICRO", {"openai": 30000, "anthropic": 170000})
    path = tmp_path / "ledger.json"
    budget.reserve_bound("gpt-4o", 0, 100, "first", path)
    budget.reserve_bound("gpt-4o", 0, 100, "second", path)
    with pytest.raises(budget.BudgetExceeded, match="openai cap"):
        budget.reserve_bound("gpt-4o", 0, 100, "third", path)
    assert sum(x["micro"] for x in budget.read_ledger(path).values()) < 200000
