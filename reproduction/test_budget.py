import concurrent.futures

import pytest

from reproduction import budget

def test_reservations_serialize_across_workers(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite"
    monkeypatch.setattr(budget, "CAP_MICRO", 25000)
    monkeypatch.setattr(budget, "PRIOR_PROBES_MICRO", 0)
    def reserve():
        try:
            budget.reserve("gpt-4o", [], 1, path)
            return True
        except budget.BudgetExceeded:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: reserve(), range(20)))
    assert sum(results) == 2
    assert sum(r["micro"] for r in budget.read_ledger(path).values()) <= 25000

def test_settlement_releases_unused_reservation(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite"
    monkeypatch.setattr(budget, "PRIOR_PROBES_MICRO", 0)
    ident = budget.reserve("gpt-4o-mini", [], 1024, path)
    budget.settle(ident, "gpt-4o-mini", dict(prompt_tokens=100, completion_tokens=10), path)
    assert budget.read_ledger(path)[ident]["micro"] == 21

def test_unknown_model_never_gets_a_reservation(tmp_path):
    with pytest.raises(ValueError, match="unpriced"):
        budget.reserve("unknown-model", [], 1, tmp_path / "budget.sqlite")

def test_reservation_retry_is_idempotent_and_checks_bounds(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    monkeypatch.setattr(budget, "PRIOR_PROBES_MICRO", 0)
    first = budget.reserve_bound("gpt-4o-mini", 10, 1024, "request-1", path)
    assert budget.reserve_bound("gpt-4o-mini", 10, 1024, "request-1", path) == first
    budget.settle(first, "gpt-4o-mini", dict(prompt_tokens=100, completion_tokens=10), path)
    assert budget.reserve_bound("gpt-4o-mini", 10, 1024, "request-1", path) == first
    assert len(budget.read_ledger(path)) == 2  # one call and zero-cost allowance
    with pytest.raises(ValueError, match="different bounds"):
        budget.reserve_bound("gpt-4o-mini", 11, 1024, "request-1", path)
