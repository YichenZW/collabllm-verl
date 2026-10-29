import concurrent.futures

from reproduction import budget


def test_provider_caps_are_independent_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "PROVIDER_CAP_MICRO", {"openai": 25000, "anthropic": 30000})
    monkeypatch.setattr(budget, "CAP_MICRO", 55000)
    ledger = tmp_path / "budget.json"
    def attempt(model):
        try:
            budget.reserve(model, [], 1, ledger)
            return True
        except budget.BudgetExceeded:
            return False
    models = ["gpt-4o", "claude-sonnet-4-5-20250929"] * 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(attempt, models))
    rows = budget.read_ledger(ledger).values()
    for vendor, limit in budget.PROVIDER_CAP_MICRO.items():
        selected = [r for r in rows if r["model"] != "probe allowance" and budget.provider(r["model"]) == vendor]
        assert len(selected) == 2
        assert sum(r["micro"] for r in selected) <= limit
