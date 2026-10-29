import concurrent.futures
import json
import threading
import urllib.error
import urllib.request

import pytest

from reproduction import budget
from reproduction.budget_service import make_server


def test_rpc_concurrency_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "RUNS", tmp_path)
    monkeypatch.setattr(budget, "CAP_MICRO", 25000)
    monkeypatch.setattr(budget, "PRIOR_PROBES_MICRO", 0)
    path = tmp_path / "ledger.json"
    token_path = tmp_path / "token"
    token_path.write_text("isolated-test-token")
    with pytest.raises(RuntimeError, match="not configured"):
        budget.reserve("gpt-4o", [], 1)
    server = make_server("127.0.0.1", 0, token_path.read_text(), path)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    config = dict(url=f"http://127.0.0.1:{server.server_port}", token_file=str(token_path))
    (tmp_path / "budget_service.json").write_text(json.dumps(config))
    try:
        def attempt(_):
            try:
                return budget.reserve("gpt-4o", [], 1)
            except budget.BudgetExceeded:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            ids = list(filter(None, pool.map(attempt, range(40))))
        assert len(ids) == 2
        assert sum(row["micro"] for row in budget.read_ledger(path).values()) <= 25000
        usage = dict(prompt_tokens=100, completion_tokens=10)
        budget.settle(ids[0], "gpt-4o", usage)
        budget.settle(ids[0], "gpt-4o", usage)
        assert budget.read_ledger(path)[ids[0]]["micro"] == 350
        with pytest.raises(RuntimeError, match="Invalid provider usage"):
            budget.settle(ids[1], "gpt-4o", dict(prompt_tokens=-100, completion_tokens=0))
        assert budget.read_ledger(path)[ids[1]]["micro"] == 10255
        request = urllib.request.Request(config["url"] + "/health", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=2)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
