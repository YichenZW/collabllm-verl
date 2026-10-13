"""Independent evaluation jobs may refresh the same report simultaneously."""
import concurrent.futures
import json

from reproduction.common import atomic_json


def test_concurrent_report_writers(tmp_path):
    path = tmp_path / "report.json"
    def write(worker):
        payload = dict(worker=worker, values=[worker] * 1000)
        for _ in range(20):
            atomic_json(path, payload)
            saved = json.loads(path.read_text())
            assert saved["values"] == [saved["worker"]] * 1000
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))
    assert not list(tmp_path.glob(".*.tmp"))
