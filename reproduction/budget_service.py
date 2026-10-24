"""Small authenticated, single-writer budget coordinator for Slurm workers.

No provider API keys or conversation text cross this endpoint. It accepts only
reservation bounds and provider usage. The existing ledger remains authoritative.
"""
import argparse
import hmac
import json
import os
import secrets
import signal
import socket
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reproduction import budget
from reproduction.common import RUNS, atomic_json


def make_server(host, port, token, path=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass  # Never log authentication headers.

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid budget request size")
                payload = json.loads(self.rfile.read(size))
                # Serialize handlers before accessing the ledger. The default
                # production path skips per-request filesystem locking.
                with budget.LOCAL_LOCK:
                    if self.path == "/reserve":
                        result = budget.reserve_bound(path=path, **payload)
                    elif self.path == "/settle":
                        result = budget.settle(path=path, **payload)
                    elif self.path == "/health":
                        result = {"cap_usd": budget.CAP_MICRO / 1e6,
                                  "provider_caps_usd": {name: cap / 1e6 for name, cap
                                                        in budget.PROVIDER_CAP_MICRO.items()},
                                  "records": len(budget.read_ledger(budget.ledger_path(path)))}
                    else:
                        raise ValueError("Unknown budget action")
                code, response = 200, {"result": result}
            except Exception as exc:
                code = 409 if isinstance(exc, budget.BudgetExceeded) else 400
                response = {"type": type(exc).__name__, "error": str(exc)}
            data = json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Reservation/settlement already persisted; RPC is idempotent.

    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = False
    return Server((host, port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    lease = RUNS / "budget-coordinator.lock"
    lease.mkdir()  # Never steal an existing coordinator lease automatically.
    instance = uuid.uuid4().hex
    endpoint = RUNS / "budget_service.json"
    try:
        if (RUNS / "api_budget.json.lock").exists():
            raise RuntimeError("Old ledger lock remains; verify old workers stopped before handover")
        os.environ["COLLABLLM_BUDGET_COORDINATOR"] = "1"
        token_path = RUNS / ".budget_service_token"
        try:
            with os.fdopen(os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
                f.write(secrets.token_urlsafe(48))
        except FileExistsError:
            pass
        token = token_path.read_text().strip()
        hostname = socket.getfqdn()
        server = make_server("0.0.0.0", args.port, token)
        address = f"http://{hostname}:{server.server_port}"
        atomic_json(lease / "owner.json", dict(instance=instance, host=hostname,
            pid=os.getpid(), job=os.getenv("SLURM_JOB_ID")))
        atomic_json(endpoint, dict(instance=instance, url=address, token_file=str(token_path),
            job=os.getenv("SLURM_JOB_ID")))
        print("BUDGET_COORDINATOR_READY", json.dumps(dict(url=address, records=len(budget.read_ledger(budget.ledger_path())),
            committed_usd=budget.status()["committed_usd"], cap_usd=budget.CAP_MICRO / 1e6)), flush=True)
        def stop(*_):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            server.server_close()
    finally:
        if endpoint.exists() and json.loads(endpoint.read_text()).get("instance") == instance:
            endpoint.unlink()
        (lease / "owner.json").unlink(missing_ok=True)
        lease.rmdir()


if __name__ == "__main__":
    main()
