"""Cross-process USD cap with advance reservations and conservative accounting.

All task API calls must go through metered_call or install_litellm_guard.
Prices verified 2026-10-13 from the providers' official API pricing pages.
Failures retain the full reservation because a timed-out request may be billed.
"""
import json
import math
import os
import sqlite3
import time
import uuid
import contextlib
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from reproduction.common import RUNS

# USD per million input / output tokens. Cache discounts deliberately ignored.
RATES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0),
         "claude-sonnet-4-5-20250929": (3.0, 15.0),
         "claude-sonnet-4-5-20250929@batch": (1.5, 7.5)}
CAP_MICRO = 118_000_000  # New experiment only; historical ledgers are separate.
PROVIDER_CAP_MICRO = {"openai": 99_000_000, "anthropic": 19_000_000}
PRIOR_PROBES_MICRO = 0  # Every new probe uses the reservation guard.
LOCAL_LOCK = threading.RLock()

def rpc(action, payload):
    """Use one writer across GPU nodes; an unavailable service fails closed."""
    config_path = RUNS / "budget_service.json"
    if not config_path.exists():
        raise RuntimeError("Budget coordinator is not configured; no paid request dispatched")
    config = json.loads(config_path.read_text())
    token = Path(config["token_file"]).read_text().strip()
    request = urllib.request.Request(config["url"] + "/" + action,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
        "Authorization": "Bearer " + token}, method="POST")
    # Requests to a private cluster host must not use an inherited HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(request, timeout=45) as response:
                return json.load(response)["result"]
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read())
            if body.get("type") == "BudgetExceeded":
                raise BudgetExceeded(body["error"]) from None
            raise RuntimeError("Budget coordinator rejected request: " + body.get("error", str(exc))) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise RuntimeError("Budget coordinator unavailable; no additional paid call dispatched") from None
            time.sleep(1)

def is_coordinator():
    return os.getenv("COLLABLLM_BUDGET_COORDINATOR") == "1"

class BudgetExceeded(RuntimeError):
    budget_exceeded = True

def ledger_path(path=None):
    # Resolve the production symlink before choosing the temporary file. This
    # keeps both fsync and atomic replacement on the ledger's actual filesystem.
    # The Slurm scratch Lustre filesystem can stall fsync for tens of seconds.
    return Path(path) if path is not None else (RUNS / "api_budget.json").resolve()

@contextlib.contextmanager
def locked(path):
    if is_coordinator() and path == ledger_path():
        # The service holds its singleton lease for its entire lifetime.
        # Only it may write the shared production ledger.
        with LOCAL_LOCK:
            yield
        return
    # Only isolated/local ledgers and historical migration use directory locks.
    # Production calls use the coordinator's in-process lock above.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 120
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError("API ledger lock timed out; inspect reservations before resuming")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()

def read_ledger(path):
    if path.exists():
        return json.loads(path.read_text())
    if path == ledger_path() and (RUNS / "api_budget.sqlite").exists():
        raise RuntimeError("Migrate the historical SQLite ledger before further API calls")
    return {"prior-probes": dict(model="probe allowance", status="allowance",
             micro=PRIOR_PROBES_MICRO, input_tokens=0, output_tokens=0, created=time.time())}

def write_ledger(path, calls):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(calls, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def migrate_sqlite():
    """Run once, only after all old SQLite-using workers have exited."""
    path = ledger_path()
    with locked(path):
        if path.exists():
            raise RuntimeError("JSON ledger already exists; refusing to overwrite")
        db = sqlite3.connect(str(RUNS / "api_budget.sqlite"), timeout=60)
        try:
            keys = ["model", "status", "micro", "input_tokens", "output_tokens", "created"]
            rows = db.execute("SELECT id,model,status,micro,input_tokens,output_tokens,created FROM calls").fetchall()
            calls = {row[0]: dict(zip(keys, row[1:])) for row in rows}
            assert "prior-probes" in calls
            write_ledger(path, calls)
        finally:
            db.close()

def rates(model):
    model = model.removeprefix("openai/").removeprefix("anthropic/")
    model = {"gpt-4o-mini-2024-07-18": "gpt-4o-mini",
             "gpt-4o-2024-08-06": "gpt-4o"}.get(model, model)
    if model in RATES:
        return RATES[model]
    raise ValueError(f"Refusing unpriced API model: {model}")

def provider(model):
    rates(model)
    return "anthropic" if "claude" in model else "openai"

def reserve(model, messages, max_tokens, path=None):
    rates(model)  # Reject unknown models before any request, including RPC.
    # For text-only requests, UTF-8 bytes upper-bound token count, with extra
    # allowance for message framing and provider metadata. No tools/images used.
    input_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
    ident = str(uuid.uuid4())
    if path is None and not is_coordinator():
        return rpc("reserve", dict(model=model, input_bytes=input_bytes, max_tokens=max_tokens, ident=ident))
    return reserve_bound(model, input_bytes, max_tokens, ident, path)

def reserve_bound(model, input_bytes, max_tokens, ident, path=None):
    inp, out = rates(model)
    if type(input_bytes) is not int or input_bytes < 0 or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Invalid reservation bounds")
    amount = math.ceil((input_bytes + 4096) * inp + max_tokens * out)
    path = ledger_path(path)
    with locked(path):
        calls = read_ledger(path)
        if ident in calls:
            # A lost RPC acknowledgement must not create another reservation.
            prior = calls[ident]
            if (prior["model"] != model or prior.get("reserved_micro", prior["micro"]) != amount
                    or prior.get("input_bytes") != input_bytes or prior.get("max_tokens") != max_tokens):
                raise ValueError("Reservation identifier was reused with different bounds")
            return ident
        used = sum(row["micro"] for row in calls.values())
        if used + amount > CAP_MICRO:
            raise BudgetExceeded(f"API cap reached: committed/reserved ${used/1e6:.4f}; "
                                 f"next reservation ${amount/1e6:.4f}; cap ${CAP_MICRO/1e6:.2f}")
        vendor = provider(model)
        vendor_used = sum(row["micro"] for row in calls.values()
                          if row["model"] != "probe allowance" and provider(row["model"]) == vendor)
        if vendor_used + amount > PROVIDER_CAP_MICRO[vendor]:
            raise BudgetExceeded(f"{vendor} cap reached: committed/reserved ${vendor_used/1e6:.4f}; "
                                 f"next reservation ${amount/1e6:.4f}; cap ${PROVIDER_CAP_MICRO[vendor]/1e6:.2f}")
        calls[ident] = dict(model=model, status="reserved", micro=amount, reserved_micro=amount,
                           input_bytes=input_bytes, max_tokens=max_tokens,
                           input_tokens=0, output_tokens=0, created=time.time())
        write_ledger(path, calls)
        return ident

def settle(ident, model, usage, path=None):
    usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    if path is None and not is_coordinator():
        return rpc("settle", dict(ident=ident, model=model, usage=usage))
    inp, out = rates(model)
    n_in = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    n_in += usage.get("cache_creation_input_tokens", 0) or 0
    n_in += usage.get("cache_read_input_tokens", 0) or 0
    n_out = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if type(n_in) is not int or type(n_out) is not int or n_in < 0 or n_out < 0:
        raise ValueError("Invalid provider usage; reservation retained")
    amount = math.ceil(n_in * inp + n_out * out)
    path = ledger_path(path)
    with locked(path):
        calls = read_ledger(path)
        if calls[ident]["model"] != model:
            raise ValueError("Settlement model differs from reserved model")
        reserved = calls[ident]["micro"]
        if amount > reserved:
            raise RuntimeError("Reported usage exceeded reservation; stop and inspect pricing")
        calls[ident].update(status="completed", micro=amount, input_tokens=n_in, output_tokens=n_out)
        write_ledger(path, calls)

def metered_call(fn, *, model, messages, max_tokens, **kwargs):
    ident = reserve(model, messages, max_tokens)
    # Callers disable SDK retries; retries must reserve separately.
    result = fn(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
    if result.usage is None:
        raise RuntimeError("API returned no usage; reservation retained")
    settle(ident, model, result.usage)
    return result

def install_litellm_guard():
    import litellm
    if getattr(litellm.completion, "_budget_guard", False):
        return
    original = litellm.completion
    def guarded(*args, **kwargs):
        if args:
            raise ValueError("Budget guard requires keyword arguments")
        kwargs["num_retries"] = 0
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        max_tokens = kwargs.pop("max_tokens", 4096)
        return metered_call(original, model=model, messages=messages, max_tokens=max_tokens, **kwargs)
    guarded._budget_guard = True
    litellm.completion = guarded

def status():
    calls = read_ledger(ledger_path())
    grouped = defaultdict(lambda: [0, 0, 0, 0])
    for row in calls.values():
        group = grouped[row["model"], row["status"]]
        for i, val in enumerate([1, row["micro"], row["input_tokens"], row["output_tokens"]]):
            group[i] += val
    rows = [[model, state, n, micro/1e6, n_in, n_out]
            for (model, state), (n, micro, n_in, n_out) in sorted(grouped.items())]
    vendors = {name: dict(cap_usd=cap/1e6, committed_usd=sum(
        row["micro"] for row in calls.values() if row["model"] != "probe allowance"
        and provider(row["model"]) == name)/1e6) for name, cap in PROVIDER_CAP_MICRO.items()}
    return {"cap_usd": CAP_MICRO/1e6, "committed_usd": sum(r[3] for r in rows),
            "providers": vendors, "details": rows}

if __name__ == "__main__":
    import sys
    if "--migrate-sqlite" in sys.argv:
        migrate_sqlite()
    print(json.dumps(status(), indent=2))
