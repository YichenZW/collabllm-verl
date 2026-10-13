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
from collections import defaultdict
from pathlib import Path

from reproduction.common import RUNS

# USD per million input / output tokens. Cache discounts deliberately ignored.
RATES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0),
         "claude-sonnet-4-5-20250929": (3.0, 15.0)}
CAP_MICRO = 38_000_000  # User's $40 limit, leaving $2 of billing headroom.
PRIOR_PROBES_MICRO = 100_000  # Conservative allowance for three tiny successful probes.

class BudgetExceeded(RuntimeError):
    budget_exceeded = True

def ledger_path(path=None):
    return Path(path or RUNS / "api_budget.json")

@contextlib.contextmanager
def locked(path):
    # Atomic mkdir works across Slurm nodes on the shared filesystem; SQLite's
    # cross-node fcntl locks caused long stalls during the initial smoke test.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 120
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError("API ledger lock timed out; no request dispatched")
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

def reserve(model, messages, max_tokens, path=None):
    inp, out = rates(model)
    # For text-only requests, UTF-8 bytes upper-bound token count, with extra
    # allowance for message framing and provider metadata. No tools/images used.
    input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 4096
    amount = math.ceil(input_bound * inp + max_tokens * out)
    path = ledger_path(path)
    with locked(path):
        calls = read_ledger(path)
        used = sum(row["micro"] for row in calls.values())
        if used + amount > CAP_MICRO:
            raise BudgetExceeded(f"API cap reached: committed/reserved ${used/1e6:.4f}; "
                                 f"next reservation ${amount/1e6:.4f}; cap ${CAP_MICRO/1e6:.2f}")
        ident = str(uuid.uuid4())
        calls[ident] = dict(model=model, status="reserved", micro=amount,
                           input_tokens=0, output_tokens=0, created=time.time())
        write_ledger(path, calls)
        return ident

def settle(ident, model, usage, path=None):
    usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    inp, out = rates(model)
    n_in = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    n_in += usage.get("cache_creation_input_tokens", 0) or 0
    n_in += usage.get("cache_read_input_tokens", 0) or 0
    n_out = usage.get("completion_tokens", usage.get("output_tokens", 0))
    amount = math.ceil(n_in * inp + n_out * out)
    path = ledger_path(path)
    with locked(path):
        calls = read_ledger(path)
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
    return {"cap_usd": CAP_MICRO/1e6, "committed_usd": sum(r[3] for r in rows), "details": rows}

if __name__ == "__main__":
    import sys
    if "--migrate-sqlite" in sys.argv:
        migrate_sqlite()
    print(json.dumps(status(), indent=2))
