"""Finish a fixed evaluation stage as GPU jobs produce saved conversations.

Only the existing reserved batch scorer may make paid calls. This worker never
submits or restarts training jobs, and local finalization has no paid fallback.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid

from reproduction.common import ROOT, RUNS, atomic_json, source_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["math-hard", "bigcodebench", "medium"], required=True)
    parser.add_argument("--variants", nargs="+", default=["base", "proactive", "sft", "offline_dpo"])
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--hours", type=float, default=18)
    args = parser.parse_args()
    if args.limit <= 0 or args.interval < 30 or not 0 < args.hours <= 48:
        raise ValueError("Invalid stage or polling bounds")
    root = RUNS / f"eval-{args.limit}"
    lease = RUNS / "batch-worker.lock"
    lease.mkdir()  # One stage controller; never steal another worker's lease.
    state_path = RUNS / "batch_worker.json"
    state = dict(instance=uuid.uuid4().hex, pid=os.getpid(), host=socket.getfqdn(),
                 started=time.time(), root=str(root), tasks=args.tasks, variants=args.variants,
                 status="running", code=source_identity("reproduction/qwen3_batch_worker.py"))
    atomic_json(lease / "owner.json", state)
    deadline = time.monotonic() + args.hours * 3600
    env = dict(os.environ, ENABLE_COLLABLLM_LOGGING="0", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    last_progress = None
    try:
        while time.monotonic() < deadline:
            for module, arguments in [
                ("qwen3_batch", ["--roots", str(root), "--submit", "--max-requests", "64"]),
                ("qwen3_finalize", ["--roots", str(root)]),
                ("qwen3_report", []), ("qwen3_status", []),
            ]:
                # Full output is retained in the worker's redirected log.
                subprocess.run([sys.executable, "-m", f"reproduction.{module}", *arguments],
                               cwd=ROOT, env=env, check=True)
            progress = {}
            for task in args.tasks:
                for variant in args.variants:
                    path = root / task / variant / "summary.json"
                    summary = json.loads(path.read_text()) if path.exists() else {}
                    progress[f"{task}/{variant}"] = {k: summary.get(k) for k in
                        ["expected", "generated", "completed", "failed"]}
            state.update(updated=time.time(), progress=progress)
            complete = all(row["expected"] == args.limit and row["completed"] == args.limit
                           and row["failed"] == 0 for row in progress.values())
            if complete:
                state["status"] = "completed"
            atomic_json(state_path, state)
            if progress != last_progress or complete:
                print("STAGE_PROGRESS", json.dumps(state), flush=True)
                last_progress = progress
            if complete:
                return
            time.sleep(args.interval)
        state.update(status="watch_timeout", updated=time.time())
        atomic_json(state_path, state)
        raise TimeoutError("Stage remains incomplete; inspect existing jobs before resuming")
    except BaseException as exc:
        state.update(status="stopped", error=type(exc).__name__ + ": " + str(exc), updated=time.time())
        atomic_json(state_path, state)
        raise
    finally:
        (lease / "owner.json").unlink()
        lease.rmdir()


if __name__ == "__main__":
    main()
