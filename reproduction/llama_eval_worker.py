"""Complete only the fixed, authorized Llama evaluation plan on Slurm."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

from reproduction.common import ROOT, RUNS, atomic_json
from reproduction import budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=11.5)
    args = parser.parse_args()
    plan = json.loads((RUNS / "plan.json").read_text())
    assert plan["scope"] == "llama-eval-200-200-100"
    assert budget.PROVIDER_CAP_MICRO == {"openai": 53000000, "anthropic": 45000000}
    lock = RUNS / "batch-worker.lock"
    lock.mkdir()
    atomic_json(lock / "owner.json", dict(pid=os.getpid(), hostname=socket.gethostname(), job=os.getenv("SLURM_JOB_ID")))
    roots = [str(RUNS / "eval-200"), str(RUNS / "eval-100")]
    deadline = time.monotonic() + args.hours * 3600
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and time.monotonic() < deadline:
            subprocess.run([sys.executable, "-m", "reproduction.qwen3_batch", "--roots", *roots,
                "--submit", "--max-requests", "128"], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "-m", "reproduction.qwen3_finalize", "--roots", *roots],
                           cwd=ROOT, check=True)
            progress = []
            for group in plan["groups"]:
                summary = json.loads((__import__('pathlib').Path(group["directory"]) / "summary.json").read_text())
                progress.append(dict(dataset=group["dataset"], variant=group["variant"],
                    expected=group["target"], **{k:summary.get(k,0) for k in ["generated","completed","failed"]}))
            finished = all(x["completed"] == x["expected"] and not x["failed"] for x in progress)
            state = dict(status="completed" if finished else "running", updated=time.time(),
                job=os.getenv("SLURM_JOB_ID"), progress=progress, budget=budget.status())
            atomic_json(RUNS / "progress.json", state)
            print("PROGRESS", json.dumps(state), flush=True)
            if finished:
                subprocess.run([sys.executable, "-m", "reproduction.llama_eval_report"], cwd=ROOT, check=True)
                return
            time.sleep(30)
        raise RuntimeError("Worker stopped before completion; inspect saved state before resuming")
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
