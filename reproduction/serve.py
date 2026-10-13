"""Serve the official base and three adapters on one allocated GPU."""
import argparse
import json
import os
import sys
from reproduction.common import BASE, RUNS

def main():
    os.environ["OMP_NUM_THREADS"] = str(min(4, int(os.getenv("SLURM_CPUS_PER_TASK", "4"))))
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--adapter", action="append", default=[], help="Additional NAME=PATH adapter")
    args = p.parse_args()
    assets = json.loads((RUNS / "assets.json").read_text())
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", assets["base"], "--served-model-name", "base",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16", "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.88", "--max-num-seqs", "16",
        "--enable-lora", "--max-lora-rank", "32", "--max-loras", "3",
        "--max-cpu-loras", str(3 + len(args.adapter)), "--disable-log-requests", "--lora-modules"]
    command += [f"{name}={assets[name]['adapter']}" for name in ["medium", "math-hard", "bigcodebench"]]
    command += args.adapter
    os.execv(sys.executable, command)

if __name__ == "__main__":
    main()
