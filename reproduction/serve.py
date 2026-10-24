"""Serve the official base and three adapters on one allocated GPU."""
import argparse
import json
import os
import sys
from pathlib import Path
from reproduction.common import RUNS, atomic_json

def main():
    os.environ["OMP_NUM_THREADS"] = str(min(4, int(os.getenv("SLURM_CPUS_PER_TASK", "4"))))
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--adapter", action="append", default=[], help="Additional NAME=PATH adapter")
    p.add_argument("--max-model-len", type=int, default=None)
    args = p.parse_args()
    assets = json.loads((RUNS / "assets.json").read_text())
    native_context = json.loads((Path(assets["base"]) / "config.json").read_text())["max_position_embeddings"]
    context = args.max_model_len or native_context
    if not 0 < context <= native_context:
        raise ValueError("Requested context must fit the pinned model's native context")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", assets["base"], "--served-model-name", "base",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16", "--max-model-len", str(context),
        "--gpu-memory-utilization", "0.88", "--max-num-seqs", "16",
        "--enable-lora", "--max-lora-rank", "32", "--max-loras", "3",
        "--max-cpu-loras", str(max(3, len(args.adapter))), "--disable-log-requests",
        "--generation-config", "vllm"]
    if args.adapter:
        command += ["--lora-modules"] + args.adapter
    atomic_json(RUNS / "logs" / f"server-config-{os.getenv('SLURM_JOB_ID', 'local')}-{args.port}.json",
        dict(command=command, model_revision=assets["revision"], native_context=native_context,
             max_model_len=context, enable_thinking=assets["enable_thinking"]))
    os.execv(sys.executable, command)

if __name__ == "__main__":
    main()
