"""Verify canonical solutions before spending API budget on coding samples."""
import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path
from reproduction.common import ROOT, RUNS, atomic_json
from reproduction.code_executor import harness_identity, run_check

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=RUNS / "code_validation",
                   help="Separate cache for rechecking canonical solutions on a new node")
    args = p.parse_args()
    rows = json.loads((RUNS / "data/bigcodebench-test.json").read_text())[:args.limit]
    env = {k: v for k, v in os.environ.items() if k in ["PATH", "LANG", "LC_ALL", "TZ"]}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1")
    def run(row):
        path = args.output / f"{row['index']:04d}.json"
        gt = row["metadata"]
        result = json.loads(path.read_text()) if path.exists() else None
        if result is None or result.get("harness") != harness_identity():
            previous = result
            try:
                result = run_check(dict(answer=gt["answer"], test=gt["test"],
                                        entry_point=gt["entry_point"]), env)
            except Exception as exc:
                atomic_json(args.output / "errors" / f"{row['index']:04d}.json",
                    dict(index=row["index"], task=gt["task_id"], error=f"{type(exc).__name__}: {exc}",
                         harness=harness_identity()))
                raise
            if previous is not None:
                result["previous_validation"] = previous
            atomic_json(path, result)
        print(json.dumps(dict(index=row["index"], task=gt["task_id"], **result)), flush=True)
        return result["pass_rate"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, rows))
    if not all(results):
        raise RuntimeError("Some canonical solutions failed; inspect before evaluation")

if __name__ == "__main__":
    main()
