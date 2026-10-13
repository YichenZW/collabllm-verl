"""Verify canonical solutions before spending API budget on coding samples."""
import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path
from reproduction.common import ROOT, RUNS, atomic_json

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
        if path.exists():
            result = json.loads(path.read_text())
        else:
            proc = subprocess.run([str(ROOT / ".venv-code/bin/python"),
                str(ROOT / "reproduction/check_code.py")],
                input=json.dumps(dict(answer=gt["answer"], test=gt["test"], entry_point=gt["entry_point"])),
                capture_output=True, text=True, timeout=300, env=env, check=True)
            result = json.loads(proc.stdout)
            atomic_json(path, result)
        print(json.dumps(dict(index=row["index"], task=gt["task_id"], **result)), flush=True)
        return result["pass_rate"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, rows))
    if not all(results):
        raise RuntimeError("Some canonical solutions failed; inspect before evaluation")

if __name__ == "__main__":
    main()
