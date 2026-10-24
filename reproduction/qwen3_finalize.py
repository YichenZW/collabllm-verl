"""Compute local metrics once every sample has prepaid Claude results."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from reproduction.common import ROOT, RUNS


def readiness(directory):
    protocol = json.loads((directory / "manifest.json").read_text())
    records = json.loads((RUNS / "data" / f"{protocol['dataset']}-test.json").read_text())
    if protocol["limit"]:
        records = records[:protocol["limit"]]
    ready = completed = 0
    for row in records:
        path = directory / "samples" / f"{row['index']:04d}.json"
        if not path.exists():
            continue
        sample = json.loads(path.read_text())
        if sample.get("completed"):
            completed += 1
            ready += 1
        elif sample.get("generated") and "judge" in sample and (
                protocol["dataset"] == "math-hard" or "extraction" in sample):
            ready += 1
    return protocol, dict(expected=len(records), ready=ready, completed=completed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    for root in args.roots:
        for path in sorted(root.glob("*/*/manifest.json")):
            protocol, state = readiness(path.parent)
            print(json.dumps(dict(directory=str(path.parent), **state)), flush=True)
            if state["ready"] != state["expected"] or state["completed"] == state["expected"]:
                continue
            # The score phase refuses paid calls if any precomputed result is absent.
            command = [sys.executable, "-m", "reproduction.evaluate", "--phase", "score"]
            for key in ["dataset", "variant", "model", "judge", "limit", "workers", "output", "seed"]:
                command += ["--" + key.replace("_", "-"), str(protocol[key])]
            if protocol.get("user"):
                command += ["--user", protocol["user"]]
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
