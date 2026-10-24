"""Refresh a read-only experiment status without dispatching API requests."""
import json
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

from reproduction.common import RUNS, atomic_json
from reproduction.budget import status


def main():
    jobs = json.loads((RUNS / "slurm_jobs.json").read_text())
    result = subprocess.run(["sacct", "-n", "-P", "-j", ",".join(str(j["job"]) for j in jobs),
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES"],
        capture_output=True, text=True, check=True)
    accounting = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 6 and fields[0].isdigit():
            accounting[int(fields[0])] = dict(zip(
                ["state", "exit_code", "elapsed", "nodes", "resources"], fields[1:6]))
    for job in jobs:
        job["slurm"] = accounting.get(job["job"], {})
    evaluations = []
    for path in RUNS.glob("eval-*/*/*/summary.json"):
        evaluations.append(dict(path=str(path), **json.loads(path.read_text())))
    training = []
    for path in RUNS.glob("checkpoints/*/*/train_results.json"):
        training.append(dict(path=str(path), **json.loads(path.read_text())))
    record = dict(updated=datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        jobs=jobs, budget=status(), evaluations=evaluations, training=training)
    for field, name in [("cpu_preflight", "preflight.json"), ("api_probe", "api_probe.json"),
                        ("gpu_smoke", "training_smoke.json"), ("resume_smoke", "resume_smoke.json"),
                        ("batch_worker", "batch_worker.json")]:
        path = RUNS / name
        record[field] = json.loads(path.read_text()) if path.exists() else None
    atomic_json(RUNS / "status.json", record)
    print(json.dumps(dict(jobs=[dict(job=j["job"], name=j["name"], **j["slurm"]) for j in jobs],
        budget=record["budget"], training_runs=len(training), evaluations=evaluations), indent=2))


if __name__ == "__main__":
    main()
