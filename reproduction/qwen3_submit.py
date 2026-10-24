"""Submit one explicitly requested Qwen GPU stage from immutable launch scripts."""
import argparse
import json
import re
import subprocess
import uuid

from reproduction.common import ROOT, RUNS, atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--after", action="append", type=int, default=[])
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--partition", choices=["general,clab", "dev"], default="general,clab")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("script", choices=["qwen3_eval.sh", "qwen3_train.sh", "qwen3_task_train.sh", "qwen3_smoke.sh", "qwen3_inference_smoke.sh"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.constraint is None:
        args.constraint = "a100|h100" if args.script in ["qwen3_eval.sh", "qwen3_inference_smoke.sh"] else "a40|a100|h100"
    if not re.fullmatch(r"qwen4b-[a-z0-9-]+", args.name):
        raise ValueError("Use an experiment-specific job name")
    launch = RUNS / "runtime/launches" / uuid.uuid4().hex
    launch.mkdir(parents=True)
    hashes = {}
    for source in (ROOT / "reproduction").glob("*.sh"):
        content = re.sub(r"bash reproduction/([a-z0-9_]+\.sh)",
            lambda match: f"bash {launch / match[1]}", source.read_text())
        target = launch / source.name
        target.write_text(content)
        target.chmod(0o444)
        hashes[source.name] = digest(content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    command = ["sbatch", "--parsable", "--job-name", args.name, "--partition", args.partition,
        "--gres", "gpu:1", "--nodes", "1", "--cpus-per-task", str(args.cpus), "--mem", args.mem,
        "--time", args.time, "--constraint", args.constraint, "--chdir", str(ROOT),
        "--output", str(RUNS / "logs" / f"{args.name}-%j.log")]
    if args.after:
        command += ["--dependency", "afterok:" + ":".join(map(str, args.after))]
    command += [str(launch / args.script)] + args.arguments
    manifest = dict(name=args.name, command=command, dependency=args.after, held=False,
                    snapshot=str(launch), code_commit=commit, script_hashes=hashes)
    atomic_json(launch / "submission.json", manifest)
    # Save the exact response locally before updating the combined inventory.
    response = subprocess.check_output(command, text=True).strip()
    (launch / "sbatch-response.txt").write_text(response + "\n")
    manifest["job"] = int(response.split(";")[0])
    atomic_json(launch / "submission.json", manifest)
    inventory = RUNS / "slurm_jobs.json"
    records = json.loads(inventory.read_text()) if inventory.exists() else []
    records.append(manifest)
    atomic_json(inventory, records)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
