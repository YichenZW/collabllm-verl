"""Resume an interrupted full DPO run without changing its training arguments."""
import argparse
import json
import os
import subprocess
import sys
import time

from reproduction.common import ROOT, RUNS, atomic_json, source_identity


def training_command(arguments):
    port = str(20000 + int(os.environ["SLURM_JOB_ID"]) % 30000)
    command = [sys.executable, "-m", "torch.distributed.run", "--master_addr=127.0.0.1",
               "--master_port", port, "--nproc_per_node=1", "-m", "scripts.train.offline_dpo"]
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        command.append("--" + key)
        if value is not True:
            command.append(str(value))
    return command


def resume_plan(directory):
    arguments = json.loads((directory / "args.json").read_text())
    final_state = directory / "trainer_state.json"
    required = ["train_results.json", "eval_results.json", "adapter_model.safetensors", "tokenizer_config.json"]
    if final_state.exists() and all((directory / name).exists() for name in required):
        state = json.loads(final_state.read_text())
        if state["global_step"] >= state["max_steps"] and state["epoch"] >= arguments["num_train_epochs"]:
            return None
    candidates = []
    required = ["trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    for checkpoint in directory.glob("checkpoint-*"):
        if all((checkpoint / name).exists() for name in required):
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            candidates.append((state["global_step"], checkpoint, state))
    if not candidates:
        raise RuntimeError("No complete optimizer/RNG checkpoint is available; refusing a fresh training restart")
    _, checkpoint, state = max(candidates, key=lambda item: item[0])
    if state["global_step"] >= state["max_steps"]:
        raise RuntimeError("Training checkpoint is complete but final artifacts are incomplete; inspect finalization")
    arguments["resume_ckpt_dir"] = str(checkpoint)
    if arguments["output_dir"] != str(directory) or arguments["max_steps"] != -1:
        raise ValueError("Conditional resume requires the original full-run output and epoch schedule")
    return arguments, checkpoint, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["math-hard", "bigcodebench", "medium"])
    parser.add_argument("source_job", type=int)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Training resume must run inside a Slurm allocation")
    inventory = json.loads((RUNS / "slurm_jobs.json").read_text())
    source = next(row for row in inventory if row["job"] == args.source_job)
    if args.task not in source["command"]:
        raise ValueError("Source job belongs to another task")
    rows = subprocess.check_output(["sacct", "-n", "-P", "-j", str(args.source_job),
                                   "--format=JobIDRaw,State"], text=True).splitlines()
    state = next(row.split("|")[1].split()[0] for row in rows
                 if row.split("|")[0] == str(args.source_job))
    if state not in ["COMPLETED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"]:
        raise RuntimeError(f"Source training job is not terminal: {state}")
    directory = RUNS / "checkpoints" / args.task / "offline_dpo"
    plan = resume_plan(directory)
    record = dict(source_job=args.source_job, source_state=state, job=os.getenv("SLURM_JOB_ID"),
                  task=args.task, started=time.time(), code=source_identity("reproduction/qwen3_resume.py"))
    path = RUNS / "resumes" / f"{args.task}-{record['job']}.json"
    if plan is None:
        atomic_json(path, dict(record, status="already_complete"))
        print("Full DPO training already completed; no further optimizer steps.", flush=True)
        return
    arguments, checkpoint, checkpoint_state = plan
    command = training_command(arguments)
    record.update(status="running", checkpoint=str(checkpoint), checkpoint_state=checkpoint_state,
                  arguments=arguments, command=command)
    atomic_json(path, record)
    print(f"Resume {args.task} at step {checkpoint_state['global_step']}/{checkpoint_state['max_steps']}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if resume_plan(directory) is not None:
        raise RuntimeError("Resumed training returned without all final artifacts")
    atomic_json(path, dict(record, status="completed", finished=time.time()))


if __name__ == "__main__":
    main()
