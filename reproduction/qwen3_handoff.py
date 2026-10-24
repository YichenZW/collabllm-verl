"""Transfer an owned training job only after its replacement GPU is ready."""
import json
import os
import subprocess
import time

from reproduction.common import RUNS, atomic_json, source_identity

TERMINAL = {"COMPLETED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"}


def job_state(job):
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", str(job),
        "--format=JobIDRaw,State,UID"], text=True)
    row = next(line.split("|") for line in output.splitlines()
               if line.split("|")[0] == str(job))
    if int(row[2]) != os.getuid():
        raise RuntimeError("Refusing to control a job owned by another user")
    return row[1].split()[0]


def probe_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Replacement GPU unavailable; source training is left running")
    torch.zeros(1, device="cuda").add_(1).cpu()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    return dict(name=props.name, total_memory=props.total_memory)


def validate_checkpoint(checkpoint, state):
    """Read the real saved states before touching the live source allocation."""
    import torch
    from safetensors import safe_open

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    steps = {int(value["step"]) for value in optimizer["state"].values()}
    if steps != {state["global_step"]}:
        raise RuntimeError("Optimizer step does not match the checkpoint")
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    if scheduler["last_epoch"] != state["global_step"]:
        raise RuntimeError("Scheduler step does not match the checkpoint")
    # This is our own trusted Trainer checkpoint, including NumPy RNG state.
    rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
    if not {"python", "numpy", "cpu", "cuda"}.issubset(rng):
        raise RuntimeError("Incomplete random-number state")
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
        count = len(list(weights.keys()))
        if not count:
            raise RuntimeError("Empty adapter checkpoint")
        for key in weights.keys():
            if not torch.isfinite(weights.get_tensor(key)).all():
                raise RuntimeError(f"Non-finite adapter tensor: {key}")
    return dict(global_step=state["global_step"], max_steps=state["max_steps"],
                optimizer_steps=sorted(steps), adapter_tensors=count)


def handoff_source(task, source_job, directory, plan_function):
    target_job = int(os.environ["SLURM_JOB_ID"])
    request_path = RUNS / "handoffs" / f"{task}-{target_job}.request.json"
    request = json.loads(request_path.read_text())
    expected = dict(task=task, source_job=source_job, target_job=target_job, status="requested")
    if any(request.get(key) != value for key, value in expected.items()) or source_job == target_job:
        raise RuntimeError("Handoff request does not match these exact jobs")
    target_state = job_state(target_job)
    if target_state != "RUNNING":
        raise RuntimeError("Replacement allocation is not running")
    source_state = job_state(source_job)
    if source_state in TERMINAL:
        return source_state
    if source_state not in {"RUNNING", "COMPLETING"}:
        raise RuntimeError(f"Unexpected source state: {source_state}")
    try:
        plan = plan_function(directory)
    except RuntimeError as exc:
        if "Training checkpoint is complete but final artifacts are incomplete" not in str(exc):
            raise
        plan = None  # The source is finalizing; wait for it instead of cancelling.
    gpu = probe_gpu()
    record = dict(expected, started=time.time(), gpu=gpu,
                  code=source_identity("reproduction/qwen3_handoff.py", "reproduction/qwen3_resume.py"))
    path = RUNS / "handoffs" / f"{task}-{target_job}.json"
    if plan is not None:
        arguments, checkpoint, state = plan
        if arguments["max_steps"] != -1 or arguments["num_train_epochs"] != 8:
            raise RuntimeError("Handoff must retain the full eight-epoch schedule")
        record.update(checkpoint=str(checkpoint), checkpoint_validation=validate_checkpoint(checkpoint, state))
        atomic_json(path, dict(record, status="replacement_and_checkpoint_verified"))
        # Recheck ownership/state immediately before this exact-job cancellation.
        if job_state(source_job) == "RUNNING":
            cancelled = subprocess.run(["scancel", str(source_job)], capture_output=True, text=True)
            if cancelled.returncode and job_state(source_job) not in TERMINAL:
                raise subprocess.CalledProcessError(cancelled.returncode, cancelled.args,
                                                    cancelled.stdout, cancelled.stderr)
            record["source_cancel_requested"] = time.time()
            atomic_json(path, dict(record, status="waiting_for_source_exit"))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        source_state = job_state(source_job)
        if source_state in TERMINAL:
            atomic_json(path, dict(record, status="source_terminal", source_state=source_state,
                                   finished=time.time()))
            return source_state
        time.sleep(2)
    raise TimeoutError("Source is not terminal; refusing simultaneous training writers")
