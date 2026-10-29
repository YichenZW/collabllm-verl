"""Two GPU optimizer steps resumed from the real two-step DPO smoke checkpoint."""
import json
import os
import subprocess

from reproduction.common import ROOT, RUNS, atomic_json, source_identity
from reproduction.qwen3_resume import training_command


def main():
    source = RUNS / "checkpoints/math-hard/offline_dpo-smoke"
    checkpoint = source / "checkpoint-2"
    output = RUNS / "checkpoints/math-hard/offline_dpo-resume-smoke"
    if output.exists():
        raise RuntimeError("Resume smoke output already exists; inspect before retrying")
    original = json.loads((source / "args.json").read_text())
    # The original short smoke schedule ended at zero LR; allow its scheduler
    # to advance once before checking that resumed weights actually change.
    args = dict(original, output_dir=str(output), resume_ckpt_dir=str(checkpoint), max_steps=4)
    command = training_command(args)
    record = dict(job=os.getenv("SLURM_JOB_ID"), source=str(checkpoint), output=str(output),
                  command=command, code=source_identity("reproduction/qwen3_resume_smoke.py",
                                                      "reproduction/qwen3_resume.py"))
    atomic_json(RUNS / "resume_smoke_started.json", record)
    subprocess.run(command, cwd=ROOT, check=True)
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 4 and state["max_steps"] == 4
    import torch
    from safetensors.torch import load_file
    before = load_file(str(checkpoint / "adapter_model.safetensors"))
    after = load_file(str(output / "adapter_model.safetensors"))
    changed = sum(not torch.equal(before[key], after[key]) for key in before)
    assert changed > 0
    optimizer = torch.load(output / "checkpoint-4/optimizer.pt", map_location="cpu", weights_only=True)
    steps = {int(value["step"]) for value in optimizer["state"].values()}
    assert steps == {4}, steps
    atomic_json(RUNS / "resume_smoke.json", dict(record, global_step=4, changed_adapter_tensors=changed,
                optimizer_steps=sorted(steps), status="passed", original_training_arguments=original))
    print("RESUME_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
