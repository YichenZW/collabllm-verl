"""One real PPO rollout batch in the isolated legacy environment; diagnostic only."""
import json
import os
import sys
import threading

import numpy as np
from dotenv import load_dotenv
from safetensors import safe_open

from reproduction import budget
from reproduction.common import ROOT, RUNS, LEGACY, atomic_json


def main():
    load_dotenv(ROOT / ".env")
    job = os.environ.get("SLURM_JOB_ID", "manual")
    output = RUNS / "checkpoints/math-hard" / f"ppo-smoke-{job}"
    output.mkdir(parents=True, exist_ok=True)
    metered = budget.metered_call
    calls_lock = threading.Lock()

    def record_call(*args, **kwargs):
        response = metered(*args, **kwargs)
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
        with calls_lock:
            with (output / "api_usage.jsonl").open("a") as f:
                f.write(json.dumps(dict(model=kwargs["model"], usage=usage)) + "\n")
        return response

    budget.metered_call = record_call
    budget.install_litellm_guard()
    source = RUNS / "data/math-hard-online-smoke"
    if not source.exists():
        raise RuntimeError("Prepare the four-conversation online diagnostic dataset first")
    initial = RUNS / "checkpoints/math-hard/sft"
    base = json.loads((RUNS / "assets.json").read_text())["base"]
    sys.argv = ["ppo", "--dataset_repo", str(source), "--dataset_name", "math-hard",
        "--model_name", str(initial), "--base_model_name", str(base),
        "--output_dir", str(output), "--use_lora", "--disable_vllm", "--max_steps", "1",
        "--batch_size", "2", "--mini_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--num_train_epochs", "1", "--num_ppo_epochs", "1", "--learning_rate", "2e-6",
        "--max_model_len", "2048", "--num_samples", "3", "--max_new_turns", "4",
        "--max_metric_workers", "2", "--metric_names", "accuracy", "token_amount",
        "--metric_weights", "1", "-0.5",
        "--user_generation_kwargs", json.dumps(dict(model="gpt-4o-mini", num_retries=2, timeout=60)),
        "--reward_generation_kwargs", json.dumps(dict(model="gpt-4o-mini", num_retries=2, retry_after=1, timeout=60)),
        "--assistant_generation_kwargs", json.dumps(dict(model="local-ppo-smoke", temperature=0.6, max_new_tokens=256)),
        "--system_prompt_file", str(LEGACY / "collabllm/prompts/llm_assistant/system_prompt.txt")]
    from scripts.train import ppo
    original_trainer = ppo.PPOTrainer
    captured = {}

    def capture_trainer(*args, **kwargs):
        captured["value_head"] = {k: v.detach().cpu().clone()
            for k, v in kwargs["model"].v_head.state_dict().items()}
        trainer = original_trainer(*args, **kwargs)
        captured["trainer"] = trainer
        return trainer

    ppo.PPOTrainer = capture_trainer
    ppo.main()
    result = json.loads((output / "train_results.json").read_text())
    steps = [json.loads(line) for line in (output / "ppo_steps-rank0.jsonl").read_text().splitlines()]
    assert result["global_step"] == 1 and len(steps) == 1
    assert np.isfinite(result["stats"]["ppo/loss/total"])
    assert np.isfinite(steps[0]["rewards"]).all()
    trainer = captured["trainer"]
    model = trainer.accelerator.unwrap_model(trainer.model)
    changed_head = sum(not captured["value_head"][k].equal(v.detach().cpu())
        for k, v in model.v_head.state_dict().items())
    optimizer_steps = max(int(state["step"].item()) for state in trainer.optimizer.state.values() if "step" in state)
    with safe_open(initial / "adapter_model.safetensors", framework="pt") as before, \
         safe_open(output / "adapter_model.safetensors", framework="pt") as after:
        changed_adapter = sum(not before.get_tensor(k).equal(after.get_tensor(k)) for k in before.keys())
    assert changed_head > 0 and changed_adapter > 0 and optimizer_steps > 0
    summary = dict(job=job, checkpoint=str(output), rollout_batches=1, optimizer_steps=optimizer_steps,
        changed_adapter_tensors=changed_adapter, changed_value_head_tensors=changed_head,
        stats=result["stats"], rewards=steps[0]["rewards"], trl_version="0.10.1",
        num_samples=3, max_new_turns=4, max_new_tokens=256, judge="gpt-4o-mini",
        initialization="our SFT", reference="base model with LoRA disabled",
        scope="PPO optimizer diagnostic only; not paper-scale PPO or a Table1 result")
    atomic_json(RUNS / "ppo_smoke.json", summary)
    print("PPO_SMOKE_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
