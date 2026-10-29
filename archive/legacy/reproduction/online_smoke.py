"""Two real online-DPO optimizer steps; diagnostic only, not a Table 1 row."""
import json
import os
import sys

import numpy as np
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from safetensors import safe_open

from reproduction.budget import install_litellm_guard
from reproduction.common import ROOT, RUNS, LEGACY, atomic_json
from reproduction.prepare import pinned_dataset


def main():
    load_dotenv(ROOT / ".env")
    install_litellm_guard()
    source = load_from_disk(str(LEGACY / "data/interactive-math-hard"))["train"]
    single = pinned_dataset("lighteval/MATH-Hard")["train"]
    by_id = {}
    for row in source:
        if row["idx"] not in by_id or len(row["prompt"]) < len(by_id[row["idx"]]["prompt"]):
            by_id[row["idx"]] = row
    rows = []
    for idx in sorted(by_id)[:4]:
        row, qa = by_id[idx], single[idx]
        rows.append(dict(prompt=row["prompt"], completion=row["chosen"], conv_id=idx,
            score=row["chosen_eval"]["reward"], single_turn_prompt=qa["problem"],
            single_turn_completion=qa["solution"],
            single_turn_metadata={"level": qa["level"], "type": qa["type"]}))
    data = RUNS / "data/math-hard-online-smoke"
    if not data.exists():
        Dataset.from_list(rows).save_to_disk(str(data))
    model = RUNS / "checkpoints/math-hard/sft"
    output = RUNS / "checkpoints/math-hard/online_dpo-smoke"
    sys.argv = ["online_dpo", "--dataset_repo", str(data), "--dataset_name", "math-hard",
        "--model_name", str(model), "--output_dir", str(output), "--use_lora", "--disable_vllm",
        "--max_steps", "2", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--learning_rate", "5e-6", "--max_model_len", "2048", "--max_new_tokens", "256",
        "--num_samples", "3", "--max_new_turns", "4", "--max_metric_workers", "2",
        "--metric_names", "accuracy", "token_amount", "--metric_weights", "1", "-0.5",
        "--user_generation_kwargs", json.dumps(dict(model="gpt-4o-mini", num_retries=2, timeout=60)),
        "--reward_generation_kwargs", json.dumps(dict(model="gpt-4o-mini", num_retries=2, retry_after=1, timeout=60)),
        "--assistant_generation_kwargs", json.dumps(dict(model="local-online-smoke", temperature=0.6, max_new_tokens=256)),
        "--system_prompt_file", str(LEGACY / "collabllm/prompts/llm_assistant/system_prompt.txt")]
    from scripts.train.online_dpo import main as train
    train()
    metrics = json.loads((output / "train_results.json").read_text())
    state = json.loads((output / "trainer_state.json").read_text())
    rewards = [json.loads(line) for line in (output / "online_rewards-rank0.jsonl").read_text().splitlines()]
    assert state["global_step"] == 2
    assert np.isfinite(metrics["train_loss"])
    assert len(rewards) == 2 and all(np.isfinite(r["rewards"]).all() for r in rewards)
    with safe_open(model / "adapter_model.safetensors", framework="pt") as before, \
         safe_open(output / "adapter_model.safetensors", framework="pt") as after:
        changed = sum(not before.get_tensor(k).equal(after.get_tensor(k)) for k in before.keys())
    assert changed > 0, "Online optimizer did not change any adapter weights"
    atomic_json(RUNS / "online_smoke.json", dict(job=os.getenv("SLURM_JOB_ID"),
        global_step=state["global_step"], train_metrics=metrics, changed_adapter_tensors=changed,
        source_conversation_ids=[row["conv_id"] for row in rows], num_samples=3, max_new_turns=4,
        max_new_tokens=256, judge="gpt-4o-mini", initialization="our SFT",
        reference="TRL0.18.1 frozen copy of initial policy",
        scope="Online optimization diagnostic only; not paper-scale Online DPO or Table1 result"))
    print("ONLINE_DPO_SMOKE_PASS", metrics, "changed_tensors", changed, flush=True)


if __name__ == "__main__":
    main()
