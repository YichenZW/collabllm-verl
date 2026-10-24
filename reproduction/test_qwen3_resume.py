import json

import pytest

from reproduction.qwen3_resume import resume_plan


def test_resume_requires_optimizer_state_and_preserves_training_arguments(tmp_path):
    arguments = dict(output_dir=str(tmp_path), max_steps=-1, num_train_epochs=8,
                     per_device_train_batch_size=1, gradient_accumulation_steps=64,
                     learning_rate=5e-6, seed=42, resume_ckpt_dir=None)
    (tmp_path / "args.json").write_text(json.dumps(arguments))
    with pytest.raises(RuntimeError, match="No complete"):
        resume_plan(tmp_path)
    checkpoint = tmp_path / "checkpoint-66"
    checkpoint.mkdir()
    state = dict(global_step=66, max_steps=264, epoch=2)
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    for name in ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        (checkpoint / name).touch()
    # An incomplete later save must never replace the valid earlier checkpoint.
    (tmp_path / "checkpoint-99").mkdir()
    (tmp_path / "checkpoint-99/trainer_state.json").write_text(json.dumps(dict(state, global_step=99)))
    restored, selected, _ = resume_plan(tmp_path)
    assert selected == checkpoint
    assert restored == dict(arguments, resume_ckpt_dir=str(checkpoint))
    final = dict(global_step=264, max_steps=264, epoch=8)
    (checkpoint / "trainer_state.json").write_text(json.dumps(final))
    with pytest.raises(RuntimeError, match="final artifacts are incomplete"):
        resume_plan(tmp_path)
    (tmp_path / "trainer_state.json").write_text(json.dumps(final))
    for name in ["train_results.json", "eval_results.json", "adapter_model.safetensors", "tokenizer_config.json"]:
        (tmp_path / name).touch()
    assert resume_plan(tmp_path) is None
