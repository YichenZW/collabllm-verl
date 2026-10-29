"""Real Llama-3.1-8B GPU inference, simulation, and MR smoke test."""
import json
import os
import faulthandler
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from collabllm.simulation import ChatSessionSimulator
from collabllm.reward import multiturn_aware_reward
from examples.metrics import *
from reproduction.common import ROOT, RUNS, BASE, atomic_json
from reproduction.budget import install_litellm_guard

def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    load_dotenv(ROOT / ".env")
    install_litellm_guard()
    torch.manual_seed(42)
    assert torch.cuda.is_available(), "A Slurm GPU allocation is required"
    assets = json.loads((RUNS / "assets.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(assets["base"], torch_dtype=torch.bfloat16,
                device_map="cuda", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(assets["base"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    inputs = tok.apply_chat_template([dict(role="user", content="What is 17 plus 25? Answer briefly.")],
                    add_generation_prompt=True, return_tensors="pt").cuda()
    with torch.inference_mode():
        output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
    base_response = tok.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
    assert base_response.strip()
    print("BASE_RESPONSE", base_response, flush=True)
    model = PeftModel.from_pretrained(model, assets["math-hard"]["adapter"]).eval()
    common = dict(task_desc="question answering", single_turn_prompt="What is 17 plus 25?",
        local_model=model, local_tokenizer=tok,
        assistant_generation_kwargs=dict(model="collabllm-math", temperature=0.6, max_new_tokens=256),
        user_generation_kwargs=dict(model="gpt-4o-mini", num_retries=2, timeout=60))
    # Seed the first user turn: for trivial arithmetic the simulator can solve
    # the task itself and legitimately terminate before any assistant response.
    chat = ChatSessionSimulator().run_chat_simulation(
                    chat_history=[dict(role="user", content="What is 17 plus 25?")],
                    num_samples=1, max_new_turns=4, verbose=False, **common)[0]
    assert any(m["role"] == "assistant" and m["content"].strip() for m in chat)
    print("CHAT_SIMULATION_PASS", len(chat), flush=True)
    rewards = multiturn_aware_reward(chat_history=chat, num_samples=1, max_new_turns=0,
        single_turn_completion="42", metric_names=["accuracy", "interactivity", "token_amount"],
        metric_weights=[1, 1, -0.5], reward_generation_kwargs=dict(model="gpt-4o-mini", num_retries=2,
        retry_after=1, timeout=60), **common)
    assert all(torch.isfinite(torch.tensor(v)).all() for v in rewards.values())
    print("METRICS_PASS", json.dumps(rewards), flush=True)
    future_rewards, futures = multiturn_aware_reward(chat_history=chat[:2], num_samples=3,
        max_new_turns=4, single_turn_completion="42", metric_names=["accuracy", "token_amount"],
        metric_weights=[1, -0.5], reward_generation_kwargs=dict(model="gpt-4o-mini", num_retries=2,
        retry_after=1, timeout=60), return_details=True, **common)
    assert len(futures) == 3 and len(future_rewards["MR"]) == 3
    atomic_json(RUNS / "gpu_smoke.json", dict(gpu=torch.cuda.get_device_name(), job=os.getenv("SLURM_JOB_ID"),
        torch=torch.__version__, base_response=base_response, chat=chat, rewards=rewards,
        future_chats=futures, future_rewards=future_rewards,
        max_allocated_gb=torch.cuda.max_memory_allocated()/1e9))
    print("GPU_SMOKE_PASS", json.dumps(rewards), flush=True)
    faulthandler.cancel_dump_traceback_later()

if __name__ == "__main__":
    main()
