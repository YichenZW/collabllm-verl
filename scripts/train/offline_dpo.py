#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts.engine.build_dataset`

DPO train a causal-LM + LoRA adapter on a multi-turn dataset.
Example (on 4 NVIDIA A100-SXM4-80GB GPUs):
-------
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=4 -m scripts.train.offline_dpo \
    --dataset_repo collabllm/collabllm-multiturn-medium \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/offline_dpo_from_base/collabllm-multiturn-medium \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_lora

CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=4 -m scripts.train.offline_dpo \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --output_dir outputs/offline_dpo/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_lora
"""

from __future__ import annotations

import argparse, os, json
from typing import Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
import torch.distributed as dist
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from collabllm.datasets.multiturn import MultiturnDataset
from trl import DPOConfig, DPOTrainer
import wandb


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--eval_ratio",   type=float, default=0.1)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0)        
    p.add_argument("--logging_steps", type=int, default=1)           
    p.add_argument("--max_prompt_length", type=int, default=4096) 
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--minimum_gap", type=float, default=0.02) 

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    p.add_argument("--system_prompt_file", type=str, default=None)
    p.add_argument("--prepared_dataset", action="store_true")
    p.add_argument("--truncation_mode", choices=["keep_start", "keep_end"], default="keep_start")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_deepspeed", action="store_true")

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation="sdpa",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation="sdpa",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    
    print(model.device)
    return model, tok

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method=None, device_id=torch.device(f"cuda:{local_rank}"))
    dist.barrier(device_ids=[local_rank])
    set_seed(args.seed)

    # Dataset
    if args.prepared_dataset:
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_repo)
    else:
        wrapped_ds = MultiturnDataset(args.dataset_repo)
        if args.system_prompt_file:
            with open(args.system_prompt_file) as f:
                wrapped_ds.sys_msg = [{"role": "system", "content": f.read()}]
        ds = wrapped_ds.to_dpo_dataset(eval_ratio=args.eval_ratio, minimum_gap=args.minimum_gap)
    if local_rank == 0:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        lora_dropout=args.peft_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None
    
    # Load model
    model, tok = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=f"cuda:{local_rank}" if args.device == "cuda" else args.device,
        is_eval=False,
    )

    # DeepSpeed zero
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }

    # Trainer config
    train_args = DPOConfig(
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=len(ds["eval"]) > 0,
        eval_steps=args.eval_steps, 
        save_strategy='epoch',
        eval_strategy="steps" if len(ds["eval"]) else "no",
        gradient_checkpointing=True,  
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=args.max_new_tokens, 
        max_prompt_length=args.max_prompt_length, 
        truncation_mode=args.truncation_mode,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        output_dir=args.output_dir,
        deepspeed=ds_cfg if args.use_deepspeed else None,
        max_steps=args.max_steps,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        fp16=not torch.cuda.is_bf16_supported(), 
        bf16=torch.cuda.is_bf16_supported(),
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )
    
    def process(row):
        if args.prepared_dataset:
            # Match the historical Llama template without duplicating EOT:
            # TRL 0.18 tokenization appends eos_token_id itself.
            prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            completions = {}
            for key in ["chosen", "rejected"]:
                full = tok.apply_chat_template(row["prompt"] +
                    [{"role": "assistant", "content": row[key].strip()}], tokenize=False)
                if not full.startswith(prompt):
                    raise ValueError("Chat template changed the prompt prefix")
                completions[key] = full[len(prompt):].removesuffix(tok.eos_token)
            return {**row, "prompt": prompt, **completions}
        # Keep conversational data so TRL uses the model's actual end-of-turn token.
        return {**row, "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}]}

    ds = ds.map(process, load_from_cache_file=False)

    trainer = DPOTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if len(ds["eval"]) else None,
        processing_class=tok,

        args=train_args,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if len(ds["eval"]):
        trainer.save_metrics("eval", trainer.evaluate())

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"offline_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
