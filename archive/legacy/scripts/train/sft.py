#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts.engine.build_dataset`

Fine-tune or SFT a causal-LM + LoRA adapter on a multi-turn dataset.
Examples:
w/ Lora (on 2 NVIDIA A100-SXM4-80GB GPUs):
-------
CUDA_VISIBLE_DEVICES=0,1 WANDB__SERVICE_WAIT=300 torchrun --master_port=56400 --nnodes=1 --nproc_per_node=2 -m scripts.train.sft \
    --dataset_repo collabllm/collabllm-multiturn-medium \
    --output_dir outputs/sft/collabllm-multiturn-medium \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_lora

w/ Lora & Quantization (4-bit) (on 4 NVIDIA A100-SXM4-80GB GPUs):
-------
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB__SERVICE_WAIT=300 torchrun --master_port=56800 --nnodes=1 --nproc_per_node=4 -m scripts.train.sft \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --output_dir outputs/debug/collabllm-multiturn-math-hard \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --lower_bound_metric "rewards.accuracy" \
    --lower_bound 0.5 \
    --use_lora  --use_4bit
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
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn SFT trainer")

    # Data / paths
    p.add_argument("--dataset_repo",    type=str, required=True)
    p.add_argument("--output_dir",      type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)
    p.add_argument("--eval_ratio",         type=float, default=0.1)
    p.add_argument("--lower_bound_metric", type=str, default=None)
    p.add_argument("--lower_bound",        type=float, default=0.0)

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

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    p.add_argument("--assistant_only_loss", action="store_true")
    p.add_argument("--system_prompt_file", type=str, default=None)
    p.add_argument("--prepared_dataset", action="store_true")
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

    tok.padding_side = "left" if is_eval else "right"
    tok.pad_token = tok.pad_token or tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
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
        ds = wrapped_ds.to_sft_dataset(eval_ratio=args.eval_ratio,
                                      lower_bound_metric=args.lower_bound_metric,
                                      lower_bound=args.lower_bound)
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
            "overlap_comm": False,
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

    # Trainer config (uses newly exposed warmup_ratio & logging_steps)
    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=len(ds["eval"]) > 0,
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        eval_strategy="steps" if len(ds["eval"]) else "no",
        group_by_length=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        max_length=args.max_seq_length,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
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
            config=train_cfg.to_dict(),
            save_code=True,
            job_type="train",
        )
    
    collator = None
    if args.assistant_only_loss:
        is_qwen = model.config.model_type == "qwen3"
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|im_start|>user\n" if is_qwen else "<|start_header_id|>user<|end_header_id|>",
            response_template="<|im_start|>assistant\n" if is_qwen else "<|start_header_id|>assistant<|end_header_id|>",
            tokenizer=tok,
        )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if len(ds["eval"]) else None,
        processing_class=tok,
        data_collator=collator,

        args=train_cfg,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if len(ds["eval"]):
        trainer.save_metrics("eval", trainer.evaluate())

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"sft-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":

    main()
