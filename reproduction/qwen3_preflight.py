"""Check Qwen chat formatting and supervised labels before GPU/API work."""
import json

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from reproduction.common import RUNS, atomic_json


def main():
    model = json.loads((RUNS / "assets.json").read_text())["base"]
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n", tokenizer=tok)
    conversation = [{"role": "system", "content": "System marker."},
        {"role": "user", "content": "First user marker."},
        {"role": "assistant", "content": "First assistant answer."},
        {"role": "user", "content": "Second user marker."},
        {"role": "assistant", "content": "Second assistant answer."}]
    tokens = tok.apply_chat_template(conversation, tokenize=True)
    batch = collator([{"input_ids": tokens, "attention_mask": [1] * len(tokens)}])
    supervised = tok.decode(batch["labels"][0][batch["labels"][0] != -100])
    assert "user marker" not in supervised and "System marker" not in supervised
    assert "First assistant answer." in supervised and "Second assistant answer." in supervised
    assert tok.eos_token_id in batch["labels"][0]
    assert tok.pad_token_id != tok.eos_token_id
    counts = {}
    for task in ["math-hard", "bigcodebench", "medium"]:
        ds = load_from_disk(str(RUNS / "data" / f"{task}-offline_dpo-paper"))
        for row in list(ds["train"].select(range(min(32, len(ds["train"]))))):
            prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            assert prompt.endswith("<think>\n\n</think>\n\n")
            for key in ["chosen", "rejected"]:
                full = tok.apply_chat_template(row["prompt"] + [{"role": "assistant", "content": row[key].strip()}], tokenize=False)
                assert full.startswith(prompt)
                completion = full[len(prompt):].rstrip("\n").removesuffix(tok.eos_token)
                assert completion == row[key].strip().lstrip("\n")
                assert tok.eos_token_id not in tok(completion, add_special_tokens=False)["input_ids"]
        counts[task] = {split: len(rows) for split, rows in ds.items()}
    atomic_json(RUNS / "preflight.json", dict(model=model, assistant_mask=True,
        eos_preserved=True, nonthinking_prefix=True, dpo_prefix_checks=192, datasets=counts))
    print("QWEN3_PREFLIGHT_PASS", counts)


if __name__ == "__main__":
    main()
