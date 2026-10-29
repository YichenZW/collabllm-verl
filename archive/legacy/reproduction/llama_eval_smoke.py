"""Local GPU inference only: check saved Llama adapters without a paid API."""
import json
import os
from openai import OpenAI

from reproduction.common import RUNS, atomic_json, source_identity


def main():
    port = int(os.environ["LLAMA_EVAL_PORT"])
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local", timeout=240, max_retries=0)
    expected = ["base"] + [f"{task}-{variant}" for task in ["bigcodebench", "medium"]
        for variant in ["sft", "offline_dpo", "released"]]
    available = {m.id for m in client.models.list().data}
    assert set(expected).issubset(available), (expected, available)
    results = []
    for model in expected:
        response = client.chat.completions.create(model=model,
            messages=[dict(role="user", content="Write a Python function add(a, b) that returns their sum.")],
            max_tokens=64, temperature=0, seed=42)
        text = response.choices[0].message.content
        assert isinstance(text, str) and text.strip()
        results.append(dict(model=model, answer=text, usage=response.usage.model_dump(),
                            finish_reason=response.choices[0].finish_reason))
    record = dict(status="passed", job=os.getenv("SLURM_JOB_ID"), paid_api_calls=0,
        results=results, code=source_identity("reproduction/llama_eval_smoke.py", "reproduction/serve.py"))
    atomic_json(RUNS / "inference_smoke.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
