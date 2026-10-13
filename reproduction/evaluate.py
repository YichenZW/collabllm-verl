"""Resumable paper-era evaluation against a local vLLM OpenAI endpoint.

Uses original prompts, splits, turn limits, character BLEU, cl100k token counts,
and historical ITR scaling. Every response and raw judge score is retained.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import shutil
import threading
import time
from pathlib import Path

import anthropic
import numpy as np
import tiktoken
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI

from collabllm.utils.extract_json_reliable import extract_json
from reproduction.common import ROOT, RUNS, TASKS, atomic_json, chat_text, digest, legacy_prompt
from reproduction.budget import metered_call, BudgetExceeded

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=TASKS, required=True)
    p.add_argument("--variant", choices=["base", "proactive", "released", "sft", "offline_dpo", "online_dpo", "ppo"], required=True)
    p.add_argument("--model", required=True, help="Model ID served by vLLM")
    p.add_argument("--local-url", default="http://127.0.0.1:8137/v1")
    p.add_argument("--judge", default="claude-sonnet-4-5-20250929")
    p.add_argument("--user", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-results", help="Reuse saved samples with identical protocol and inputs")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = TASKS[args.dataset]
    user_model = args.user or cfg["user"]
    api = OpenAI(base_url=os.getenv("OPENAI_API_BASE") or None, timeout=90, max_retries=0)
    judge_api = anthropic.Anthropic(timeout=90, max_retries=0) if args.judge.startswith("claude") else api
    local = OpenAI(base_url=args.local_url, api_key="local", timeout=240, max_retries=2)
    records = json.loads((RUNS / "data" / f"{args.dataset}-test.json").read_text())
    if args.limit:
        records = records[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(args), user_model=user_model, protocol="d573e49",
                    test_sha256=digest(records), judge_changed_from_paper=True,
                    itr_scaling="(raw_score - 2.5) * 2", bleu="character-level NLTK sentence_bleu")
    manifest.pop("local_url")  # Endpoint location does not change experimental identity.
    manifest.pop("seed_results")
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output uses different settings; choose a new output directory")
    atomic_json(manifest_path, manifest)
    if args.seed_results:
        source = Path(args.seed_results)
        prior = json.loads((source / "manifest.json").read_text())
        identity = ["dataset", "variant", "model", "judge", "user_model", "seed", "protocol",
                    "judge_changed_from_paper", "itr_scaling", "bleu"]
        if any(prior[k] != manifest[k] for k in identity):
            raise ValueError("Seed results used a different experimental protocol")
        for row in records:
            src = source / "samples" / f"{row['index']:04d}.json"
            dst = out / "samples" / src.name
            if src.exists() and not dst.exists():
                if json.loads(src.read_text())["input"] != row:
                    raise ValueError("Seed sample input differs")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    enc = tiktoken.get_encoding("cl100k_base")
    calls_lock = threading.Lock()
    # Generation/API workers can run concurrently, but benchmark timeouts
    # should not depend on twelve Python test processes sharing two Slurm CPUs.
    code_slot = threading.BoundedSemaphore(1)

    def request_json(prompt, model, kind):
        for attempt in range(3):
            if model.startswith("claude"):
                resp = metered_call(judge_api.messages.create, model=model, max_tokens=4096, temperature=0,
                        messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in resp.content if b.type == "text")
                usage = resp.usage.model_dump()
            else:
                resp = metered_call(api.chat.completions.create, model=model, max_tokens=4096,
                        temperature=1.0 if kind == "user" else 0.0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}])
                text = resp.choices[0].message.content
                usage = resp.usage.model_dump()
            with calls_lock:
                with (out / "api_usage.jsonl").open("a") as f:
                    f.write(json.dumps(dict(kind=kind, model=model,
                        resolved_model=getattr(resp, "model", model), usage=usage)) + "\n")
            try:
                value = extract_json(text)
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError, KeyError):
                pass
        raise RuntimeError(f"{kind}: invalid JSON after 3 attempts")

    def run(row):
        target = out / "samples" / f"{row['index']:04d}.json"
        state = json.loads(target.read_text()) if target.exists() else dict(index=row["index"],
                input=row, chat=[], completed=False)
        if state.get("completed"):
            return state
        chat = state["chat"]
        if not chat and args.variant not in ["base", "proactive"]:
            chat.append(dict(role="system", content=legacy_prompt("llm_assistant/system_prompt.txt")))
        while len([m for m in chat if m["role"] != "system"]) < 2 * cfg["turns"]:
            if chat and "[[TERMINATE CHAT]]" in chat[-1]["content"]:
                break
            if not chat or chat[-1]["role"] in ["system", "assistant"]:
                prompt = legacy_prompt(f"user_simulator_cot/{cfg['task']}.txt").format(
                        question=row["prompt"], answer=row["completion"], chat_history=chat_text(chat))
                payload = request_json(prompt, user_model, "user")
                response = payload["response"]
                role = "user"
            else:
                messages = chat
                if args.variant == "proactive":
                    prompt = legacy_prompt("llm_assistant/proact.txt").format(
                        chat_history=chat_text(chat), max_new_tokens=cfg["max_tokens"])
                    messages = [dict(role="user", content=prompt)]
                result = local.chat.completions.create(model=args.model, messages=messages,
                        temperature=cfg["temperature"], top_p=0.9, max_tokens=cfg["max_tokens"],
                        seed=args.seed + row["index"] * 100 + len(chat))
                response = result.choices[0].message.content
                role = "assistant"
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Empty model response")
            chat.append(dict(role=role, content=response.strip()))
            atomic_json(target, state)
        if "judge" not in state:
            prompt = legacy_prompt(f"llm_judge/{cfg['task']}.txt").format(
                    chat_history="", chat=chat_text(chat), question=row["prompt"], answer=row["completion"])
            state["judge"] = request_json(prompt, args.judge, "judge")
            atomic_json(target, state)
        raw_itr = float(state["judge"]["interactivity"]["score"])
        if not 1 <= raw_itr <= 3:
            raise ValueError(f"Invalid raw interactivity score: {raw_itr}")
        metrics = {"itr": (raw_itr - 2.5) * 2,
                   "itr_div3": raw_itr / 3,
                   "tokens_k": len(enc.encode("".join(m["content"] + " " for m in chat
                          if m["role"] == "assistant"), disallowed_special=())) / 1000}
        if args.dataset == "math-hard":
            metrics["accuracy"] = float(state["judge"]["accuracy"]["score"])
            if metrics["accuracy"] not in [0, 1]:
                raise ValueError("Invalid accuracy judge score")
        else:
            if "extraction" not in state:
                prompt = legacy_prompt("funtional/extract_answer.txt").format(chat=chat_text(chat))
                state["extraction"] = request_json(prompt, args.judge, "extract")
                atomic_json(target, state)
            answer = state["extraction"]["final_answer"]
            if args.dataset == "medium":
                # Preserve the original implementation: strings iterate as characters.
                metrics["bleu"] = float(sentence_bleu([row["completion"]], answer))
            else:
                gt = row["metadata"]
                env = {k: v for k, v in os.environ.items() if k in
                       ["PATH", "LANG", "LC_ALL", "TZ"]}
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
                def check(candidate):
                    with code_slot:
                        proc = subprocess.run([str(ROOT / ".venv-code/bin/python"),
                                str(ROOT / "reproduction/check_code.py")],
                                input=json.dumps(dict(answer=candidate, test=gt["test"], entry_point=gt["entry_point"])),
                                capture_output=True, text=True, timeout=300, env=env, check=True)
                    return json.loads(proc.stdout)
                gold_path = RUNS / "code_validation" / f"{row['index']:04d}.json"
                gold = json.loads(gold_path.read_text()) if gold_path.exists() else check(gt["answer"])
                atomic_json(gold_path, gold)
                if not gold["pass_rate"]:
                    raise RuntimeError("Canonical code failed; repair evaluation environment before scoring this sample")
                result = check(answer)
                metrics["pass_rate"] = result["pass_rate"]
                state["code_result"] = result["result"]
        state.update(metrics=metrics, completed=True)
        atomic_json(target, state)
        print(json.dumps(dict(index=row["index"], **metrics)), flush=True)
        return state

    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, row): row["index"] for row in records}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(dict(index=futures[future], error=f"{type(exc).__name__}: {exc}"))
                print("FAILED sample", futures[future], type(exc).__name__, flush=True)
            summary = dict(expected=len(records), completed=len(results), failed=len(errors), errors=errors)
            if results:
                summary["metrics"] = {k: dict(mean=float(np.mean([r["metrics"][k] for r in results])),
                  sem=float(np.std([r["metrics"][k] for r in results], ddof=1)/np.sqrt(len(results))) if len(results)>1 else None)
                  for k in results[0]["metrics"]}
            atomic_json(out / "summary.json", summary)
    if errors:
        raise RuntimeError(f"{len(errors)} samples failed; saved partial results can be resumed")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
