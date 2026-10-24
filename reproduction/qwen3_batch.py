"""Reserved, resumable Claude batch scoring of already-generated dialogues.

Model, prompts, temperature and max_tokens match synchronous evaluation. Only the
transport and its documented 50% pricing differ. Unknown submissions retain every
reservation and are never automatically submitted a second time.
"""
import argparse
import json
import uuid
from pathlib import Path

from reproduction import budget
from reproduction.common import ROOT, RUNS, TASKS, atomic_json, chat_text, digest, legacy_prompt
from reproduction.judge_protocol import (VERSION, LEGACY_VERSION, score_params, load_score,
    save_score, apply_score, reuse_cached)

MODEL = "claude-sonnet-4-5-20250929"
BILLING_MODEL = MODEL + "@batch"


def candidates(roots, manifests):
    active, attempts, emitted = set(), {}, set()
    for manifest in manifests:
        if manifest.get("scoring_protocol") != VERSION:
            continue
        for item in manifest["items"]:
            key = item["request_key"]
            attempts[key] = attempts.get(key, 0) + 1
            if item["status"] in ["reserved", "submitted"]:
                active.add(key)
    for root in roots:
        for mp in sorted(Path(root).glob("*/*/manifest.json")):
            protocol = json.loads(mp.read_text())
            if protocol["judge"] != MODEL or protocol.get("underlying_model") != "Qwen/Qwen3-4B":
                raise ValueError("Batch scoring requires the pinned Qwen / Claude protocol")
            if protocol.get("scoring_protocol") != VERSION:
                raise ValueError("Migrate obsolete scoring before submitting any new calls")
            task = protocol["dataset"]
            for sample in sorted((mp.parent / "samples").glob("*.json")):
                state = json.loads(sample.read_text())
                if state.get("completed") or not state.get("generated"):
                    continue
                for kind, field in [("judge", "judge"), ("extract", "extraction")]:
                    params = score_params(task, state, kind, MODEL)
                    key = digest(params)
                    if field in state or key in active or (kind == "extract" and task == "math-hard"):
                        continue
                    if key in emitted or load_score(params) is not None:
                        continue
                    if attempts.get(key, 0) >= 3:
                        raise RuntimeError(f"Three unsuccessful batch attempts for {key}")
                    emitted.add(key)
                    yield dict(sample=str(sample.resolve()), kind=kind, field=field, dataset=task,
                        request_key=key, custom_id=key[:48], params=params)


def submit(client, items, destination, max_requests=64):
    record = dict(state="preparing", items=[], billing_model=BILLING_MODEL, scoring_protocol=VERSION)
    for item in items:
        if len(record["items"]) >= max_requests:
            break
        try:
            reservation = budget.reserve(BILLING_MODEL, item["params"]["messages"], item["params"]["max_tokens"])
        except budget.BudgetExceeded:
            break
        record["items"].append(dict(item, reservation=reservation, status="reserved"))
        atomic_json(destination, record)
    if not record["items"]:
        return None
    record["state"] = "submitting"
    atomic_json(destination, record)
    # SDK retries are disabled. All individual requests are already reserved.
    response = client.messages.batches.create(requests=[
        dict(custom_id=item["custom_id"], params=item["params"]) for item in record["items"]])
    record.update(state="submitted", batch_id=response.id, provider=response.model_dump(mode="json"))
    for item in record["items"]:
        item["status"] = "submitted"
    atomic_json(destination, record)
    return record


def refresh(client, path):
    record = json.loads(path.read_text())
    if record["state"] == "ended":
        return record
    if not record.get("batch_id"):
        raise RuntimeError(f"Unconfirmed batch submission: {path}; reservations retained, no automatic retry")
    response = client.messages.batches.retrieve(record["batch_id"])
    record["provider"] = response.model_dump(mode="json")
    atomic_json(path, record)
    if response.processing_status != "ended":
        return record
    from collabllm.utils.extract_json_reliable import extract_json
    from reproduction.response_schema import validate_payload

    lookup = {item["custom_id"]: item for item in record["items"]}
    for result in client.messages.batches.results(record["batch_id"]):
        item = lookup[result.custom_id]
        if item["status"] not in ["reserved", "submitted"]:
            continue
        item["raw_result"] = result.model_dump(mode="json")
        if result.result.type != "succeeded":
            item["status"] = result.result.type
            atomic_json(path, record)
            continue  # Retain the full reservation for failed/unknown requests.
        message = result.result.message
        budget.settle(item["reservation"], BILLING_MODEL, message.usage)
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            payload = validate_payload(extract_json(text), item["kind"], item["dataset"])
        except (ValueError, TypeError, KeyError) as exc:
            item.update(status="invalid", validation_error=str(exc))
            atomic_json(path, record)
            continue
        sample = Path(item["sample"])
        metadata = dict(batch_id=record["batch_id"], custom_id=result.custom_id,
            model=message.model, usage=message.usage.model_dump(), transport="batch")
        protocol = record.get("scoring_protocol", LEGACY_VERSION)
        if protocol != VERSION:
            # Already submitted old requests still get settled and archived, but
            # must never overwrite the corrected result, even if they finish later.
            item["parsed_payload"] = payload
            previous = item.get("usage_logged")
        else:
            cached = save_score(item["params"], payload, metadata)
            state = json.loads(sample.read_text())
            previous = state.get("batch_api", {}).get(item["kind"])
            if previous is not None and previous != metadata:
                raise RuntimeError("Another scorer already wrote this sample; refusing to overwrite")
            apply_score(state, item["kind"], cached)
            state.setdefault("batch_api", {})[item["kind"]] = metadata
            atomic_json(sample, state)
        if previous is None:
            with (sample.parent.parent / "api_usage.jsonl").open("a") as f:
                f.write(json.dumps(dict(kind=item["kind"], model=MODEL,
                    resolved_model=message.model, transport="batch", scoring_protocol=protocol,
                    usage=message.usage.model_dump())) + "\n")
            item["usage_logged"] = True
        item["status"] = "completed"
        atomic_json(path, record)
    if any(item["status"] in ["reserved", "submitted"] for item in record["items"]):
        raise RuntimeError("Batch response omitted requests; reservations retained")
    record["state"] = "ended"
    atomic_json(path, record)
    return record


def main():
    import anthropic
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", type=Path, default=[])
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--max-requests", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 1000:
        raise ValueError("Invalid batch size")
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic(timeout=90, max_retries=0)
    directory = RUNS / "claude_batches"
    directory.mkdir(exist_ok=True)
    lease = directory / "controller.lock"
    lease.mkdir()
    try:
        records = []
        for path in sorted(directory.glob("batch-*.json")):
            records.append(refresh(client, path))
        reused = reuse_cached(list(dict.fromkeys(args.roots + [RUNS / "eval-8"])))
        if args.submit:
            path = directory / f"batch-{uuid.uuid4().hex}.json"
            record = submit(client, candidates(args.roots, records), path, args.max_requests)
            if record:
                records.append(record)
        print(json.dumps(dict(reused_requests=reused, batches=[dict(batch_id=r.get("batch_id"), state=r["state"],
            requests=len(r["items"]), provider_status=r.get("provider", {}).get("processing_status"),
            request_counts=r.get("provider", {}).get("request_counts")) for r in records],
            budget=budget.status()), indent=2))
    finally:
        lease.rmdir()


if __name__ == "__main__":
    main()
