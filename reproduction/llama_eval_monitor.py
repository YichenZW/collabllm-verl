"""Read-only experiment watcher; wake this chat only on completion or trouble.

Uses the installed Codex CLI's documented `queue` command on the app-server host.
It never starts experiments, calls model APIs, changes budgets, or edits results.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

RUNS = Path(__file__).resolve().parents[1].with_name("collabllm-llama-eval-200-200-100-runs")
FAILED = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def snapshot():
    jobs = {kind: str(read(RUNS / f"{kind}_job.json")["job"])
            for kind in ["budget", "scoring", "generation", "final_audit"]}
    output = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
        "--format=JobID,State,ExitCode"], text=True, timeout=30)
    states = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 3:
            states[fields[0]] = dict(state=fields[1].split()[0].rstrip("+"), exit_code=fields[2])
    groups = []
    for task, count in [("math-hard", 200), ("bigcodebench", 200), ("medium", 100)]:
        for variant in ["base", "proactive", "sft", "offline_dpo", "released"]:
            directory = RUNS / f"eval-{count}" / task / variant
            summary = read(directory / "summary.json")
            marker = directory / "generation_complete.json"
            generated = read(marker)["expected"] if marker.exists() else summary.get("generated", 0)
            groups.append(dict(task=task, variant=variant, expected=count, generated=generated,
                               completed=summary.get("completed", 0), failed=summary.get("failed", 0)))
    budget = dict(openai=0, anthropic=0)
    for row in read(RUNS / "api_budget.json").values():
        if row["model"] != "probe allowance":
            budget["anthropic" if "claude" in row["model"] else "openai"] += row["micro"]
    return dict(jobs={kind: dict(job=job, **states.get(job, {"state":"UNKNOWN"}))
                      for kind, job in jobs.items()}, groups=groups,
                generated=sum(x["generated"] for x in groups),
                completed=sum(x["completed"] for x in groups), expected=2500,
                budget_committed_usd={k:v/1e6 for k,v in budget.items()})


def events(state):
    completed_path, audit_path = RUNS / "COMPLETED.json", RUNS / "final_execution_audit.json"
    if completed_path.exists() and audit_path.exists():
        completed, audit = read(completed_path), read(audit_path)
        assert completed["status"] == "complete" and completed["completed"] == completed["expected"] == 2500
        assert state["completed"] == 2500 and all(not x["failed"] for x in state["groups"])
        assert audit["coordinator_released"]
        for relative, expected in completed["final_artifact_sha256"].items():
            assert hashlib.sha256((RUNS / relative).read_bytes()).hexdigest() == expected
        assert state["budget_committed_usd"]["openai"] <= 53
        assert state["budget_committed_usd"]["anthropic"] <= 45
        if all(state["jobs"][kind].get("state") == "COMPLETED" and
               state["jobs"][kind].get("exit_code") == "0:0"
               for kind in ["scoring", "generation", "final_audit"]):
            return [("complete-2500", "全部 2,500 条评测与最终审计通过，GPU 生成和评分作业均成功退出。")]
    result = []
    for kind, job in state["jobs"].items():
        if kind == "budget" and audit_path.exists() and read(audit_path).get("coordinator_released"):
            continue
        if job["state"] in FAILED or (job["state"] == "COMPLETED" and kind in ["scoring", "final_audit"]
                                      and state["completed"] < 2500):
            result.append((f"job-{job['job']}-{job['state']}", f"{kind} 作业异常：{job}"))
    for provider, cap in [("openai", 53), ("anthropic", 45)]:
        used = state["budget_committed_usd"][provider]
        if used > cap or (cap - used < 0.25 and state["completed"] < 2500):
            result.append((f"budget-{provider}", f"{provider} 已结算加预留 ${used:.4f}，本轮上限 ${cap}。"))
    return result


def notify(codex, thread, event_id, detail):
    message = (
        f"[Llama 评测自动监控；事件 {event_id}] 用户已要求持续 monitor，跑完通知。{detail}\n"
        f"请读取 {RUNS}/monitor_state.json 并核实实际结果。"
        "若完成，请向用户发送中文完成通知，包含 5 个方法的 Math/Code/Writing 结果、"
        "实际 API 用量和最终报告链接；注明 PPO/Online DPO 仍暂缓。"
        "若异常，请诊断并在既有授权和 OpenAI $53 / Claude $45 本轮上限内恢复，必要时通知用户。"
        "不要扩大任务，不要重跑已完成样本，不要调整预算。"
        "这是本线程监控事件，不是新的实验请求；若该事件已处理，请避免重复通知。"
    )
    result = subprocess.run([codex, "queue", "--thread", thread, "--message", message],
        capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"codex queue exited {result.returncode}: {result.stderr[-1000:]}")
    return result.stdout[-1000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Read-only diagnostic; never sends a notification")
    args = parser.parse_args()
    codex = shutil.which("codex")
    assert codex, "Codex CLI missing"
    state_path = RUNS / "monitor_state.json"
    if args.once:
        state = snapshot()
        print(json.dumps(dict(snapshot=state, events=events(state)), ensure_ascii=False))
        return
    with (RUNS / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(state_path) if state_path.exists() else {}
        sent = previous.get("sent_events", {})
        attempted = previous.get("attempted_events", {})
        consecutive_errors = 0
        while True:
            now = time.time()
            state = dict(status="monitoring", updated=now, pid=os.getpid(), hostname=socket.gethostname(),
                         thread=args.thread, interval_seconds=args.interval,
                         sent_events=sent, attempted_events=attempted)
            try:
                state["snapshot"] = snapshot()
                pending = events(state["snapshot"])
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                state["observation_error"] = f"{type(exc).__name__}: {exc}"
                pending = [("monitor-observation-error", state["observation_error"])] if consecutive_errors >= 3 else []
            save(state_path, state)
            for event_id, detail in pending:
                if event_id in sent or now - attempted.get(event_id, 0) < 600:
                    continue
                attempted[event_id] = now
                save(state_path, state)
                try:
                    receipt = notify(codex, args.thread, event_id, detail)
                except Exception as exc:
                    state["notification_error"] = f"{type(exc).__name__}: {exc}"
                    save(state_path, state)
                    continue
                sent[event_id] = dict(at=time.time(), receipt=receipt)
                print(json.dumps(dict(event=event_id, detail=detail, queued=True)), flush=True)
                if event_id == "complete-2500":
                    state["status"] = "completed_notification_queued"
                    save(state_path, state)
                    return
                save(state_path, state)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
