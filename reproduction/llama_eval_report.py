"""Audit the fixed cohort and compare with Qwen on exactly the same inputs."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from reproduction.common import ROOT, RUNS, atomic_json, digest
from reproduction import report, analyze_results, budget
from reproduction.code_executor import harness_identity
from reproduction.judge_protocol import VERSION, score_params

LABELS = dict(base="Base", proactive="Proactive Base", sft="SFT", offline_dpo="Offline DPO",
              released="作者公开 checkpoint（训练阶段未确认）")
TARGETS = {"math-hard":200, "bigcodebench":200, "medium":100}
METRICS = {"math-hard":"accuracy", "bigcodebench":"pass_rate", "medium":"bleu"}
QWEN = ROOT.parent / "collabllm-qwen3-4b-runs"


def samples(directory, task, expected):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["limit"] == expected and manifest["scoring_protocol"] == VERSION
    rows = json.loads((RUNS / "data" / f"{task}-test.json").read_text())[:expected]
    assert manifest["test_sha256"] == digest(rows)
    paths = list((directory / "samples").glob("*.json"))
    assert len(paths) == expected
    values = {s["index"]:s for s in [json.loads(p.read_text()) for p in paths]}
    assert set(values) == {r["index"] for r in rows}
    for row in rows:
        s = values[row["index"]]
        assert s["input"] == row and s["completed"] and s["scoring_protocol"] == VERSION
        for kind in (["judge"] if task == "math-hard" else ["judge","extract"]):
            assert s["score_requests"][kind] == digest(score_params(task,s,kind,manifest["judge"]))
        if task == "bigcodebench":
            assert s["code_harness"] == harness_identity()
    return values, manifest


def main():
    # A report labeled complete is only emitted after every required group passes.
    checked, paired, snapshots = [], [], {}
    for task, count in TARGETS.items():
        for variant in LABELS:
            directory = RUNS / f"eval-{count}" / task / variant
            vals, manifest = samples(directory, task, count)
            summary = json.loads((directory / "summary.json").read_text())
            assert summary["completed"] == count and summary["failed"] == 0
            score = mean(s["metrics"][METRICS[task]] for s in vals.values()) * 100
            assert abs(score - summary["metrics"][METRICS[task]]["mean"]*100) < 1e-8
            checked.append(dict(dataset=task, variant=variant, count=count, score=score))
            for filename in ["manifest.json", "summary.json"]:
                path = directory / filename
                snapshots[str(path.relative_to(RUNS))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if variant == "released":
                continue
            qvals, qmanifest = samples(QWEN / f"eval-{count}" / task / variant, task, count)
            for key in ["test_sha256","judge","user_model","seed","protocol","scoring_protocol"]:
                assert manifest[key] == qmanifest[key], (task,variant,key)
            assert vals.keys() == qvals.keys()
            qscore = mean(s["metrics"][METRICS[task]] for s in qvals.values()) * 100
            paired.append(dict(task=task, variant=variant, n=count,
                llama_score_percent=score, qwen_score_percent=qscore,
                qwen_minus_llama_points=qscore-score,
                llama_tokens_k=mean(s["metrics"]["tokens_k"] for s in vals.values()),
                qwen_tokens_k=mean(s["metrics"]["tokens_k"] for s in qvals.values()),
                llama_itr_percent=100*mean(s["metrics"]["itr"] for s in vals.values()),
                qwen_itr_percent=100*mean(s["metrics"]["itr"] for s in qvals.values())))
    active = []
    for path in (RUNS / "claude_batches").glob("batch-*.json"):
        record = json.loads(path.read_text())
        if record["state"] != "ended":
            active.append(str(path))
    assert not active, active
    report.main()
    analysis_source = QWEN / "code_split_audit.json"
    audit = RUNS / "code_split_audit.json"
    if not audit.exists():
        audit.symlink_to(analysis_source)
    analyze_results.main()
    atomic_json(RUNS / "matched_qwen_comparison.json", paired)
    with (RUNS / "matched_qwen_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    data = json.loads((RUNS / "report.json").read_text())
    lines = ["# Llama-3.1-8B CollabLLM：200 Math / 200 Code / 100 Writing", "",
        "五种已有方法的 2,500 条评测均已完成。复用 1,172 条旧对话，新增 1,328 条；"
        "所有样本使用补全 Claude JSON 指令后的同一评分协议。旧评分保留在原运行目录。", "",
        "| 任务 | 方法 | n | 任务分数 | 输出 k tokens | ITR % | 论文分数 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in data["primary_results"]:
        paper = row["paper_task_score_percent"]
        lines.append(f"| {row['task']} | {LABELS[row['variant']]} | {row['n']} | {row['task_score_percent']:.2f} | "
            f"{row['assistant_tokens_k']:.3f} | {row['itr_percent']:.2f} | {paper if paper is not None else '—'} |")
    lines += ["", "任务分数分别为 accuracy %、code pass rate %、character BLEU ×100。"
        "Code 使用固定前 200 题，是论文 600 题中的子集；Math 和 Writing 覆盖各自完整测试集。", "",
        "Qwen 同题比较（Code 取已有 200 题结果，不使用 225 题的平均分）：", "",
        "| 任务 | 方法 | Llama | Qwen | Qwen − Llama |", "|---|---|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['task']} | {LABELS[row['variant']]} | {row['llama_score_percent']:.2f} | "
                     f"{row['qwen_score_percent']:.2f} | {row['qwen_minus_llama_points']:+.2f} |")
    b = budget.status()
    lines += ["", "沿用已完成的 SFT / Offline DPO 权重，没有重新训练。作者公开 checkpoint 的训练阶段未确认，"
        "不能当作本地 PPO 或 Online DPO；这两项完整训练继续暂缓。", "",
        "裁判为 Claude Sonnet 4.5，替代论文的 Claude 3.5；使用原代码提示词、轮数、用户模拟模型和字符 BLEU。"
        "生成采用 vLLM，训练沿用 BF16 / 现代 TRL，因此不能宣称与论文运行条件完全一致。"
        "单次对话与单次训练种子的差异不代表统计上确定的优势。", "",
        "代码测试与 Qwen 使用相同的执行器版本和原始断言。训练数据重叠的敏感性结果及配对检验见 "
        "`statistical_analysis.json`；公开 checkpoint 的训练数据来源不能由该重叠检查推断。", "",
        f"本轮 API 已结算加保留金额：OpenAI ${b['providers']['openai']['committed_usd']:.4f} / $53，"
        f"Claude ${b['providers']['anthropic']['committed_usd']:.4f} / $45。金额按用量保守计算，并非供应商账单。", ""]
    (RUNS / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    completed = dict(status="complete", updated=datetime.now(timezone.utc).isoformat(),
        expected=2500, completed=2500, groups=checked, final_artifact_sha256=snapshots,
        budget=b, active_batches=active, original_results_modified=False,
        report=str(RUNS / "RESULTS.zh-CN.md"), comparison=str(RUNS / "matched_qwen_comparison.csv"))
    atomic_json(RUNS / "COMPLETED.json", completed)
    print(json.dumps(dict(status="complete", completed=2500, budget=b)))


if __name__ == "__main__":
    main()
