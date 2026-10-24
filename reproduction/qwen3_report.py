"""Qwen-specific paired comparisons; incomplete experiments remain explicit."""
import csv
import json
from collections import defaultdict

from reproduction import report
from reproduction.common import ROOT, RUNS, atomic_json
from reproduction.code_executor import harness_identity
from reproduction.judge_protocol import VERSION

VARIANTS = ("base", "proactive", "sft", "offline_dpo")
TASKS = ("math-hard", "bigcodebench", "medium")
LABELS = dict(base="Base", proactive="Proactive Base", sft="SFT", offline_dpo="Offline DPO")


def paired_results(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row.get("scoring_protocol") != VERSION:
            continue
        if row["n"] != row["expected"] or row["failed"]:
            continue
        if row["task"] == "bigcodebench":
            samples = [json.loads(p.read_text()) for p in
                       (RUNS / row["run"] / "samples").glob("*.json")]
            if len(samples) != row["expected"] or any(
                    not s.get("completed") or s.get("code_harness") != harness_identity()
                    for s in samples):
                continue
        key = (row["task"], row["run"].split("/")[0], row["user"], row["judge"])
        groups[key][row["variant"]] = row
    selected = {}
    for (task, _, _, _), group in groups.items():
        if not all(variant in group for variant in VARIANTS):
            continue
        counts = {group[v]["n"] for v in VARIANTS}
        if len(counts) != 1:
            raise ValueError("Paired comparison has different sample counts")
        signatures = []
        for variant in VARIANTS:
            manifest = json.loads((RUNS / group[variant]["run"] / "manifest.json").read_text())
            signatures.append(tuple(manifest[k] for k in ["test_sha256", "underlying_model",
                "model_revision", "enable_thinking", "generation_config", "seed", "protocol", "scoring_protocol"]))
        if len(set(signatures)) != 1:
            raise ValueError("Paired comparisons have different inputs or protocols")
        if task not in selected or group["base"]["n"] > selected[task]["base"]["n"]:
            selected[task] = group
    return [selected[task][variant] for task in TASKS if task in selected for variant in VARIANTS]


def main():
    report.main()
    path = RUNS / "report.json"
    data = json.loads(path.read_text())
    data["individual_complete_results"] = [row for row in data["primary_results"]
                                           if row.get("scoring_protocol") == VERSION]
    data["primary_results"] = paired_results(data["measured_results"])
    data["model"] = json.loads((RUNS / "assets.json").read_text())
    data["plan"] = json.loads((RUNS / "plan.json").read_text())
    data["caveats"] = [
        "An omitted Claude JSON-format wrapper was corrected; superseded scores are archived and all primary scores require the restored archived wrapper",
        "Qwen/Qwen3-4B non-thinking replaces paper Llama-3.1-8B-Instruct; paper values are context, not controlled comparisons",
        "vLLM and deterministic per-request assistant seeds retain the prior reproduction pipeline; they do not reproduce archived local-HF random streams",
        "Proactive Base applies the intended archived proactive prompt; the archived local-HF branch ignored prompt_method",
        "Claude Sonnet 4.5 replaces retired Claude 3.5; the initial pilot is synchronous and expansions use Message Batches",
        "Archived task-specific judge prompts differ from paper v3 Appendix D.4; historical ITR scaling is preserved",
        "Single seed; paired fixed-prefix subsets selected by API cost, not outcomes; sampling uncertainty remains",
        "BF16 and modern TRL replace the archived NF4 training implementation",
        "Archived training data and original test splits are retained; code overlap sensitivity is reported separately",
        "A full archived-function data audit passed; BigCodeBench ground-truth JSON retains identical content but uses a different key insertion order",
        "Code execution retains original tests with documented HTTP-header and owned-child cleanup compatibility fixes; code_harness hashes identify rescored outputs",
        "Private setuptools80.9 overlay restores archived librosa pkg_resources imports; shared packages remain unchanged",
        "Audited BigCodeBench/111 and /276 reference/test inconsistencies retain original candidate scoring and enter a separate sensitivity analysis",
        "Only fully completed four-method comparisons enter primary_results; individual and partial runs remain available",
        "PPO/Online DPO are not measured unless a full training and evaluation record explicitly exists",
        "The API ledger includes conservative reservations; usage-derived expenses are estimates, not provider invoices",
    ]
    atomic_json(path, data)
    if data["primary_results"]:
        with (RUNS / "comparison.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data["primary_results"][0]))
            writer.writeheader()
            writer.writerows(data["primary_results"])
    else:
        (RUNS / "comparison.csv").unlink(missing_ok=True)
    budget = data["api_budget"]["providers"]
    lines = ["# Qwen3-4B CollabLLM 实验结果", "",
        "底座为 `Qwen/Qwen3-4B`，关闭 thinking。下表只列同一任务、相同题目上四个方法均已完成的结果。", "",
        "已补回原版 Claude API 包装层的 JSON 格式指令。遗漏此指令时产生的旧分数已归档；"
        "下表仅接受修正后的评分，对话和训练保持不变。", "",
        "| 任务 | 方法 | 样本数 | 任务分数 (%) | 助手输出 (k tokens) | ITR (%) | 论文任务分数 (%) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in data["primary_results"]:
        lines.append(f"| {row['task']} | {LABELS[row['variant']]} | {row['n']} | "
            f"{row['task_score_percent']:.2f} | {row['assistant_tokens_k']:.3f} | "
            f"{row['itr_percent']:.2f} | {row['paper_task_score_percent']:.2f} |")
    if not data["primary_results"]:
        lines += ["", "尚无四个方法全部完成的配对结果。进行中的单项结果见 `report.json`。"]
    paired_counts = {(row["task"], row["variant"]): row["n"] for row in data["primary_results"]}
    individual = [row for row in data["individual_complete_results"]
                  if row["n"] > paired_counts.get((row["task"], row["variant"]), 0)]
    if individual:
        lines += ["", "以下单项已完成，但相同样本规模的四方法对照尚未齐全，暂不作方法优劣结论。", "",
            "| 任务 | 方法 | 已完成样本 | 任务分数 (%) | 助手输出 (k tokens) | ITR (%) |",
            "|---|---|---:|---:|---:|---:|"]
        for row in individual:
            lines.append(f"| {row['task']} | {LABELS[row['variant']]} | {row['n']} | "
                f"{row['task_score_percent']:.2f} | {row['assistant_tokens_k']:.3f} | "
                f"{row['itr_percent']:.2f} |")
    lines += ["", "任务分数分别为 math accuracy、code pass rate、writing character BLEU。"
        "ITR 保留原代码 `(raw−2.5)×2`，因此可为负。论文数字使用不同底座和裁判，只作背景参考。", "",
        f"本轮已结算加保留金额：OpenAI ${budget['openai']['committed_usd']:.6f}，"
        f"Claude ${budget['anthropic']['committed_usd']:.6f}；程序硬上限分别为 "
        f"${budget['openai']['cap_usd']:g} 和 ${budget['anthropic']['cap_usd']:g}。", "",
        "金额按 API 用量和预留额保守估算，并非供应商账单；失败调用和修正评分的费用也计入同一预算。", "",
        "按已确认的预算优先级，本轮评测 Base、Proactive Base、SFT 和 Offline DPO；"
        "PPO、Online DPO 未运行。完整数据、置信区间和训练状态见 `report.json`、"
        "`comparison.csv`、`status.json`；配对检验及代码训练集重叠敏感性分析见 `statistical_analysis.json`。", "",
        "训练复用作者发布的准备后数据，未重新采集 Qwen 偏好数据。SFT 为 3 epochs，Offline DPO 为 8 epochs；"
        "使用最后一个 epoch 的 checkpoint。裁判使用 Claude Sonnet 4.5 替代已退役的 3.5，"
        "训练使用 BF16 和现代 TRL，生成使用 vLLM。单次训练种子和每题一次对话不能衡量重复训练或 API 重跑的变异。", ""]
    coverage = data["plan"]["evaluation"].get("final_coverage")
    if coverage:
        description = "、".join(
            f"{task} {coverage[task]['samples_per_variant']}/{data['plan']['evaluation']['paper_test_size'][task]}"
            for task in TASKS)
        lines += [f"最终每方法覆盖：{description}。样本数按实际 API 成本与预留余量决定，"
            "未按结果或显著性选择；预算决策见 `plan.json` 的 `evaluation.final_coverage`。", "",
            "离线对照图见 `figures/qwen3_table1_comparison.png`（另有 SVG/PDF）；"
            "完整验证结果见 `final_audit.json` 和 `final_offline_pipeline_audit.json`。", ""]
    if (RUNS / "code_split_audit.json").exists():
        from reproduction.analyze_results import main as analyze
        analyze()
        statistics = json.loads((RUNS / "statistical_analysis.json").read_text())
        lines += ["配对比较使用相同题目上的两侧精确 McNemar 检验。差值为后者减前者，"
            "Holm 校正在各任务列出的比较族内分别进行。", "",
            "| 任务 | 前者 → 后者 | 样本数 | 差值 (百分点) | 原始 p | Holm 校正 p |",
            "|---|---|---:|---:|---:|---:|"]
        for task, key in [("math-hard", "math_paired"), ("bigcodebench", "code_paired")]:
            for comparison in statistics[key]:
                lines.append(f"| {task} | {LABELS[comparison['a']]} → {LABELS[comparison['b']]} | "
                    f"{comparison['n']} | {comparison['difference_percentage_points']:+.2f} | "
                    f"{comparison['mcnemar_exact_two_sided_p']:.4f} | "
                    f"{comparison['mcnemar_holm_adjusted_p']:.4f} |")
        lines += ["", "Writing BLEU 的标准误见结果数据与图；上表的二元正确性检验不适用于 BLEU。", ""]
    lines += ["具体设置与限制见 `report.json` 的 caveats 和代码目录 `reproduction/QWEN3.md`。"]
    (RUNS / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
