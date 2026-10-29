"""Plot completed Qwen comparisons and separate paper references, without API calls."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from reproduction.common import ROOT, RUNS, atomic_json


VARIANTS = [("base", "Base"), ("proactive", "Proactive Base"),
            ("sft", "SFT"), ("offline_dpo", "Offline DPO")]
TASKS = [("math-hard", "Math", "Accuracy", 200),
         ("bigcodebench", "Code", "Pass rate", 600),
         ("medium", "Writing", "Character BLEU", 100)]


def main():
    report = json.loads((RUNS / "report.json").read_text())
    paper = json.loads((ROOT / "reproduction/paper_table1.json").read_text())
    measured = {}
    for row in report["primary_results"]:
        key = row["task"], row["variant"]
        if key in measured or row["n"] != row["expected"] or row["failed"]:
            raise ValueError("Plot requires unique, completed primary comparisons")
        measured[key] = row
    if set(measured) != {(task, variant) for task, *_ in TASKS for variant, _ in VARIANTS}:
        raise ValueError("All three tasks and four methods must be complete")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.83, bottom=0.12,
                        hspace=0.65, wspace=0.25)
    fields = ["task_score_percent", "assistant_tokens_k", "itr_percent"]
    for task_i, (task, title, metric, paper_n) in enumerate(TASKS):
        counts = {measured[task, variant]["n"] for variant, _ in VARIANTS}
        if len(counts) != 1:
            raise ValueError("Methods must use equal sample counts")
        n = counts.pop()
        for col, field in enumerate(fields):
            ax = axes[task_i, col]
            ax.set_yticks(range(len(VARIANTS)))
            ax.set_yticklabels([label for _, label in VARIANTS] if col == 0 else [])
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_ylim(len(VARIANTS) - 0.45, -0.6)
            ax.grid(axis="x", color="#d9e0e7", linewidth=0.7)
            ax.set_axisbelow(True)
            for y, (variant, label) in enumerate(VARIANTS):
                row = measured[task, variant]
                value, error = row[field], None
                if col == 0:
                    if task == "medium":
                        error = row["task_score_sem_percent"]
                    else:
                        error = [[value - row["task_score_ci95_low_percent"]],
                                 [row["task_score_ci95_high_percent"] - value]]
                ax.errorbar(value, y - 0.10, xerr=error, fmt="o", color="#2166ac",
                            markersize=6, capsize=3, linewidth=1.4, zorder=3)
                reference = paper["rows"][label][paper["task_order"].index(task)][col]
                ax.plot(reference, y + 0.13, marker="x", color="#767d85",
                        markersize=7, linestyle="none")
            if col == 0:
                ax.set_title(f"{title}: {metric} (%)\nQwen n={n}; paper n={paper_n}",
                             loc="left", fontweight="bold", pad=12)
                ax.set_xlim(left=0)
            elif col == 1:
                ax.set_title("Assistant output (k tokens)", loc="left", pad=12)
                ax.set_xlim(left=0)
            else:
                ax.set_title("ITR (%)", loc="left", pad=12)
                ax.axvline(0, color="#9aa6b2", linewidth=0.8, linestyle="--")
    fig.suptitle("CollabLLM on Qwen3-4B", x=0.15, ha="left", y=0.975,
                 fontsize=21, fontweight="bold")
    fig.text(0.15, 0.934,
        "Qwen3-4B non-thinking / Sonnet 4.5 vs paper Llama-3.1-8B-Instruct / Sonnet 3.5.",
        fontsize=11, color="#485563")
    fig.legend(handles=[
        Line2D([], [], marker="o", color="#2166ac", linestyle="none", label="Measured Qwen experiment"),
        Line2D([], [], marker="x", color="#767d85", linestyle="none", label="Published Table 1 reference")],
        loc="upper left", bbox_to_anchor=(0.145, 0.916), ncol=2, frameon=False)
    fig.text(0.15, 0.068,
        "Accuracy/pass rate: 95% Wilson intervals. Character BLEU: ±1 SEM. One training seed.\n"
        "ITR uses the archived (raw score − 2.5) × 2 scale. PPO and Online DPO were not run.\n"
        "Paper values provide context; model, judge and implementation differences prevent a controlled comparison.",
        fontsize=9, color="#485563", va="top", linespacing=1.5)
    out = RUNS / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / f"qwen3_table1_comparison.{suffix}", dpi=180)
    plt.close(fig)
    atomic_json(out / "qwen3_table1_comparison_data.json",
                dict(measured=list(measured.values()), paper=paper, caveats=report["caveats"]))
    print(out / "qwen3_table1_comparison.png")


if __name__ == "__main__":
    main()
