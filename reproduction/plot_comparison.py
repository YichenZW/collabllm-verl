"""Plot actual completed evaluations against separate Table 1 references.

Run with .venv-code/bin/python after reproduction.report. Missing measurements
stay blank; pilot/sample counts and the changed judge are explicit.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / "collabllm-runs"
VARIANTS = [("base", "Base"), ("proactive", "Proactive Base"), ("sft", "SFT"),
            ("ppo", "PPO"), ("offline_dpo", "Offline DPO"), ("online_dpo", "Online DPO"),
            ("released", "Released adapter")]
TASKS = [("medium", "Writing", "Character BLEU", 100),
         ("bigcodebench", "Code", "Pass rate", 600),
         ("math-hard", "Math", "Accuracy", 200)]


def main():
    report = json.loads((RUNS / "report.json").read_text())
    paper = json.loads((ROOT / "reproduction/paper_table1.json").read_text())
    measured = {}
    for row in report["primary_results"]:
        key = row["task"], row["variant"]
        if key in measured:
            raise ValueError("Multiple evaluation protocols found; plot them separately")
        measured[key] = row
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.855, bottom=0.11, hspace=0.42, wspace=0.15)
    fields = ["task_score_percent", "assistant_tokens_k", "itr_percent"]
    for task_i, (task, title, score_name, paper_n) in enumerate(TASKS):
        labels = []
        for variant, label in VARIANTS:
            row = measured.get((task, variant))
            labels.append(f"{label}   n={row['n'] if row else '—'}")
        for col, field in enumerate(fields):
            ax = axes[task_i, col]
            ax.set_yticks(range(len(VARIANTS)))
            ax.set_yticklabels(labels if col == 0 else [])
            ax.tick_params(axis="y", length=0, pad=10)
            ax.set_ylim(len(VARIANTS) - 0.45, -0.6)
            ax.grid(axis="x", color="#d9e0e7", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            for y, (variant, label) in enumerate(VARIANTS):
                row = measured.get((task, variant))
                if row:
                    value = row[field]
                    error = None
                    if col == 0:
                        if task == "medium":
                            error = row["task_score_sem_percent"]
                        elif row["task_score_ci95_low_percent"] is not None:
                            error = [[max(0, value - row["task_score_ci95_low_percent"])],
                                     [max(0, row["task_score_ci95_high_percent"] - value)]]
                    ax.errorbar(value, y - 0.12, xerr=error, fmt="o", color="#2166ac",
                                markersize=5.5, capsize=2.5, linewidth=1.3, zorder=3)
                if label in paper["rows"]:
                    reference = paper["rows"][label][paper["task_order"].index(task)][col]
                    ax.plot(reference, y + 0.13, marker="x", color="#7b838c", markersize=6, linestyle="none")
            if col == 0:
                ax.set_title(f"{title} · {score_name} (%)\nPaper test n={paper_n}", loc="left", fontweight="bold", pad=12)
                ax.set_xlim(-3, 100)
            elif col == 1:
                ax.set_title("Assistant output tokens (k)", loc="left", fontweight="bold", pad=12)
                ax.set_xlim(left=0)
            else:
                ax.set_title("ITR (historical scale, %)", loc="left", fontweight="bold", pad=12)
                ax.set_xlim(-305, 105)
                ax.axvline(0, color="#9aa6b2", linewidth=0.8, linestyle="--")
    fig.suptitle("CollabLLM: measured reproduction and Table 1 references", x=0.18,
                 ha="left", y=0.985, fontsize=18, fontweight="bold")
    fig.text(0.18, 0.948, "Different judges: reproduction uses Claude Sonnet 4.5; the paper used Sonnet 3.5.",
             fontsize=11, color="#485563")
    handles = [Line2D([], [], marker="o", color="#2166ac", linestyle="none", label="Measured (completed samples only)"),
               Line2D([], [], marker="x", color="#7b838c", linestyle="none", label="Published Table 1 reference")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.174, 0.938), ncol=2, frameon=False)
    fig.text(0.18, 0.062,
        "Uncertainty: 95% Wilson intervals for accuracy/pass rate; ±1 SEM for BLEU. Blank measured rows have no completed evaluation.\n"
        "ITR preserves the archived (raw score − 2.5) × 2 transform, including negative values. Released adapter training stage is unknown.\n"
        "These are not identical-protocol replications. See reproduction/README.md for judge, training, split, and sample-count limitations.",
        fontsize=9, color="#485563", va="top", linespacing=1.5)
    out = RUNS / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "table1_comparison.png", dpi=180)
    fig.savefig(out / "table1_comparison.svg")
    (out / "table1_comparison_data.json").write_text(json.dumps(dict(measured=list(measured.values()), paper=paper), indent=2) + "\n")
    print(out / "table1_comparison.png")


if __name__ == "__main__":
    main()
