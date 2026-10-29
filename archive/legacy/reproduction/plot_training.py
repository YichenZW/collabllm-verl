"""Export measured SFT training curves; run with .venv-code/bin/python."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / "collabllm-runs"

def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    rows = []
    for ax, (task, title) in zip(axes, [("medium", "Writing"), ("math-hard", "Math"), ("bigcodebench", "Code")]):
        folder = RUNS / "checkpoints" / task / "sft"
        state = json.loads((folder / "trainer_state.json").read_text())
        history = [r for r in state["log_history"] if "loss" in r]
        held_out = json.loads((folder / "eval_results.json").read_text())["eval_loss"]
        ax.plot([r["epoch"] for r in history], [r["loss"] for r in history],
                color="#2166ac", lw=1.7, marker=".", label="Training loss")
        ax.axhline(held_out, color="#d6604d", ls="--", lw=1.3, label="Final validation loss")
        ax.set(title=title, xlabel="Epoch", ylabel="Cross-entropy loss", xlim=(0, 3.05))
        ax.grid(alpha=0.15)
        rows.extend(dict(task=task, step=r["step"], epoch=r["epoch"], loss=r["loss"]) for r in history)
    axes[0].legend(fontsize=8)
    fig.suptitle("Measured LoRA SFT: 450 training / 50 validation conversations per task", fontsize=12)
    output = RUNS / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "sft_training.png", dpi=180)
    fig.savefig(output / "sft_training.svg")
    with (output / "sft_training.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "step", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(output / "sft_training.png")

if __name__ == "__main__":
    main()
