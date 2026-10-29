"""Count archived online-training rollouts before committing limited judge credit."""
import json

from datasets import load_from_disk

from reproduction.common import LEGACY, RUNS, atomic_json, digest


def prefix_count(messages, window_size=2):
    # Preserve the archived scripts' literal indexing rule, including its
    # comparison to the number of even indices (not conversation length).
    even = list(range(2, len(messages), 2))
    selected = [i for i in even if i < len(even) - window_size]
    return len(selected or even)


def main():
    rows = []
    for task in ["math-hard", "bigcodebench", "medium"]:
        data = load_from_disk(str(RUNS / "data" / f"{task}-sft-paper"))
        train = sum(prefix_count(row["messages"]) for row in data["train"])
        all_slices = train + sum(prefix_count(row["messages"]) for row in data["eval"])
        # PPO's rollout batch is eight; dropped partial batches only lower this
        # nominal count. Reference-policy rewards and retries are excluded.
        ppo_rollouts = (train // 8 * 8) * 5 * 2
        online_rollouts = all_slices * 2 * 3
        rows.append(dict(task=task, sft_train_conversations=len(data["train"]),
            online_dpo_conversations=len(data["train"]) + len(data["eval"]),
            ppo_train_prefixes=train, online_dpo_train_prefixes=all_slices,
            ppo_nominal_reward_rollouts=ppo_rollouts,
            online_dpo_nominal_reward_rollouts=online_rollouts,
            claude_calls_per_successful_rollout=1 if task == "math-hard" else 2,
            reference_rewards_and_retries_included=False))
    scripts = ["scripts/ppo_train.py", "scripts/ppo_train.sh",
               "scripts/dpo_train_online.py", "scripts/dpo_train_online.sh",
               "collabllm/core/multithread.py", "collabllm/core/multiturn_reward.py"]
    result = dict(tasks=rows, archived_source_hashes={p: digest((LEGACY / p).read_text()) for p in scripts},
        interpretation="Nominal full launcher schedule, not measured training or exact released checkpoint provenance.",
        caveats=["Full PPO uses 5 epochs and 2 reward rollouts per candidate; Online DPO uses 1 epoch, 2 candidates and 3 rollouts each",
                 "Each code/writing reward additionally extracts the final answer through Claude",
                 "Online batch scoring would require awaiting each policy-dependent step; evaluation batch transport alone does not implement online training",
                 "Actual token lengths differ between short training lookahead and full evaluation conversations; pilot costs are not exact training cost estimates",
                 "The published evaluation configuration refers to selected intermediate checkpoints; their complete selection history is unavailable"])
    atomic_json(RUNS / "online_cost_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
