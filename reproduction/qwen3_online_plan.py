"""Prepare a full online-training proposal from archived schedules and saved costs.

This command makes no API calls, changes no budget, and submits no training jobs.
The cost projection uses evaluation token usage as a proxy, not a training pilot.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from reproduction.common import ROOT, RUNS, LEGACY, TASKS, atomic_json, digest
from reproduction.qwen3_costs import summarize_usage
from reproduction.judge_protocol import VERSION, load_score, score_params


def scoring_costs_from_samples(directory):
    """Use request metadata already embedded in audited completed samples.

    Older seed samples need the separate scoring cache; current samples already
    contain the same provenance. Avoid thousands of redundant Lustre reads.
    """
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    if (manifest.get('scoring_protocol') != VERSION or
            summary.get('scoring_protocol') != VERSION or summary['failed'] or
            summary['completed'] != summary['expected']):
        raise ValueError(f'Incomplete or obsolete scoring: {directory}')
    paths = list((directory / 'samples').glob('*.json'))
    assert len(paths) == summary['expected']
    lines = []
    for path in paths:
        sample = json.loads(path.read_text())
        assert sample['completed'] and sample['scoring_protocol'] == VERSION
        for kind in (['judge'] if manifest['dataset'] == 'math-hard' else ['judge', 'extract']):
            params = score_params(manifest['dataset'], sample, kind, manifest['judge'])
            assert sample['score_requests'][kind] == digest(params)
            metadata = sample.get('score_sources', {}).get(kind)
            if metadata is None:
                cached = load_score(params)
                if cached is None:
                    raise ValueError(f'Missing paid scoring provenance: {path}')
                metadata = cached['metadata']
            lines.append(json.dumps(dict(kind=kind, model=manifest['judge'],
                usage=metadata['usage'], transport=metadata['transport'])))
    return dict(n=len(paths), **summarize_usage(lines, len(paths)))


def user_cost(directory, model):
    rates = {'gpt-4o': (2.5, 10), 'gpt-4o-mini': (0.15, 0.6)}
    input_rate, output_rate = rates[model]
    total = calls = 0
    for line in (directory / 'api_usage.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['kind'] != 'user':
            continue
        u = row['usage']
        total += (u.get('prompt_tokens', u.get('input_tokens', 0)) * input_rate
                  + u.get('completion_tokens', u.get('output_tokens', 0)) * output_rate) / 1e6
        calls += 1
    if not calls:
        raise ValueError(f'No saved user usage: {directory}')
    return dict(model=model, observed_calls=calls, repriced_total_usd=total,
                per_call_usd=total / calls, source=str(directory / 'api_usage.jsonl'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT.parent / 'collabllm-qwen3-4b-online-runs' / 'proposal.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing proposal; choose another output path')
    completed = json.loads((RUNS / 'COMPLETED.json').read_text())
    assert completed['status'] == 'complete'
    counts = json.loads((RUNS / 'online_cost_audit.json').read_text())['tasks']
    limits = {'math-hard': 200, 'bigcodebench': 225, 'medium': 100}
    stages, evidence = [], {}
    for task in ['math-hard', 'bigcodebench', 'medium']:
        info = next(x for x in counts if x['task'] == task)
        cfg = TASKS[task]
        evidence[task] = {}
        for method, initial, epochs in [('ppo', 'sft', 5), ('online_dpo', 'offline_dpo', 1)]:
            directory = RUNS / f'eval-{limits[task]}' / task / initial
            costs = scoring_costs_from_samples(directory)
            user = 'gpt-4o' if method == 'ppo' and task == 'medium' else 'gpt-4o-mini'
            observed_user = user_cost(directory, user)
            evaluation_user = user_cost(directory, cfg['user'])
            rollout_count = info[f'{method}_nominal_reward_rollouts']
            prefix_count = info[f'{method}_train_prefixes']
            checkpoint = RUNS / 'checkpoints' / task / initial
            assert (checkpoint / 'adapter_model.safetensors').exists()
            stage = dict(task=task, method=method, initial_checkpoint=str(checkpoint),
                epochs=epochs, prefix_count=prefix_count, rollout_count=rollout_count,
                claude_requests=rollout_count * info['claude_calls_per_successful_rollout'],
                learning_rate=2e-6 if method == 'ppo' else 5e-6,
                window_rounds=2, samples_per_candidate=2 if method == 'ppo' else 3,
                candidates_per_prefix=1 if method == 'ppo' else 2,
                candidate_max_new_tokens=cfg['max_tokens'], max_query_tokens=4096,
                user_model=user, judge_model='claude-sonnet-4-5-20250929',
                judge_temperature=0.2, judge_max_tokens=4096,
                reward=dict(interactivity='raw score / 3 (archived training default)',
                    task_metric='accuracy' if task == 'math-hard' else 'pass_rate' if task == 'bigcodebench' else 'character_bleu',
                    interactivity_weight=1, task_weight=1,
                    token_cost_weight=1e-4 if task == 'medium' else 5e-4,
                    token_scope='cl100k tokens of newline-joined candidate plus forward conversation, all roles'),
                reference='base model with adapter disabled' if method == 'ppo' else 'frozen initial Offline DPO policy',
                training_cost_proxy_usd=dict(
                    anthropic=rollout_count * costs['claude_synchronous_per_sample_usd'],
                    openai=rollout_count * 2 * observed_user['per_call_usd']))
            if method == 'ppo':
                stage.update(rollout_batch_size=8, mini_batch_size=2, gradient_accumulation_steps=4,
                    ppo_epochs=2, use_score_scaling=True, use_score_norm=False,
                    rollout_batches=prefix_count // 8 * epochs,
                    nominal_optimizer_updates=prefix_count // 8 * epochs * 2,
                    assistant_temperature=0.8, trainer_seed=0, dataset_seed=42)
            else:
                batch = 2 if task == 'math-hard' else 1
                stage.update(per_device_train_batch_size=batch, gradient_accumulation_steps=4,
                    beta=0.1, loss='sigmoid', warmup_ratio=0.1, lr_scheduler='cosine',
                    nominal_optimizer_updates=math.ceil(math.ceil(prefix_count / batch) / 4),
                    minimum_gap_filter=False,
                    candidate_temperature=0.9, candidate_top_p=1.0, candidate_top_k=0,
                    candidate_min_new_tokens=cfg['max_tokens'],
                    lookahead_temperature=cfg['temperature'],
                    note='Archived minimum_gap is parsed but not applied to online pairs. Fix first-row judge mapping for batches with multiple QA pairs. See implementation_audit.json for required runtime integration and validation.')
            stages.append(stage)
            evidence[task][initial] = dict(
                score_source=str(directory), score_sample_count=costs['n'],
                claude_sync_usd_per_evaluation=costs['claude_synchronous_per_sample_usd'],
                claude_batch_usd_per_evaluation=costs['claude_batch_projected_per_sample_usd'],
                training_user=observed_user, evaluation_user=evaluation_user)
            print(json.dumps(dict(task=task, method=method, cost_proxy_usd=stage['training_cost_proxy_usd'])), flush=True)
    # Full original test sizes for both new methods, plus the outstanding code
    # prefix for all four completed methods to obtain a paired six-method table.
    evaluation = []
    for task in ['math-hard', 'bigcodebench', 'medium']:
        for method, initial in [('ppo', 'sft'), ('online_dpo', 'offline_dpo')]:
            e = evidence[task][initial]
            # Evaluation generation cost comes from actual full dialogues.
            directory = Path(e['score_source'])
            history = json.loads((RUNS / ('code225_cost_observation.json' if task == 'bigcodebench'
                                          else 'writing100_cost_observation.json')).read_text()) if task != 'math-hard' else None
            if task == 'bigcodebench':
                generation = next(x for x in history['records'] if x['variant'] == initial)['generation']['openai_per_sample_usd']
            elif task == 'medium':
                generation = history['records'][initial]['generation']['openai_per_sample_usd']
            else:
                # This stage reuses 32 dialogues, so its usage covers 168 new ones.
                generation = e['evaluation_user']['repriced_total_usd'] / (limits[task] - 32)
            evaluation.append(dict(task=task, method=method, n=TASKS[task]['n'],
                proxy_usd=dict(anthropic=TASKS[task]['n'] * e['claude_batch_usd_per_evaluation'],
                               openai=TASKS[task]['n'] * generation)))
    for row in json.loads((RUNS / 'code225_cost_observation.json').read_text())['records']:
        evaluation.append(dict(task='bigcodebench', method=row['variant'], n=600-225,
            proxy_usd=dict(anthropic=(600-225)*row['scoring']['claude_batch_projected_per_sample_usd'],
                openai=(600-225)*row['generation']['openai_per_sample_usd'])))
    total = {provider: sum(s['training_cost_proxy_usd'][provider] for s in stages)
             + sum(e['proxy_usd'][provider] for e in evaluation) for provider in ['openai', 'anthropic']}
    proposed_cap = {p: math.ceil(1.5 * value / 50) * 50 for p, value in total.items()}
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), status='awaiting_additional_api_budget',
        paid_training_submitted=False, additional_api_authorization=None,
        model='Qwen/Qwen3-4B', enable_thinking=False, task_order=['math-hard','bigcodebench','medium'],
        stages=stages, evaluation=evaluation, proxy_evidence=evidence,
        nominal_rollouts=sum(s['rollout_count'] for s in stages),
        nominal_claude_requests=sum(s['claude_requests'] for s in stages),
        total_cost_proxy_usd=total, proposed_additional_caps_with_50percent_margin_usd=proposed_cap,
        prior_completed_run=str(RUNS), prior_committed_usd=completed['budget'],
        caveats=[
            'Projection only, not a quote or measured online-training cost. Short lookahead, prefix lengths, temperature and termination differ from evaluation.',
            'Training assumes synchronous Claude prices; batch discounts are used only for final evaluation.',
            'Two simulator calls per rollout are budgeted; early termination can reduce this and retries can increase it.',
            'GPU time is not measured yet. Full saved-state resume and per-request metering are required before paid training.',
            'Original reference rewards are disabled; the complete recorded launcher schedule is used, not the selected intermediate paper checkpoints.',
            'Preserve completed offline results and their ledger; new spending requires explicit additional authorization.'
        ])
    result['archived_source_sha256'] = {name: hashlib.sha256((LEGACY/name).read_bytes()).hexdigest()
        for name in ['scripts/ppo_train.py','scripts/ppo_train.sh','scripts/dpo_train_online.py',
                     'scripts/dpo_train_online.sh','external/online_dpo_trainer.py','collabllm/metrics/llm_judge.py']}
    atomic_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), rollouts=result['nominal_rollouts'],
        claude_requests=result['nominal_claude_requests'], proxy_usd=total,
        proposed_additional_caps_usd=proposed_cap, stages=[{k:s[k] for k in
            ['task','method','epochs','prefix_count','nominal_optimizer_updates','training_cost_proxy_usd']} for s in stages]), indent=2))


if __name__ == '__main__':
    main()
