"""Recover every paper online-training prefix and its real task annotation.

Runs archived preprocessing functions offline. No model API is imported or used.
"""
import argparse
import copy
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, interleave_datasets, load_from_disk
from transformers import AutoTokenizer

from reproduction.common import LEGACY, ROOT, RUNS, TASKS, atomic_json, digest, legacy_prompt
from reproduction.prepare import pinned_dataset
from reproduction.qwen3_data_audit import archived_functions


def single_turn_train(task):
    config = {
        'math-hard': ('lighteval/MATH-Hard', 'math_hard.py', 'MATH', {}),
        'bigcodebench': ('bigcode/bigcodebench', 'bigcodebench.py', 'BigCodeBench', {'split': 'v0.1.2'}),
        'medium': ('Kamaljp/medium_articles', 'medium.py', 'Medium', {}),
    }
    repo, filename, class_name, kwargs = config[task]
    raw = pinned_dataset(repo, **kwargs)
    if task == 'medium':
        raw = dict(raw)
    encoder = tiktoken.get_encoding('cl100k_base')
    ns = dict(json=json, random=random, tqdm=lambda rows: rows,
        num_tokens_from_string=lambda text: len(encoder.encode(text, disallowed_special=())))
    source = LEGACY / 'collabllm/datasets' / filename
    preprocess = archived_functions(source, ['preprocess'], ns, class_name)['preprocess']
    # Training load_single_turn_dataset uses BigCodeBench's default 0.8.
    # The archived evaluation explicitly uses 0.45; these are different splits.
    rows = preprocess(SimpleNamespace(train_ratio=0.8 if task == 'bigcodebench' else 0.02,
        test_ratio=0.005, max_tokens=512), raw)
    return [row for row in rows if row['metadata']['split'] == 'train']


def conversations(task, single, holdout):
    raw = list(load_from_disk(str(LEGACY / 'data' / f'interactive-{task}'),
                             keep_in_memory=True)['train'])
    qa = [row['chat'] for row in single]
    ns = dict(copy=copy, random=random, re=re, np=np, List=List, Dataset=Dataset,
        DatasetDict=DatasetDict, interleave_datasets=interleave_datasets,
        SYSTEM_PROMPT=legacy_prompt('llm_assistant/system_prompt.txt'),
        datasets_info={task: {'task': TASKS[task]['task']}},
        load_dataset=lambda *a, **kw: DatasetDict(train=Dataset.from_list(raw)),
        load_single_turn_dataset=lambda *a, **kw: {'train': {'chat': qa}})
    archived_functions(LEGACY / 'collabllm/datasets/__init__.py',
        ['add_sys_prompt', 'load_dpo_dataset', 'filter_dpo', 'split_train_dev_datasets'], ns)
    ns['load_multiturn_dataset'] = lambda name, is_dpo, **kw: ns['load_dpo_dataset'](name, **kw)
    train, _ = ns['split_train_dev_datasets'](f'collabllm/collabllm-{task}',
        is_dpo=True, to_sft_dataset=True, n_eval_per_dataset=holdout,
        add_system_prompt=True, return_eval_as_dict=False, seed=42)
    for row in train:
        assert row['qa'] == qa[row['idx']]
    return train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path,
        default=ROOT.parent / 'collabllm-qwen3-4b-online-runs')
    args = p.parse_args()
    out = args.output
    assert not (out / 'data_audit.json').exists(), 'Preserve existing evidence'
    proposal = json.loads((out / 'proposal.json').read_text())
    assets = json.loads((RUNS / 'assets.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets['base'], local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Check.'}],
        tokenize=False, add_generation_prompt=True)
    assert rendered.endswith('<think>\n\n</think>\n\n'), 'Expected pinned non-thinking template'
    records = []
    for task in ['math-hard', 'bigcodebench', 'medium']:
        single = single_turn_train(task)
        for method, holdout in [('ppo', 50), ('online_dpo', 0)]:
            train = conversations(task, single, holdout)
            tokenizer.truncation_side = 'right' if method == 'ppo' else 'left'
            filename = 'ppo_train.py' if method == 'ppo' else 'dpo_train_online.py'
            ns = dict(np=np, copy=copy, defaultdict=defaultdict, Dataset=Dataset,
                args=SimpleNamespace(window_size=2, max_query_tokens=4096), tokenizer=tokenizer)
            process = archived_functions(LEGACY / 'scripts' / filename,
                                         ['process_dataset'], ns)['process_dataset']
            processed = process(train)
            expected = next(s for s in proposal['stages'] if s['task'] == task and s['method'] == method)
            assert len(processed) == expected['prefix_count']
            result = []
            for position, row in enumerate(processed):
                history = row['chat_history'] if method == 'ppo' else row['chat']
                ids = row['input_ids'] if method == 'ppo' else row['prompt_input_ids']
                ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
                assert history[0]['role'] == 'system' and history[-1]['role'] == 'user'
                assert row['qa'] == single[row['idx']]['chat']
                assert 0 < len(ids) <= 4096
                result.append(dict(position=position, conversation_id=int(row['idx']),
                    history=history, qa=row['qa'], metadata=single[row['idx']]['metadata'],
                    input_ids=ids))
            path = out / 'data' / f'{task}-{method}.json'
            atomic_json(path, result)
            record = dict(task=task, method=method, conversations=len(train), prefixes=len(result),
                all_task_annotations_verified=True, archived_prefix_function_executed=True,
                sha256=digest(result), path=str(path), truncation_side=tokenizer.truncation_side,
                min_input_tokens=min(len(x['input_ids']) for x in result),
                max_input_tokens=max(len(x['input_ids']) for x in result),
                full_single_turn_training_pool=len(single))
            records.append(record)
            print(json.dumps(record), flush=True)
    sources = ['collabllm/datasets/__init__.py','collabllm/datasets/math_hard.py',
        'collabllm/datasets/bigcodebench.py','collabllm/datasets/medium.py',
        'scripts/ppo_train.py','scripts/dpo_train_online.py']
    atomic_json(out / 'data_audit.json', dict(updated=datetime.now(timezone.utc).isoformat(),
        status='passed', records=records, model_revision=assets['revision'],
        no_paid_api_calls=True, source_sha256={n:hashlib.sha256((LEGACY/n).read_bytes()).hexdigest()
                                             for n in sources},
        notes=['Real train-split QA annotations are required for online rewards; they were intentionally unused in the prior SFT/DPO fidelity check.',
               'BigCodeBench training QA uses archived default train_ratio0.8; final evaluation keeps original0.45 split and its documented overlap.']))


if __name__ == '__main__':
    main()
