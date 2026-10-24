"""Paper online reward semantics, independent of API clients and trainers.

These helpers do not initiate generation or spend credit. Final evaluation keeps
its separate ITR transformation and temperature-zero scoring protocol.
"""
import math

from reproduction.common import TASKS, chat_text, legacy_prompt
from reproduction.judge_protocol import wrap_json_prompt
from reproduction.response_schema import validate_payload

VERSION = 'd573e49-online-reward-qwen3-v1'


def paired_row_indices(batch_size):
    """Match HF repeat(2, 1): a0,b0,...,a1,b1,..., not adjacent pairs.

    The archived trainer incorrectly gives every completion the first row's QA.
    A batch of two different math problems must retain both task annotations.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Expected a positive integer batch size')
    return list(range(batch_size)) * 2


def request_params(task, kind, qa, history, forward, model):
    """Build exact archived prompts, with the current judge compatibility fix.

    Claude Sonnet 4.5 replaces the retired 3.5 judge. Keep temperature 0.2 and
    omit top_p: this model does not permit setting both sampling parameters.
    """
    cfg = TASKS[task]
    if len(qa) != 2 or [m['role'] for m in qa] != ['user', 'assistant']:
        raise ValueError('Expected the original question and ground truth')
    if kind == 'judge':
        prompt = legacy_prompt(f"llm_judge/{cfg['task']}.txt").format(
            question=qa[0]['content'], answer=qa[1]['content'],
            chat_history=chat_text(history), chat=chat_text(forward))
    elif kind == 'extract':
        prompt = legacy_prompt('funtional/extract_answer.txt').format(
            chat=chat_text(history + forward))
    elif kind == 'user':
        prompt = legacy_prompt(f"user_simulator_cot/{cfg['task']}.txt").format(
            question=qa[0]['content'], answer=qa[1]['content'],
            chat_history=chat_text(history + forward))
    else:
        raise ValueError(f'Unknown request kind: {kind}')
    params = dict(model=model, max_tokens=4096,
        temperature=1.0 if kind == 'user' else 0.2,
        messages=[dict(role='user', content=wrap_json_prompt(prompt, model))])
    if kind == 'user':
        if not model.startswith('gpt-4o'):
            raise ValueError('The paper simulator requires the specified GPT-4o model')
        params.update(top_p=1.0, response_format={'type': 'json_object'})
    elif model != 'claude-sonnet-4-5-20250929':
        raise ValueError('Unexpected reward model; record a new protocol before changing it')
    return params


def reward_components(task, forward, qa, judge, *, extraction=None,
                      code_pass_rate=None, encoder=None):
    """One candidate's one sampled continuation; average across rollouts later.

    `forward` includes the initial candidate and subsequent user/assistant turns.
    Code execution is supplied by the validated, isolated benchmark harness.
    """
    validate_payload(judge, 'judge', task)
    if not forward or forward[0]['role'] != 'assistant':
        raise ValueError('Reward must include the original candidate assistant turn')
    if any(m['role'] not in ('user', 'assistant') for m in forward):
        raise ValueError('Forward conversation must exclude system/history messages')
    if task == 'math-hard':
        task_score = float(judge['accuracy']['score'])
    elif task == 'bigcodebench':
        if code_pass_rate not in (0, 1):
            raise ValueError('Missing verified code pass/fail result')
        task_score = float(code_pass_rate)
    elif task == 'medium':
        from nltk.translate.bleu_score import sentence_bleu
        validate_payload(extraction, 'extract', task)
        # Passing strings, not word-token lists, intentionally preserves the
        # archived SentenceBLEU character-level metric.
        task_score = sentence_bleu([qa[-1]['content']], extraction['final_answer'])
    else:
        raise ValueError(f'Unknown task: {task}')
    if encoder is None:
        import tiktoken
        encoder = tiktoken.get_encoding('cl100k_base')
    length = len(encoder.encode('\n'.join(m['content'] for m in forward),
                                disallowed_special=()))
    interactivity = float(judge['interactivity']['score']) / 3
    cost_weight = 1e-4 if task == 'medium' else 5e-4
    reward = interactivity + task_score - cost_weight * length
    if not all(math.isfinite(x) for x in (interactivity, task_score, reward)):
        raise ValueError('Non-finite reward')
    return dict(interactivity=interactivity, task_score=task_score,
        forward_tokens=length, cost_weight=cost_weight, reward=reward,
        protocol=VERSION)
