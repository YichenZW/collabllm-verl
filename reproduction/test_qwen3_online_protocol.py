"""Compare online reward components against actual archived function bodies."""
import ast
import copy
import unittest
from types import SimpleNamespace

from reproduction.common import LEGACY, TASKS, chat_text, legacy_prompt
from reproduction.qwen3_online_protocol import (
    paired_row_indices, request_params, reward_components,
)


def archived_method(relative, class_name, name, namespace):
    tree = ast.parse((LEGACY / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LEGACY / relative), 'exec'), namespace)
    return namespace[name]


class OnlineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.qa = [{'role': 'user', 'content': 'Write about a cat.'},
                   {'role': 'assistant', 'content': 'The small cat sat on the mat.'}]
        self.history = [{'role': 'system', 'content': 'Assistant system.'},
                        {'role': 'user', 'content': 'Can you help me?'}]
        self.forward = [{'role': 'assistant', 'content': 'What should I describe?'},
                        {'role': 'user', 'content': 'A cat on a mat.'},
                        {'role': 'assistant', 'content': 'A small cat sat on the mat.'}]

    def test_math_rescaling_matches_archived_judge_and_prompt(self):
        recorded = []
        raw = {'interactivity': {'score': 2}, 'accuracy': {'score': 1}}

        def api(prompt, **kwargs):
            recorded.append(prompt)
            return copy.deepcopy(raw)

        prompt = legacy_prompt('llm_judge/question-answering.txt')

        class Handler:
            def __call__(self, **kw):
                return prompt.format(**kw)

            def check_format(self, value):
                return True

        oracle = archived_method('collabllm/metrics/llm_judge.py', 'LLMJudge', '__call__',
            dict(get_llm_output=api, chat_template=chat_text, print=lambda *a: None))
        judge = SimpleNamespace(prompt_handler=Handler(), llm_kwargs={}, rescale_func=lambda x: x / 3)
        expected = oracle(judge, self.qa, self.forward, self.history)
        result = reward_components('math-hard', self.forward, self.qa, raw)
        self.assertEqual(result['interactivity'], expected['interactivity']['score'])
        self.assertEqual(result['task_score'], expected['accuracy']['score'])
        params = request_params('math-hard', 'judge', self.qa, self.history,
                                self.forward, 'claude-sonnet-4-5-20250929')
        self.assertEqual(params['messages'][0]['content'],
                         'You are a helpful assistant designed to output in JSON format.' + recorded[0])
        self.assertEqual(params['temperature'], 0.2)
        self.assertNotIn('top_p', params)

    def test_writing_bleu_matches_archived_class(self):
        from nltk.translate.bleu_score import sentence_bleu
        oracle = archived_method('collabllm/metrics/bleu_score.py', 'SentenceBLEU', '__call__',
                                 dict(sentence_bleu=sentence_bleu))
        answer = self.forward[-1]['content']
        expected = oracle(SimpleNamespace(), self.qa, self.forward, final_answer=answer)['bleu_score']
        result = reward_components('medium', self.forward, self.qa,
            {'interactivity': {'score': 3}}, extraction={'final_answer': answer})
        self.assertEqual(result['task_score'], expected)
        self.assertEqual(result['cost_weight'], 1e-4)

    def test_token_penalty_includes_candidate_and_simulated_user(self):
        import tiktoken
        encoder = tiktoken.get_encoding('cl100k_base')
        expected = len(encoder.encode('\n'.join(x['content'] for x in self.forward)))
        result = reward_components('bigcodebench', self.forward, self.qa,
            {'interactivity': {'score': 1}}, code_pass_rate=1)
        self.assertEqual(result['forward_tokens'], expected)
        self.assertAlmostEqual(result['reward'], 1 / 3 + 1 - 5e-4 * expected)
        with self.assertRaises(ValueError):
            reward_components('bigcodebench', self.forward, self.qa,
                              {'interactivity': {'score': 1}})

    def test_multi_problem_pair_mapping(self):
        # Concrete two-problem ordering produced by tensor.repeat(2, 1).
        prompts = ['problem A', 'problem B']
        responses = ['answer A0', 'answer B0', 'answer A1', 'answer B1']
        pairs = [(prompts[i], r) for i, r in zip(paired_row_indices(2), responses)]
        self.assertEqual(pairs, [('problem A', 'answer A0'), ('problem B', 'answer B0'),
                                 ('problem A', 'answer A1'), ('problem B', 'answer B1')])

    def test_extraction_gets_history_and_full_forward_conversation(self):
        params = request_params('medium', 'extract', self.qa, self.history,
                                self.forward, 'claude-sonnet-4-5-20250929')
        text = params['messages'][0]['content']
        self.assertIn('Can you help me?', text)
        self.assertIn('What should I describe?', text)
        self.assertIn('A cat on a mat.', text)
        self.assertNotIn('Assistant system.', text)


if __name__ == '__main__':
    unittest.main()
