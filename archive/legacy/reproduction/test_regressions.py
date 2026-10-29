from copy import deepcopy
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.modules.user_simulator import UserSimulator
from collabllm.metric import SingleTurnOrChatMetric
from examples.metrics.bleu import BLEUMetric

def rows():
    return [dict(prompt=[dict(role="user", content="question")], completion="answer",
        conv_id=0, score=1, single_turn_prompt="question", single_turn_completion="answer",
        single_turn_metadata={})]

def test_load_local_dataset_dict(tmp_path):
    DatasetDict(train=Dataset.from_list(rows())).save_to_disk(tmp_path)
    loaded = MultiturnDataset(str(tmp_path))
    assert len(loaded.to_sft_dataset()["train"]) == 1

def test_input_conversion_does_not_mutate_or_duplicate_system_messages():
    data = MultiturnDataset(rows())
    original = deepcopy(data.data)
    first = data.to_inputs_dataset()["train"].to_list()
    second = data.to_inputs_dataset()["train"].to_list()
    assert first == second
    assert data.data == original
    assert len([m for m in first[0]["prompt"] if m["role"] == "system"]) == 1

def test_invalid_user_json_raises_clear_error(monkeypatch):
    monkeypatch.setattr("litellm.completion", lambda **kw: SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"wrong": true}'))]))
    with pytest.raises(RuntimeError, match="valid response"):
        UserSimulator(model="test", num_retries=2)([])

def test_extraction_accepts_no_metadata(monkeypatch):
    monkeypatch.setattr("litellm.completion", lambda **kw: SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"thought":"done","final_completion":"abc"}'))]))
    metric = SingleTurnOrChatMetric("document->bleu", model="test")
    assert metric._extract_final_completion([dict(role="assistant", content="abc")], None) == "abc"

def test_paper_character_bleu_is_preserved():
    assert BLEUMetric().score("", "abcd", "abcd") == 1
