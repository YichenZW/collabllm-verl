import pytest

from reproduction.response_schema import validate_payload


@pytest.mark.parametrize("payload,kind,dataset", [
    ({"thought": "planning"}, "user", "medium"),
    ({"response": None}, "user", "medium"),
    ({"final_answer": []}, "extract", "bigcodebench"),
    ({"interactivity": {"score": 2}}, "judge", "math-hard"),
    ({"interactivity": {"score": 4}}, "judge", "medium"),
])
def test_malformed_provider_response_is_rejected(payload, kind, dataset):
    with pytest.raises(ValueError):
        validate_payload(payload, kind, dataset)


def test_historical_response_fields_are_preserved():
    payload = {"thought": "done", "response": "00123"}
    assert validate_payload(payload, "user", "medium") is payload
    judge = {"interactivity": {"score": "2"}, "accuracy": {"score": 1}}
    assert validate_payload(judge, "judge", "math-hard") is judge
