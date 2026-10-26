import subprocess

import pytest

from reproduction.evaluation_recovery import check_candidate, parse_response


def test_absent_api_text_is_retryable_validation_failure():
    for text in [None, "", " "]:
        with pytest.raises(ValueError, match="no text"):
            parse_response(text, "user", "medium")
    assert parse_response('{"response":"Continue"}', "user", "medium") == {"response":"Continue"}


def test_candidate_timeout_counts_as_failure_without_hiding_infrastructure_errors():
    def timed_out(answer):
        raise subprocess.TimeoutExpired("checker", 300)
    result = check_candidate(timed_out, "while True: pass", {"version":"unchanged"})
    assert result["pass_rate"] == 0 and result["outer_timeout_seconds"] == 300
    assert result["harness"] == {"version":"unchanged"}
    def broken_environment(answer):
        raise subprocess.CalledProcessError(1, "checker")
    with pytest.raises(subprocess.CalledProcessError):
        check_candidate(broken_environment, "answer", {})
