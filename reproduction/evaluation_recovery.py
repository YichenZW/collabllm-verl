"""Keep invalid responses retryable and timed-out candidates in the denominator."""
import subprocess

from collabllm.utils.extract_json_reliable import extract_json
from reproduction.response_schema import validate_payload


def parse_response(text, kind, dataset):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{kind}: API returned no text")
    return validate_payload(extract_json(text), kind, dataset)


def check_candidate(check, answer, harness):
    # Caller must first validate the canonical answer. Infrastructure errors and
    # canonical timeouts remain fatal; only this generated candidate is graded.
    try:
        return check(answer)
    except subprocess.TimeoutExpired as exc:
        # run_check already killed/reaped this candidate's isolated process group.
        return dict(pass_rate=0.0, result=["fail", f"outer_timeout:{exc.timeout}"],
                    harness=harness, outer_timeout_seconds=exc.timeout)
