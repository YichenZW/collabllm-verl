"""Validate the fields consumed from historical simulator and judge replies."""


def validate_payload(value, kind, dataset):
    if not isinstance(value, dict):
        raise ValueError(f"{kind}: expected a JSON object")
    if kind == "user":
        if not isinstance(value.get("response"), str) or not value["response"].strip():
            raise ValueError("user: response must be a nonempty string")
    elif kind == "extract":
        # An empty extracted answer is a valid model outcome (score zero), not
        # a reason to pay for another extraction or exclude the sample.
        if not isinstance(value.get("final_answer"), str):
            raise ValueError("extract: final_answer must be a string")
    elif kind == "judge":
        try:
            score = float(value["interactivity"]["score"])
            if not 1 <= score <= 3:
                raise ValueError("interactivity score must be between 1 and 3")
            if dataset == "math-hard" and float(value["accuracy"]["score"]) not in (0, 1):
                raise ValueError("accuracy score must be 0 or 1")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"judge: missing or invalid score ({exc})") from exc
    else:
        raise ValueError(f"Unknown response kind: {kind}")
    return value
