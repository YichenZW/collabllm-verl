"""Validate the fields consumed from historical simulator and judge replies."""


def validate_payload(value, kind, dataset):
    if not isinstance(value, dict):
        raise ValueError(f"{kind}: expected a JSON object")
    if kind in ("user", "extract"):
        key = "response" if kind == "user" else "final_answer"
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"{kind}: {key} must be a nonempty string")
    elif kind == "judge":
        try:
            score = float(value["interactivity"]["score"])
            if score not in (1, 2, 3):
                raise ValueError("interactivity score must be 1, 2, or 3")
            if dataset == "math-hard" and float(value["accuracy"]["score"]) not in (0, 1):
                raise ValueError("accuracy score must be 0 or 1")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"judge: missing or invalid score ({exc})") from exc
    else:
        raise ValueError(f"Unknown response kind: {kind}")
    return value
