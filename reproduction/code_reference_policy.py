"""Recognize audited reference bugs without changing candidate test scoring."""
import ast
import json

from reproduction.common import ROOT, digest


def reviewed_reference_issue(metadata, result):
    """Only an exact dataset record and known assertion failures are tolerated.

    A tolerated reference retains pass_rate=0. Candidate answers must still
    pass the original tests. New import, timeout, or assertion errors stop eval.
    """
    issues = json.loads((ROOT / "reproduction/code_reference_issues.json").read_text())
    issue = issues.get(metadata["task_id"])
    if not issue or digest(metadata) != issue["metadata_digest"] or result["pass_rate"]:
        return None
    if result.get("result", [None])[0] != "fail":
        return None
    try:
        failures = ast.literal_eval(result["result"][1])
    except (ValueError, SyntaxError, IndexError, TypeError):
        return None
    if not isinstance(failures, dict) or set(failures) != set(issue["failing_tests"]):
        return None
    if not all(isinstance(trace, str) and all(fragment in trace for fragment in
               issue["required_trace_fragments"]) for trace in failures.values()):
        return None
    return dict(task_id=metadata["task_id"], **issue)
