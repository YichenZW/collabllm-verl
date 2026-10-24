"""A reviewed reference bug cannot mask a new environment/data failure."""
import json
import unittest
from unittest.mock import patch

from reproduction.common import digest
from reproduction.code_reference_policy import reviewed_reference_issue


class ReferencePolicyTest(unittest.TestCase):
    def test_only_exact_reviewed_failure_is_tolerated(self):
        metadata = dict(task_id="Example/1", test="unchanged", answer="unchanged")
        issues = {"Example/1": dict(metadata_digest=digest(metadata),
            failing_tests=["test_constant"],
            required_trace_fragments=["AssertionError: True is not false"])}
        expected = dict(pass_rate=0.0, result=["fail", repr(
            {"test_constant": "Traceback\nAssertionError: True is not false\n"})])
        with patch("pathlib.Path.read_text", return_value=json.dumps(issues)):
            self.assertIsNotNone(reviewed_reference_issue(metadata, expected))
            self.assertIsNone(reviewed_reference_issue(dict(metadata, test="changed"), expected))
            for failures in [
                {"test_constant": "ModuleNotFoundError: scipy"},
                {"test_constant": "AssertionError: True is not false", "other": "Timeout"},
            ]:
                self.assertIsNone(reviewed_reference_issue(metadata,
                    dict(pass_rate=0.0, result=["fail", repr(failures)])))
            self.assertIsNone(reviewed_reference_issue(metadata,
                dict(pass_rate=0.0, result=["timeout", expected["result"][1]])))
            self.assertIsNone(reviewed_reference_issue(metadata,
                dict(pass_rate=1.0, result=["pass", "{}"])))


if __name__ == "__main__":
    unittest.main()
