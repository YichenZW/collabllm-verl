"""The live source can only be stopped after exact-job and checkpoint checks."""
import json
from types import SimpleNamespace

import pytest

from reproduction import qwen3_handoff as handoff


def test_accounting_requires_current_job_owner(monkeypatch):
    monkeypatch.setattr(handoff.os, "getuid", lambda: 123)
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kw:
                        SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id specified"))
    monkeypatch.setattr(handoff.subprocess, "check_output", lambda *args, **kw: "101|RUNNING|456\n")
    with pytest.raises(RuntimeError, match="another user"):
        handoff.job_state(101)


def test_controller_state_prevents_stale_accounting_at_startup(monkeypatch):
    monkeypatch.setattr(handoff.os, "getuid", lambda: 123)
    monkeypatch.setattr(handoff.subprocess, "check_output", lambda *args, **kw: "101|PENDING|123\n")
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kw:
                        SimpleNamespace(returncode=0, stdout="JobId=101 UserId=owner(123) JobState=RUNNING", stderr=""))
    assert handoff.job_state(101) == "RUNNING"
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kw:
                        SimpleNamespace(returncode=0, stdout="JobId=101 UserId=other(456) JobState=RUNNING", stderr=""))
    with pytest.raises(RuntimeError, match="another user"):
        handoff.job_state(101)


def test_controller_failure_is_not_misread_as_archived_job(monkeypatch):
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kw:
                        SimpleNamespace(returncode=1, stdout="", stderr="Unable to contact slurm controller"))
    with pytest.raises(RuntimeError, match="Cannot query"):
        handoff.job_state(101)


def setup_request(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "RUNS", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    directory = tmp_path / "handoffs"
    directory.mkdir()
    request = directory / "math-hard-202.request.json"
    request.write_text(json.dumps(dict(task="math-hard", source_job=101, target_job=202, status="requested")))
    monkeypatch.setattr(handoff, "source_identity", lambda *args: {})
    return request


def test_handoff_fails_closed_before_any_cancel(tmp_path, monkeypatch):
    request = setup_request(tmp_path, monkeypatch)
    cancellations = []
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kw: cancellations.append(args))
    monkeypatch.setattr(handoff, "job_state", lambda job: "RUNNING")
    monkeypatch.setattr(handoff, "probe_gpu", lambda: {"name": "test"})
    plan = lambda directory: ({"max_steps": -1, "num_train_epochs": 8}, directory, {"global_step": 33})
    def invalid_checkpoint(*args):
        raise RuntimeError("Invalid optimizer file")
    monkeypatch.setattr(handoff, "validate_checkpoint", invalid_checkpoint)
    with pytest.raises(RuntimeError, match="optimizer"):
        handoff.handoff_source("math-hard", 101, tmp_path, plan)
    assert not cancellations
    data = json.loads(request.read_text());data["source_job"] = 999;request.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="exact jobs"):
        handoff.handoff_source("math-hard", 101, tmp_path, plan)
    assert not cancellations


def test_gpu_failure_never_stops_source_and_valid_handoff_waits_for_exit(tmp_path, monkeypatch):
    setup_request(tmp_path, monkeypatch)
    cancellations = []
    def cancel(command, **kwargs):
        cancellations.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(handoff.subprocess, "run", cancel)
    monkeypatch.setattr(handoff, "job_state", lambda job: "RUNNING")
    plan = lambda directory: ({"max_steps": -1, "num_train_epochs": 8}, directory, {"global_step": 33})
    def failed_gpu():
        raise RuntimeError("GPU unavailable")
    monkeypatch.setattr(handoff, "probe_gpu", failed_gpu)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        handoff.handoff_source("math-hard", 101, tmp_path, plan)
    assert not cancellations
    monkeypatch.setattr(handoff, "probe_gpu", lambda: {"name": "test"})
    monkeypatch.setattr(handoff, "validate_checkpoint", lambda *args: {"global_step": 33})
    monkeypatch.setattr(handoff, "job_state", lambda job: "CANCELLED" if cancellations and job == 101 else "RUNNING")
    assert handoff.handoff_source("math-hard", 101, tmp_path, plan) == "CANCELLED"
    assert cancellations == [["scancel", "101"]]
    result = json.loads((tmp_path / "handoffs/math-hard-202.json").read_text())
    assert result["status"] == "source_terminal"
