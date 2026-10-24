"""Versioned, bounded subprocess execution for canonical and generated code."""
import functools
import hashlib
import json
import os
import signal
import subprocess

from reproduction.common import ROOT


@functools.lru_cache(maxsize=1)
def harness_identity():
    return {name: hashlib.sha256((ROOT / "reproduction" / name).read_bytes()).hexdigest()
            for name in ["check_code.py", "code_executor.py", "code_network.py", "code_runtime.py"]}


def run_check(data, env, timeout=300):
    process = subprocess.Popen([str(ROOT / ".venv-code/bin/python"),
        str(ROOT / "reproduction/check_code.py")], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(json.dumps(data), timeout=timeout)
    except BaseException:
        # Kill only this check's isolated process group, including Pool workers
        # holding its pipes open. Never signal the Slurm job's shared group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
    result = json.loads(stdout)
    if result.get("harness") != harness_identity():
        raise RuntimeError("Code checker changed during this process; resume with a consistent version")
    return result
