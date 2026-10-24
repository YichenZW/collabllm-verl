"""Keep benchmark child-process cleanup within the test's process tree."""
import contextlib
import os
from pathlib import Path


def is_descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            # comm can contain spaces or parentheses; fields after its final ')'
            # begin with state, then the parent PID.
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def install_execution_compat():
    import bigcodebench.eval as evaluator

    if getattr(evaluator.safe_environment, "_owned_children_compat", False):
        return os.cpu_count()
    original_environment = evaluator.safe_environment

    @contextlib.contextmanager
    def environment():
        root_pid = os.getpid()
        original_kill = os.kill
        with original_environment():
            guarded_kill = os.kill
            def kill_owned(pid, sig):
                # The upstream guard registers subprocess.Popen children but
                # misses multiprocessing.Pool forks, preventing Pool.terminate.
                if pid > 0 and is_descendant(pid, root_pid):
                    return original_kill(pid, sig)
                return guarded_kill(pid, sig)
            os.kill = kill_owned
            try:
                yield
            finally:
                os.kill = guarded_kill
    environment._owned_children_compat = True
    evaluator.safe_environment = environment
    # Pool() in Python 3.10 otherwise uses the whole node's CPU count, even
    # inside a small Slurm allocation. Two test workers fit our CPU allocations.
    cpus = min(2, len(os.sched_getaffinity(0)))
    os.cpu_count = lambda: cpus
    return cpus
