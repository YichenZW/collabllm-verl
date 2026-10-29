import os
import subprocess
import sys

from reproduction.code_runtime import is_descendant


def test_process_tree_check_rejects_parent_and_tracks_live_child():
    root = os.getpid()
    assert is_descendant(root, root)
    assert not is_descendant(os.getppid(), root)
    assert not is_descendant(-1, root)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert is_descendant(child.pid, root)
    finally:
        child.terminate()
        child.wait(timeout=5)
    assert not is_descendant(child.pid, root)
