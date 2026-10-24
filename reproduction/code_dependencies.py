"""Private compatibility dependency for the archived librosa evaluator.

Install once with:
uv pip install --python .venv-code/bin/python --target "$COLLABLLM_RUNS/code_python_overlay" --no-deps setuptools==80.9.0

The shared code environment is never modified. Package metadata and the actual
pkg_resources implementation are included in every checker identity.
"""
import hashlib
import sys

from reproduction.common import RUNS

OVERLAY = RUNS / "code_python_overlay"


def overlay_identity():
    metadata = OVERLAY / "setuptools-80.9.0.dist-info"
    if not metadata.is_dir():
        raise RuntimeError("Install the private setuptools==80.9.0 code overlay first")
    return {str(path.relative_to(OVERLAY)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [metadata / "METADATA", metadata / "RECORD",
                         OVERLAY / "pkg_resources/__init__.py"]}


def activate_overlay():
    overlay_identity()  # Fail before executing any candidate if incomplete.
    sys.path.insert(0, str(OVERLAY))
