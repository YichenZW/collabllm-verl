"""BigCodeBench evaluation subprocess, run in the separate official-deps env."""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.code_dependencies import activate_overlay
activate_overlay()
from bigcodebench.eval import untrusted_check
from reproduction.code_executor import harness_identity
from reproduction.code_network import configure_fixture_http
from reproduction.code_runtime import install_execution_compat

def main():
    data = json.load(sys.stdin)
    network_profile = configure_fixture_http(data["test"])
    cpu_limit = install_execution_compat()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = untrusted_check(data["answer"], data["test"], data["entry_point"],
                 max_as_limit=30*1024, max_data_limit=30*1024, max_stack_limit=10,
                 min_time_limit=1, gt_time_limit=2)
    print(json.dumps(dict(pass_rate=float(result[0] == "pass"), result=[str(x) for x in result],
                          harness=harness_identity(), network_profile=network_profile,
                          multiprocessing_cpu_limit=cpu_limit)))

if __name__ == "__main__":
    main()
