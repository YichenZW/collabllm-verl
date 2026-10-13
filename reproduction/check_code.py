"""BigCodeBench evaluation subprocess, run in the separate official-deps env."""
import contextlib
import io
import json
import sys
from bigcodebench.eval import untrusted_check

def main():
    data = json.load(sys.stdin)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = untrusted_check(data["answer"], data["test"], data["entry_point"],
                 max_as_limit=30*1024, max_data_limit=30*1024, max_stack_limit=10,
                 min_time_limit=1, gt_time_limit=2)
    print(json.dumps(dict(pass_rate=float(result[0] == "pass"), result=[str(x) for x in result])))

if __name__ == "__main__":
    main()
