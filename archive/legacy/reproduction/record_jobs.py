"""Refresh states of this experiment's known Slurm jobs only."""
import subprocess
import re
import shlex
from pathlib import Path
from reproduction.common import RUNS, atomic_json

IDS = [2147616, 2147617, 2147629, 2147639, 2147664, 2147667,
       2147668, 2147669, 2147670, 2147676, 2147683, 2147693, 2147694, 2147695, 2147702, 2147703,
       2147728, 2147729, 2147730, 2147731, 2147738, 2147766, 2147780,
       2147782, 2147783, 2147784, 2147791, 2147792, 2147798, 2147799, 2147800, 2147801,
       2147828, 2147921, 2149161, 2149162, 2149236, 2149237, 2149375]
def main():
    fields = ["JobID", "JobName", "State", "ExitCode", "Elapsed", "NodeList", "AllocTRES", "Start", "End",
              "WorkDir", "StdOut", "SubmitLine"]
    proc = subprocess.run(["sacct", "-X", "-j", ",".join(map(str, IDS)), "--parsable2", "--noheader",
                            "--format=" + ",".join(fields)], capture_output=True, text=True, check=True)
    # SubmitLine is last: constraint expressions can themselves contain '|'.
    jobs = [dict(zip(fields, line.split("|", len(fields) - 1))) for line in proc.stdout.splitlines()]
    for row in jobs:
        args = shlex.split(row["SubmitLine"])
        output = row["StdOut"] or None
        for i, arg in enumerate(args):
            if output:
                break
            if arg.startswith("--output="):
                output = arg.split("=", 1)[1]
            elif arg in ("--output", "-o") and i + 1 < len(args):
                output = args[i + 1]
        if output:
            for key, value in [("%j", row["JobID"]), ("%x", row["JobName"]), ("%N", row["NodeList"])]:
                output = output.replace(key, value)
            if not Path(output).is_absolute():
                output = str(Path(row["WorkDir"]) / output)
        row["log_path"] = output
        row["gpu_names"] = []
        if row["log_path"] and Path(row["log_path"]).is_file():
            with Path(row["log_path"]).open(errors="replace") as log:
                header = log.read(16384)
            row["gpu_names"] = sorted(set(re.findall(r"^\d+,\s*(NVIDIA[^,\n]+),", header, re.M)))
    atomic_json(RUNS / "slurm_jobs.json", jobs)
    for row in jobs:
        print(row["JobID"], row["JobName"], row["State"], row["ExitCode"], row["Elapsed"])

if __name__ == "__main__":
    main()
