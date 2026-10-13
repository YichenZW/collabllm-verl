"""Refresh states of this experiment's known Slurm jobs only."""
import subprocess
from reproduction.common import RUNS, atomic_json

IDS = [2147616, 2147617, 2147629, 2147639, 2147664, 2147667,
       2147668, 2147669, 2147670, 2147676, 2147683, 2147693, 2147694, 2147695, 2147702, 2147703,
       2147728, 2147729, 2147730, 2147731, 2147738, 2147766, 2147780,
       2147782, 2147783, 2147784, 2147791, 2147792, 2147798, 2147799, 2147800, 2147801,
       2147828]
def main():
    fields = ["JobID", "JobName", "State", "ExitCode", "Elapsed", "NodeList", "AllocTRES", "Start", "End", "SubmitLine"]
    proc = subprocess.run(["sacct", "-X", "-j", ",".join(map(str, IDS)), "--parsable2", "--noheader",
                            "--format=" + ",".join(fields)], capture_output=True, text=True, check=True)
    jobs = [dict(zip(fields, line.split("|"))) for line in proc.stdout.splitlines()]
    atomic_json(RUNS / "slurm_jobs.json", jobs)
    for row in jobs:
        print(row["JobID"], row["JobName"], row["State"], row["ExitCode"], row["Elapsed"])

if __name__ == "__main__":
    main()
