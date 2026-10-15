"""Run all experiments in one go.

    python experiments/run_all.py --mode quick       # functional check, about 1-2 minutes
    python experiments/run_all.py --mode full        # complete series for the paper (long)
    python experiments/run_all.py --plot-only        # only redraw all figures from the CSVs

For the full series: plug in the laptop, close other programs and do not use
the machine during the run, so other load does not distort the measurements.
"""

import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    "exp_size.py",
    "exp_distribution.py",
    "exp_duplicates.py",
    "exp_access_pattern.py",
    "exp_structure.py",
    "exp_amortization.py",
    "exp_analysis_cost.py",
]


def main() -> None:
    arguments = sys.argv[1:]
    here = Path(__file__).parent
    for script in EXPERIMENTS:
        start = time.perf_counter()
        print(f"=== {script} {' '.join(arguments)}", flush=True)
        subprocess.run([sys.executable, str(here / script), *arguments], check=True)
        print(f"=== {script} finished in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
