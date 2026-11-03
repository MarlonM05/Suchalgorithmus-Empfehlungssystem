"""Shared plumbing for the experiment scripts in ``experiments/``.

Every experiment can run in two modes:

* ``quick``: few repetitions and small data, for a functional check within
  about a minute. The numbers are not meant for the paper.
* ``full``: the complete series for the paper (30 repetitions, large data).

Results go to ``results/<mode>/<experiment>.csv`` (raw values plus
environment) and figures to ``results/<mode>/<experiment>*.pdf`` by default.
Use ``--results-dir`` to select a custom output directory.
"""

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from .environment import collect_environment, read_results, throttling_warning, write_results
from .timing import TimingSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR: Path | None = None
MODES = ("quick", "full")
SEED = 20261015  # fixed and logged in every result file [mytkowicz2009]

TIMING = {
    "quick": TimingSettings(repetitions=5, warmup=1, min_block_ns=200_000, max_block_queries=20_000, count_queries=50),
    "full": TimingSettings(repetitions=30, warmup=2, min_block_ns=1_000_000, max_block_queries=100_000, count_queries=200),
}


def parse_args(description: str, configure: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    """Command line of every experiment script.

    ``configure`` can add arguments that only one experiment needs.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=MODES, default="quick", help="quick check or full series (default: quick)")
    parser.add_argument("--plot-only", action="store_true", help="only redraw the figures from the existing CSV")
    parser.add_argument("--results-dir", type=Path, help="custom output directory for CSVs and figures")
    if configure is not None:
        configure(parser)
    args = parser.parse_args()
    global RESULTS_DIR
    RESULTS_DIR = args.results_dir
    warning = throttling_warning()
    if warning and not args.plot_only:
        print(warning, flush=True)
    return args


def result_path(experiment: str, mode: str, suffix: str = ".csv") -> Path:
    output_dir = RESULTS_DIR if RESULTS_DIR is not None else RESULTS_ROOT / mode
    return output_dir / f"{experiment}{suffix}"


def save_results(frame: pd.DataFrame, experiment: str, mode: str, settings: TimingSettings | None = None) -> Path:
    """Write raw results together with environment, mode, seed and timing settings.

    ``settings`` defaults to the standard timing of the mode; experiments with
    their own timing (e.g. the grid) pass theirs, so the file logs what was used.
    """
    settings = settings or TIMING[mode]
    metadata = {
        "experiment": experiment,
        "mode": mode,
        "seed": str(SEED),
        **{f"timing_{key}": str(value) for key, value in vars(settings).items()},
        **collect_environment(),
    }
    path = result_path(experiment, mode)
    write_results(frame, path, metadata)
    return path


def run_or_load(experiment: str, mode: str, plot_only: bool, measure) -> pd.DataFrame:
    """Measure (``measure()`` returns the raw table) and save, or load existing results."""
    if not plot_only:
        save_results(measure(), experiment, mode)
    return load_results(experiment, mode)


def load_results(experiment: str, mode: str) -> pd.DataFrame:
    frame, _ = read_results(result_path(experiment, mode))
    return frame
