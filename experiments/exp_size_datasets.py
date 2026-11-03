"""Experiment: data size n with several data sets per size.

Same setting as exp_size.py (uniformly distributed, sorted, no duplicates,
only hits with uniformly distributed positions; all array algorithms), but
each size is measured on DATASETS different data sets instead of one. This
checks whether the results of exp_size.py depend on the particular data set
drawn for each size, or whether they are a property of the algorithm and n.

Quick run on the measurement laptop (Apple M5, AC power, low-power mode off):
between the five data sets, binary and branchless binary search differ by at
most 4 %, linear search by at most 10 %. The hash lookup varies more at
n <= 100 (up to a factor of 6), because its times there are around 1 ns,
close to the loop overhead of the block. The non-monotonic runtime of binary
search in exp_size.py (faster at n = 1000 than at n = 100 and 300) appears
with every data set, so it is a property of the size, not of one data set
(docs/annahmen.md A40).

Data set 0 of every size uses the same seed as exp_size.py, so its values are
directly comparable with size.csv.

Every team member can run it: ``--label`` (default ``messlaptop``) goes into
the file names, so runs on different machines do not overwrite each other.
The report takes the absolute times from the measurement laptop (A18) and
uses other machines only to compare ratios.

Outputs (results/<mode>/, <label> from --label):
    size_datasets_<label>.csv             raw measurements
    size_datasets_<label>_summary.csv     per n and algorithm: median over the data sets
                                          of the per-data-set medians, plus min and max
    size_datasets_<label>_runtime.pdf     ns per query over n (median over data sets,
                                          band from the fastest to the slowest data set)
    size_datasets_<label>_speedup.csv     speedup vs. binary search per n, plus geometric mean
"""

import argparse
import platform
import re

import pandas as pd

from recommender.benchmark import Configuration, geometric_mean, run_configurations, speedups, summarize
from recommender.benchmark.experiment import SEED, TIMING, load_results, parse_args, result_path, save_results
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import plot_over_size

from exp_size import QUERY_POOL, SIZES

DEFAULT_LABEL = "messlaptop"
DATASETS = 5
DATASET_SEED_STEP = 1_000  # keeps the seeds of different sizes and data sets apart


def configurations(mode: str) -> list[Configuration]:
    return [
        Configuration(
            DatasetConfig(n=n, distribution="uniform", order="sorted", seed=SEED + i + DATASET_SEED_STEP * d),
            QueryConfig(count=QUERY_POOL, hit_ratio=1.0, position="uniform", seed=SEED + i + DATASET_SEED_STEP * d),
        )
        for i, n in enumerate(SIZES[mode])
        for d in range(DATASETS)
    ]


def result_name(label: str) -> str:
    """File name prefix; the label is reduced to letters, digits and dashes."""
    clean = re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-") or DEFAULT_LABEL
    return f"size_datasets_{clean}"


def add_label_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"name of the machine in the result file names (default: {DEFAULT_LABEL}); use another one for runs on other computers",
    )


def summarize_over_datasets(raw: pd.DataFrame) -> pd.DataFrame:
    """Median per data set first, then median, min and max over the data sets."""
    per_dataset = summarize(raw, ["n", "dataset_seed", "algorithm"])
    return (
        per_dataset.groupby(["n", "algorithm"], as_index=False)["median"]
        .agg(median="median", dataset_min="min", dataset_max="max", datasets="count")
    )


def plot(name: str, mode: str) -> None:
    raw = load_results(name, mode)
    summary = summarize_over_datasets(raw)
    summary.to_csv(result_path(name + "_summary", mode), index=False)
    plot_over_size(
        summary,
        "median",
        "Laufzeit pro Anfrage [ns]",
        f"Median über {DATASETS} Datensätze, Bereich: schnellster bis langsamster Datensatz",
        result_path(name + "_runtime", mode, ".pdf"),
        band_columns=("dataset_min", "dataset_max"),
    )

    relative = speedups(summary, ["n"], baseline="binary")
    table = relative.pivot(index="n", columns="algorithm", values="speedup")
    table.loc["geometrisches Mittel"] = [geometric_mean(table[c]) for c in table.columns]
    table.to_csv(result_path(name + "_speedup", mode))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0], configure=add_label_argument)
    name = result_name(args.label)
    if platform.system() != "Darwin" and not args.plot_only:
        # The power check in environment.py only works on macOS.
        print("HINWEIS: Netzteil anschließen und den Energiemodus auf höchste Leistung stellen, sonst läuft die CPU gedrosselt.", flush=True)
    if not args.plot_only:
        frame = run_configurations(configurations(args.mode), TIMING[args.mode], seed=SEED)
        save_results(frame, name, args.mode)
    plot(name, args.mode)
    print(f"done: {result_path(name, args.mode).parent / f'{name}*'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
