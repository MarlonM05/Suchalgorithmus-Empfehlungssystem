"""Experiment: data size n with several data sets per size.

Same setting as exp_size.py (uniformly distributed, sorted, no duplicates,
only hits with uniformly distributed positions; all array algorithms), but
each size is measured on DATASETS different data sets instead of one. With a
single data set, the runtime of the binary search is not monotonic in n
(e.g. n = 300: 10.9 ns, n = 1000: 7.9 ns), because one particular data set
and the branch predictor interact [khuong2017]. The median over several data
sets shows whether this is a property of the algorithm or of the data set.

Data set 0 of every size uses the same seed as exp_size.py, so its values are
directly comparable with size.csv.

Outputs (results/<mode>/):
    size_datasets.csv             raw measurements
    size_datasets_summary.csv     per n and algorithm: median over the data sets
                                  of the per-data-set medians, plus min and max
    size_datasets_runtime.pdf     ns per query over n (median over data sets)
    size_datasets_speedup.csv     speedup vs. binary search per n, plus geometric mean
"""

import pandas as pd

from recommender.benchmark import Configuration, geometric_mean, run_configurations, speedups, summarize
from recommender.benchmark.experiment import SEED, TIMING, load_results, parse_args, result_path, save_results
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import plot_over_size

from exp_size import QUERY_POOL, SIZES

NAME = "size_datasets"
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


def summarize_over_datasets(raw: pd.DataFrame) -> pd.DataFrame:
    """Median per data set first, then median, min and max over the data sets."""
    per_dataset = summarize(raw, ["n", "dataset_seed", "algorithm"])
    return (
        per_dataset.groupby(["n", "algorithm"], as_index=False)["median"]
        .agg(median="median", dataset_min="min", dataset_max="max", datasets="count")
    )


def plot(mode: str) -> None:
    raw = load_results(NAME, mode)
    summary = summarize_over_datasets(raw)
    summary.to_csv(result_path(NAME + "_summary", mode), index=False)
    plot_over_size(
        summary,
        "median",
        "Laufzeit pro Anfrage [ns]",
        f"Laufzeit in Abhängigkeit von der Datengröße (Median über {DATASETS} Datensätze)",
        result_path(NAME + "_runtime", mode, ".pdf"),
        band=False,
    )

    relative = speedups(summary, ["n"], baseline="binary")
    table = relative.pivot(index="n", columns="algorithm", values="speedup")
    table.loc["geometrisches Mittel"] = [geometric_mean(table[c]) for c in table.columns]
    table.to_csv(result_path(NAME + "_speedup", mode))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    if not args.plot_only:
        frame = run_configurations(configurations(args.mode), TIMING[args.mode], seed=SEED)
        save_results(frame, NAME, args.mode)
    plot(args.mode)
    print(f"done: results/{args.mode}/{NAME}*")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
