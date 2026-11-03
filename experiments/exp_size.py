"""Experiment: runtime and comparisons as a function of the data size n.

Feature: Datengröße. Setting: uniformly distributed, sorted, no duplicates,
only hits with uniformly distributed positions; all array algorithms.

Outputs (results/<mode>/):
    size.csv                  raw measurements
    size_runtime.pdf          ns per query over n (median, 95 % CI)
    size_comparisons.pdf      comparisons per query over n
    size_speedup.csv          speedup vs. binary search per n, plus geometric mean
"""

import pandas as pd

from recommender.benchmark import Configuration, geometric_mean, run_configurations, speedups, summarize
from recommender.benchmark.experiment import SEED, TIMING, load_results, parse_args, result_path, save_results
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import plot_over_size

NAME = "size"
SIZES = {
    "quick": [10, 100, 1_000, 10_000, 100_000],
    "full": [10, 16, 30, 32, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000],
}
QUERY_POOL = 10_000


def configurations(mode: str) -> list[Configuration]:
    return [
        Configuration(
            DatasetConfig(n=n, distribution="uniform", order="sorted", seed=SEED + i),
            QueryConfig(count=QUERY_POOL, hit_ratio=1.0, position="uniform", seed=SEED + i),
        )
        for i, n in enumerate(SIZES[mode])
    ]


def plot(mode: str) -> None:
    raw = load_results(NAME, mode)
    summary = summarize(raw, ["n", "algorithm"])
    plot_over_size(summary, "median", "Laufzeit pro Anfrage [ns]", "Laufzeit in Abhängigkeit von der Datengröße", result_path(NAME + "_runtime", mode, ".pdf"))

    counts = raw.groupby(["n", "algorithm"], as_index=False)["comparisons"].first()
    plot_over_size(counts, "comparisons", "Vergleiche pro Anfrage", "Schlüsselvergleiche in Abhängigkeit von der Datengröße", result_path(NAME + "_comparisons", mode, ".pdf"), band=False)

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
    print(f"done: {result_path(NAME, args.mode).parent / f'{NAME}*'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
