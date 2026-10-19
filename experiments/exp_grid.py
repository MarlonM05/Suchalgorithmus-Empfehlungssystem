"""Experiment: measurement grid for training and evaluating the recommender (M4).

Measures all cost components (preprocessing, cost per query of every search
algorithm, feature analysis) for every combination of size, distribution,
order, duplicate share and storage structure, each with three access
patterns. See recommender/benchmark/grid.py.

Outputs (results/<mode>/): grid_features.csv, grid_components.csv, grid_sampling.csv
"""

import itertools

from recommender.benchmark.experiment import SEED, parse_args, save_results
from recommender.benchmark.grid import measure_grid
from recommender.benchmark.timing import TimingSettings
from recommender.data import DISTRIBUTIONS, ORDERS, STRUCTURES, DatasetConfig

GRID = {
    "quick": {"n": [10, 1_000, 100_000], "distribution": ["uniform", "lognormal"], "duplicate_share": [0.0, 0.5]},
    "full": {"n": [10, 100, 1_000, 10_000, 100_000, 1_000_000], "distribution": list(DISTRIBUTIONS), "duplicate_share": [0.0, 0.1, 0.5]},
}
# The grid trades precision per point for many points: the best strategy is
# usually ahead by a wide margin, so 10 repetitions suffice (docs/annahmen.md A25).
TIMING = {
    "quick": TimingSettings(repetitions=3, warmup=1, min_block_ns=100_000, max_block_queries=5_000),
    "full": TimingSettings(repetitions=10, warmup=1, min_block_ns=500_000, max_block_queries=50_000),
}


def configurations(mode: str) -> list[DatasetConfig]:
    grid = GRID[mode]
    combinations = itertools.product(grid["n"], grid["distribution"], ORDERS, grid["duplicate_share"], STRUCTURES)
    return [
        DatasetConfig(n=n, distribution=d, order=o, duplicate_share=dup, structure=s, seed=SEED + i)
        for i, (n, d, o, dup, s) in enumerate(combinations)
    ]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    if args.plot_only:
        return
    features, components, sampling = measure_grid(configurations(args.mode), TIMING[args.mode], seed=SEED)
    for name, frame in [("grid_features", features), ("grid_components", components), ("grid_sampling", sampling)]:
        save_results(frame, name, args.mode, TIMING[args.mode])
    print(f"done: results/{args.mode}/grid_*  ({len(features)} datasets)")


if __name__ == "__main__":
    main()
