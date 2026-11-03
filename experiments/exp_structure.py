"""Experiment: array (random access) versus linked list (sequential access only).

Feature: Speicherstruktur. Setting: uniformly distributed, sorted, no
duplicates, only hits; linear and jump search on both structures, binary search
on the array as reference. Expected: on the list every search costs O(n) node
visits; jump search saves comparisons but not node visits [shneiderman1978].

Outputs (results/<mode>/):
    structure.csv, structure_runtime.pdf, structure_accesses.pdf
"""

import pandas as pd

from recommender.benchmark import Configuration, run_configurations, summarize
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import plot_over_size

NAME = "structure"
SIZES = {
    "quick": [10, 100, 1_000, 10_000],
    "full": [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
}
ALGORITHMS = ["linear", "jump", "binary", "list_linear", "list_jump"]


def configurations(mode: str) -> list[Configuration]:
    return [
        Configuration(
            DatasetConfig(n=n, distribution="uniform", order="sorted", structure=structure, seed=SEED + i),
            QueryConfig(count=10_000, seed=SEED + i),
        )
        for i, n in enumerate(SIZES[mode])
        for structure in ("array", "linked_list")
    ]


def measure(mode: str) -> pd.DataFrame:
    return run_configurations(configurations(mode), TIMING[mode], seed=SEED, algorithms=ALGORITHMS)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    raw = run_or_load(NAME, args.mode, args.plot_only, lambda: measure(args.mode))
    summary = summarize(raw, ["n", "algorithm"])
    plot_over_size(summary, "median", "Laufzeit pro Anfrage [ns]", "Array (Direktzugriff) gegenüber verketteter Liste", result_path(NAME + "_runtime", args.mode, ".pdf"))
    counts = raw.groupby(["n", "algorithm"], as_index=False)["accesses"].first()
    plot_over_size(counts, "accesses", "Zugriffe pro Anfrage", "Gelesene Elemente bzw. besuchte Knoten", result_path(NAME + "_accesses", args.mode, ".pdf"), band=False)
    print(f"done: {result_path(NAME, args.mode).parent / f'{NAME}*'}")


if __name__ == "__main__":
    main()
