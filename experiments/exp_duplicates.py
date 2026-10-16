"""Experiment: runtime as a function of the share of duplicates.

Feature: Duplikate. Setting: fixed n, uniformly distributed, sorted; share of
duplicates 0 %, 10 %, 50 %; only hits (first occurrence) with uniformly
distributed positions; all array algorithms.

Outputs (results/<mode>/): duplicates.csv, duplicates.pdf
"""

from recommender.benchmark import Configuration, run_configurations, summarize
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import german_number, plot_by_category

NAME = "duplicates"
N = {"quick": 100_000, "full": 1_000_000}
SHARES = [0.0, 0.1, 0.5]


def configurations(mode: str) -> list[Configuration]:
    return [
        Configuration(
            DatasetConfig(n=N[mode], distribution="uniform", order="sorted", duplicate_share=share, seed=SEED),
            QueryConfig(count=10_000, seed=SEED),
        )
        for share in SHARES
    ]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    raw = run_or_load(NAME, args.mode, args.plot_only, lambda: run_configurations(configurations(args.mode), TIMING[args.mode], seed=SEED))
    summary = summarize(raw, ["duplicate_share", "algorithm"])
    plot_by_category(
        summary,
        "duplicate_share",
        SHARES,
        [f"{round(share * 100)} % Duplikate" for share in SHARES],
        "Laufzeit pro Anfrage [ns]",
        f"Laufzeit nach Duplikatanteil (n = {german_number(N[args.mode])}, gleichverteilt, sortiert)",
        result_path(NAME, args.mode, ".pdf"),
    )
    print(f"done: results/{args.mode}/{NAME}*")


if __name__ == "__main__":
    main()
