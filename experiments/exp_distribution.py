"""Experiment: runtime as a function of the value distribution.

Feature: Verteilung. Setting: fixed n, sorted, no duplicates, only hits with
uniformly distributed positions; all array algorithms. Expected: interpolation
search profits from uniform data and suffers on skewed data (lognormal)
[peterson1957], [vansandt2019]; the other algorithms do not depend on the
values.

Outputs (results/<mode>/): distribution.csv, distribution.pdf
"""

from recommender.benchmark import Configuration, run_configurations, summarize
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DISTRIBUTIONS, DatasetConfig, QueryConfig, generate_dataset
from recommender.features import extract_features
from recommender.viz.plots import german_number, plot_by_category

NAME = "distribution"
N = {"quick": 100_000, "full": 1_000_000}
LABELS = {
    "uniform": "gleichverteilt",
    "normal": "normalverteilt",
    "exponential": "exponentiell",
    "lognormal": "lognormal",
    "clustered": "geclustert",
}


def dataset(distribution: str, mode: str) -> DatasetConfig:
    return DatasetConfig(n=N[mode], distribution=distribution, order="sorted", seed=SEED)


def configurations(mode: str) -> list[Configuration]:
    return [Configuration(dataset(d, mode), QueryConfig(count=10_000, seed=SEED)) for d in DISTRIBUTIONS]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    raw = run_or_load(NAME, args.mode, args.plot_only, lambda: run_configurations(configurations(args.mode), TIMING[args.mode], seed=SEED))
    summary = summarize(raw, ["distribution", "algorithm"])
    # Label every distribution with its measured uniformity feature (KS distance).
    labels = []
    for d in DISTRIBUTIONS:
        ks = extract_features(generate_dataset(dataset(d, args.mode))).ks_uniform
        labels.append(f"{LABELS[d]}\nKS = {german_number(round(ks, 2))}")
    plot_by_category(
        summary,
        "distribution",
        list(DISTRIBUTIONS),
        labels,
        "Laufzeit pro Anfrage [ns]",
        f"Laufzeit nach Werteverteilung (n = {german_number(N[args.mode])}, sortiert)",
        result_path(NAME, args.mode, ".pdf"),
    )
    print(f"done: {result_path(NAME, args.mode).parent / f'{NAME}*'}")


if __name__ == "__main__":
    main()
