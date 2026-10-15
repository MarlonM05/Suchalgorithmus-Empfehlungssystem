"""Experiment: runtime as a function of the access pattern.

Feature: Zugriffsmuster. Setting: fixed n, uniformly distributed, sorted, no
duplicates; three patterns: hits at uniformly distributed positions, hits
concentrated at the front, only misses. Expected: exponential search profits
from targets near the front [bentley1976]; linear search pays the full n for
misses.

Outputs (results/<mode>/): access_pattern.csv, access_pattern.pdf
"""

from recommender.benchmark import Configuration, run_configurations, summarize
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DatasetConfig, QueryConfig
from recommender.viz.plots import german_number, plot_by_category

NAME = "access_pattern"
N = {"quick": 100_000, "full": 1_000_000}
# pattern key -> (hit ratio, position, label)
PATTERNS = {
    "hits_uniform": (1.0, "uniform", "Treffer,\ngleichverteilt"),
    "hits_front": (1.0, "front", "Treffer,\nvorne gehäuft"),
    "misses": (0.0, "uniform", "nur\nFehlschläge"),
}


def configurations(mode: str) -> list[Configuration]:
    return [
        Configuration(
            DatasetConfig(n=N[mode], distribution="uniform", order="sorted", seed=SEED),
            QueryConfig(count=10_000, hit_ratio=hit_ratio, position=position, seed=SEED),
        )
        for hit_ratio, position, _ in PATTERNS.values()
    ]


def pattern_of(hit_ratio: float, position: str) -> str:
    for key, (ratio, pos, _) in PATTERNS.items():
        if ratio == hit_ratio and (ratio == 0.0 or pos == position):
            return key
    raise ValueError((hit_ratio, position))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    raw = run_or_load(NAME, args.mode, args.plot_only, lambda: run_configurations(configurations(args.mode), TIMING[args.mode], seed=SEED))
    raw["pattern"] = [pattern_of(h, p) for h, p in zip(raw["hit_ratio"], raw["position"])]
    summary = summarize(raw, ["pattern", "algorithm"])
    plot_by_category(
        summary,
        "pattern",
        list(PATTERNS),
        [label for _, _, label in PATTERNS.values()],
        "Laufzeit pro Anfrage [ns]",
        f"Laufzeit nach Zugriffsmuster (n = {german_number(N[args.mode])}, gleichverteilt, sortiert)",
        result_path(NAME, args.mode, ".pdf"),
    )
    print(f"done: results/{args.mode}/{NAME}*")


if __name__ == "__main__":
    main()
