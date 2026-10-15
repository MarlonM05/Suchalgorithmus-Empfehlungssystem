"""Experiment: cost and accuracy of the feature analysis.

Features must be cheap compared to the search itself [kerschke2019]. Two parts:

1. Cost over n: sample-based feature analysis (sample size 1000) and the full
   sortedness check, compared with the cost of one binary and one linear
   search on the same data.
2. Sample size: cost and accuracy of the estimated disorder for sample sizes
   100 ... 10 000 on nearly sorted data with exactly known disorder. Every
   repetition uses a new seed, i.e. a new sample (see docs/annahmen.md A17).

Note: the analysis steps are timed per call, so their times include the
call overhead from Python into the compiled code (a few hundred ns), whereas
the searches are timed in blocks. This is the cost the tool really pays, but
it dominates for tiny n.

Outputs (results/<mode>/):
    analysis_cost.csv, analysis_cost.pdf
    analysis_sample.csv, analysis_sample.pdf
"""

import numpy as np
import pandas as pd

from recommender.algorithms import ALGORITHMS
from recommender.benchmark import summarize, time_search
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DatasetConfig, QueryConfig, generate_dataset, generate_queries
from recommender.features import extract_features, verify_sorted
from recommender.viz import plots

SIZES = {"quick": [100, 10_000, 1_000_000], "full": [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]}
SAMPLE_SIZES = {"quick": [100, 1_000, 10_000], "full": [100, 300, 1_000, 3_000, 10_000]}
SAMPLE_N = {"quick": 100_000, "full": 1_000_000}
# The estimation error needs many independent samples; this is cheap because
# only the estimate is evaluated (the timing uses the same calls).
ACCURACY_REPETITIONS = {"quick": 50, "full": 200}


def measure_cost(mode: str) -> pd.DataFrame:
    settings = TIMING[mode]
    rng = np.random.default_rng(SEED)
    rows = []
    for i, n in enumerate(SIZES[mode]):
        print(f"[cost {i + 1}/{len(SIZES[mode])}] n = {n}", flush=True)
        data = generate_dataset(DatasetConfig(n=n, order="sorted", seed=SEED + i))
        pool = generate_queries(data, QueryConfig(count=10_000, seed=SEED + i))
        repetitions = range(settings.warmup + settings.repetitions)
        series = {
            "features": [extract_features(data, seed=SEED + r).analysis_time_ns for r in repetitions][settings.warmup :],
            "verify_sorted": [verify_sorted(data).time_ns for _ in repetitions][settings.warmup :],
            "binary": time_search(ALGORITHMS["binary"], data, pool, settings, rng)[0],
            "linear": time_search(ALGORITHMS["linear"], data, pool, settings, rng)[0],
        }
        for algorithm, samples in series.items():
            rows += [{"n": n, "algorithm": algorithm, "repetition": r, "ns_per_query": ns} for r, ns in enumerate(samples)]
    return pd.DataFrame(rows)


def measure_sample(mode: str) -> pd.DataFrame:
    settings = TIMING[mode]
    keys = generate_dataset(DatasetConfig(n=SAMPLE_N[mode], order="nearly_sorted", swap_fraction=0.01, seed=SEED))
    true_disorder = float(np.mean(keys[:-1] > keys[1:]))
    rows = []
    for size in SAMPLE_SIZES[mode]:
        for r in range(settings.warmup + ACCURACY_REPETITIONS[mode]):
            features = extract_features(keys, sample_size=size, seed=SEED + r)
            if r >= settings.warmup:
                rows.append(
                    {
                        "sample_size": size,
                        "repetition": r - settings.warmup,
                        "analysis_ns": features.analysis_time_ns,
                        "disorder": features.disorder,
                        "true_disorder": true_disorder,
                    }
                )
    return pd.DataFrame(rows)


def plot_sample(raw: pd.DataFrame, mode: str) -> None:
    grouped = raw.groupby("sample_size")
    sizes = np.array(sorted(raw["sample_size"].unique()))
    time_us = grouped["analysis_ns"].median().loc[sizes] / 1e3
    rmse = grouped.apply(lambda g: float(np.sqrt(np.mean((g["disorder"] - g["true_disorder"]) ** 2))), include_groups=False).loc[sizes]
    p = raw["true_disorder"].iloc[0]

    plots.apply_style()
    fig, (left, right) = plots.plt.subplots(1, 2, figsize=(6.3, 2.8))
    left.plot(sizes, time_us, **{**plots.style_kwargs("features"), "linestyle": "-"})
    left.set_title("Kosten der Analyse", loc="left")
    left.set_ylabel("Median [µs]")
    right.plot(sizes, rmse, **{**plots.style_kwargs("features"), "linestyle": "-", "label": "gemessen"})
    right.plot(sizes, np.sqrt(p * (1 - p) / sizes), color=plots.INK_MUTED, linestyle="--", linewidth=1, label="Theorie √(p(1−p)/s)")
    right.set_title("Fehler des geschätzten Sortiergrads", loc="left")
    right.set_ylabel("RMSE")
    right.legend()
    for axes in (left, right):
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel("Stichprobengröße s")
    fig.tight_layout()
    plots.save_figure(fig, result_path("analysis_sample", mode, ".pdf"))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    cost = run_or_load("analysis_cost", args.mode, args.plot_only, lambda: measure_cost(args.mode))
    summary = summarize(cost, ["n", "algorithm"])
    plots.plot_over_size(summary, "median", "Zeit [ns]", "Analysekosten im Vergleich zu einer einzelnen Suche", result_path("analysis_cost", args.mode, ".pdf"))
    sample = run_or_load("analysis_sample", args.mode, args.plot_only, lambda: measure_sample(args.mode))
    plot_sample(sample, args.mode)
    print(f"done: results/{args.mode}/analysis_*")


if __name__ == "__main__":
    main()
