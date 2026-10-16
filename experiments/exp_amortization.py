"""Experiment: when do sorting or hashing pay off on unsorted data?

Feature: Sortierung (unsorted data) together with the expected number of
queries q. Three strategies for q queries on unsorted data:

    only linear search:              T(q) = q * t_linear
    sort, then binary search:        T(q) = t_sort + q * t_binary
    build hash table, then lookup:   T(q) = t_build + q * t_hash

The theory predicts that linear search wins for small q and sorting or hashing
for large q. This experiment measures the five components separately (each
with the full methodology) and determines the break-even points empirically:

    q* = t_sort / (t_linear - t_binary)   and   q* = t_build / (t_linear - t_hash)

Outputs (results/<mode>/):
    amortization.csv               raw measurements per component
    amortization_break_even.csv    break-even points per n (from medians)
    amortization.pdf               T(q) per strategy, one panel per n
"""

import numpy as np
import pandas as pd

from recommender.algorithms import ALGORITHMS
from recommender.algorithms.hashing import build_hash_table
from recommender.benchmark import sort_copy, time_call, time_search
from recommender.benchmark.experiment import SEED, TIMING, parse_args, result_path, run_or_load
from recommender.data import DatasetConfig, QueryConfig, generate_dataset, generate_queries
from recommender.viz import plots

NAME = "amortization"
SIZES = {"quick": [1_000, 100_000], "full": [1_000, 10_000, 100_000, 1_000_000, 10_000_000]}
PANEL_SIZES = {"quick": [1_000, 100_000], "full": [1_000, 100_000, 10_000_000]}

# strategy -> (preprocessing component, per-query component, legend label, style key)
STRATEGIES = {
    "linear": (None, "linear_query", "nur lineare Suche", "linear"),
    "sort_binary": ("sort", "binary_query", "Sortieren + binäre Suche", "binary"),
    "hash": ("hash_build", "hash_query", "Hashtabelle + Lookup", "hash"),
}


def measure(mode: str) -> pd.DataFrame:
    settings = TIMING[mode]
    rng = np.random.default_rng(SEED)
    rows = []
    for i, n in enumerate(SIZES[mode]):
        print(f"[{i + 1}/{len(SIZES[mode])}] n = {n}", flush=True)
        data = generate_dataset(DatasetConfig(n=n, distribution="uniform", order="unsorted", seed=SEED + i))
        pool = generate_queries(data, QueryConfig(count=10_000, seed=SEED + i))
        sorted_data = sort_copy(data)
        table = build_hash_table(data)
        components = {
            "linear_query": time_search(ALGORITHMS["linear"], data, pool, settings, rng)[0],
            "binary_query": time_search(ALGORITHMS["binary"], sorted_data, pool, settings, rng)[0],
            "hash_query": time_search(ALGORITHMS["hash"], table, pool, settings, rng)[0],
            "sort": time_call(lambda: sort_copy(data), settings),
            "hash_build": time_call(lambda: build_hash_table(data), settings),
        }
        for component, samples in components.items():
            rows += [{"n": n, "component": component, "repetition": r, "ns": ns} for r, ns in enumerate(samples)]
    return pd.DataFrame(rows)


def break_even(medians: pd.Series) -> dict[str, float]:
    """Number of queries from which a strategy beats linear search (inf if never)."""
    result = {}
    for strategy in ("sort_binary", "hash"):
        prep, query, _, _ = STRATEGIES[strategy]
        saving = medians["linear_query"] - medians[query]
        result[strategy] = medians[prep] / saving if saving > 0 else float("inf")
    return result


def plot(raw: pd.DataFrame, mode: str) -> None:
    medians = raw.groupby(["n", "component"])["ns"].median().unstack()
    table = pd.DataFrame({n: break_even(medians.loc[n]) for n in medians.index}).T
    table.index.name = "n"
    table.to_csv(result_path(NAME + "_break_even", mode))

    plots.apply_style()
    panels = PANEL_SIZES[mode]
    fig, axes_list = plots.plt.subplots(1, len(panels), figsize=(6.3, 2.9), sharey=False)
    q = np.logspace(0, 7, 200)
    for axes, n in zip(np.atleast_1d(axes_list), panels):
        m = medians.loc[n]
        for strategy, (prep, query, label, style) in STRATEGIES.items():
            total = (m[prep] if prep else 0.0) + q * m[query]
            kwargs = plots.style_kwargs(style)
            kwargs.update(marker=None, label=label)
            axes.plot(q, total / 1e6, **kwargs)  # milliseconds
        for strategy, q_star in break_even(m).items():
            if np.isfinite(q_star) and q_star <= q[-1]:
                y = q_star * m["linear_query"] / 1e6
                axes.plot([q_star], [y], marker="o", color=plots.INK, markersize=4, linestyle="none")
                axes.annotate(f"q* ≈ {plots.german_number(round(q_star))}", (q_star, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=plots.INK_SECONDARY)
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_title(f"n = {plots.german_number(n)}", loc="left")
        axes.set_xlabel("Anzahl Anfragen q")
    np.atleast_1d(axes_list)[0].set_ylabel("Gesamtzeit [ms]")
    fig.suptitle("Unsortierte Daten: Gesamtkosten für q Anfragen", x=0.02, ha="left", fontsize=10)
    plots.legend_below(fig, np.atleast_1d(axes_list)[0], columns=3)
    plots.save_figure(fig, result_path(NAME, mode, ".pdf"))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    raw = run_or_load(NAME, args.mode, args.plot_only, lambda: measure(args.mode))
    plot(raw, args.mode)
    print(f"done: results/{args.mode}/{NAME}*")


if __name__ == "__main__":
    main()
