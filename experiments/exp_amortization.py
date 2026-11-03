"""Amortization analysis based on the shared measurement grid.

The measured grid rows are reused only when the dataset and query profile
match exactly. For unsorted uniform array data with no duplicates, the model
compares linear search, sorting plus branchless binary search, and hashing:

    T(q) = analysis + preprocessing + q * query cost

Selected complete runs repeat the same analysis, required preparation, and
queries to validate this additive model.

Outputs (results/<mode>/):
    amortization_summary.csv       component medians and break-even points
    amortization_validation.csv    complete-run times beside model predictions
    amortization_break_even.csv    break-even points in compact form
    amortization.pdf               total model cost T(q)
"""

from typing import Callable

import numpy as np
import pandas as pd

from recommender.algorithms import ALGORITHMS, new_counter
from recommender.algorithms.hashing import build_hash_table
from recommender.benchmark.experiment import (
    load_results,
    parse_args,
    result_path,
    save_results,
)
from recommender.benchmark.grid import PREPROCESSING
from recommender.benchmark.timing import TimingSettings, block_runner, sort_copy, time_call
from recommender.data import DatasetConfig, QueryConfig, generate_dataset, generate_queries
from recommender.features import extract_features
from recommender.viz import plots

NAME = "amortization"
SIZES = {"quick": [1_000, 100_000], "full": [1_000, 10_000, 100_000, 1_000_000]}
VALIDATION_SIZES = {"quick": [1_000, 100_000], "full": [1_000, 100_000]}
VALIDATION_QS = {"quick": [1, 10, 100], "full": [1, 100, 500]}
VALIDATION_TIMING = {
    "quick": TimingSettings(repetitions=3, warmup=1, max_block_queries=1_000),
    "full": TimingSettings(repetitions=5, warmup=1, max_block_queries=1_000),
}

# strategy -> (preprocessing component, per-query component, label, style key)
STRATEGIES = {
    "linear": (None, "linear_query_ns", "nur lineare Suche", "linear"),
    "sort_binary": ("sort_ns", "binary_query_ns", "Sortieren + binäre Suche", "binary"),
    "hash": ("hash_build_ns", "hash_query_ns", "Hashtabelle + Lookup", "hash"),
}
GRID_COMPONENTS = {
    "verify_ns": (PREPROCESSING, "verify"),
    "sort_ns": (PREPROCESSING, "sort"),
    "hash_build_ns": (PREPROCESSING, "hash_build"),
    "linear_query_ns": ("hits_uniform", "q_linear"),
    "binary_query_ns": ("hits_uniform", "q_after_sort_binary_branchless"),
    "hash_query_ns": ("hits_uniform", "q_hash"),
}


def _grid_scenarios(features: pd.DataFrame, mode: str) -> pd.DataFrame:
    required = {"dataset_id", "seed", "n", "distribution", "order", "duplicate_share", "structure"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(
            f"Grid features lack {sorted(missing)}; regenerate them with "
            f"`python experiments/exp_grid.py --mode {mode}`"
        )
    if "seed_replicate" not in features.columns:
        # Legacy grid (one seed per configuration, e.g. results/full of the Apple M5 run).
        features = features.assign(seed_replicate=0)
    selected = features[
        (features["n"].isin(SIZES[mode]))
        & (features["distribution"] == "uniform")
        & (features["order"] == "unsorted")
        & (features["duplicate_share"] == 0.0)
        & (features["structure"] == "array")
    ]
    if selected.empty:
        raise ValueError(f"No matching unsorted array datasets found in the {mode} measurement grid")
    return selected.copy()


def summarize_grid(features: pd.DataFrame, components: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Combine exact-match grid component medians into model costs."""
    scenarios = _grid_scenarios(features, mode)
    required = {"dataset_id", "access", "component", "ns"}
    missing = required - set(components.columns)
    if missing:
        raise ValueError(
            f"Grid components lack {sorted(missing)}; regenerate them with "
            f"`python experiments/exp_grid.py --mode {mode}`"
        )
    # The legacy grid has no query_id/query_seed columns; it holds exactly one
    # query pool per dataset and access pattern, so no further filter is needed.
    exact_queries = {"query_id", "query_seed"} <= set(components.columns)
    rows = []
    selected_ids = set(scenarios["dataset_id"].astype(int))
    for dataset_id in selected_ids:
        row = scenarios.loc[scenarios["dataset_id"] == dataset_id].iloc[0].to_dict()
        for output, (access, component) in GRID_COMPONENTS.items():
            matches = components[
                (components["dataset_id"] == dataset_id)
                & (components["access"] == access)
                & (components["component"] == component)
            ]
            if access != PREPROCESSING and exact_queries:
                matches = matches[
                    (matches["query_id"] == f"{dataset_id}:hits_uniform")
                    & (matches["query_seed"] == int(row["seed"]))
                ]
            if matches.empty:
                raise ValueError(
                    f"Grid is missing exact-match component {component!r} for dataset {dataset_id} "
                    f"and access pattern {access!r}"
                )
            row[output] = float(matches["ns"].median())
        row["inspection_ns"] = float(row["analysis_ns"])
        row["sort_break_even_q"] = _break_even(row["sort_ns"], row["linear_query_ns"], row["binary_query_ns"])
        row["hash_break_even_q"] = _break_even(
            row["hash_build_ns"], row["linear_query_ns"], row["hash_query_ns"]
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    grouping = ["n", "distribution", "order", "swap_fraction", "duplicate_share", "structure"]
    seed_spread = (
        summary.groupby(grouping, as_index=False)
        .agg(
            seed_count=("seed_replicate", "nunique"),
            sort_break_even_q_seed_min=("sort_break_even_q", "min"),
            sort_break_even_q_seed_median=("sort_break_even_q", "median"),
            sort_break_even_q_seed_max=("sort_break_even_q", "max"),
            hash_break_even_q_seed_min=("hash_break_even_q", "min"),
            hash_break_even_q_seed_median=("hash_break_even_q", "median"),
            hash_break_even_q_seed_max=("hash_break_even_q", "max"),
        )
    )
    summary = summary.merge(seed_spread, on=grouping, validate="many_to_one")
    return summary.sort_values(["n", "seed"]).reset_index(drop=True)


def _break_even(preparation_ns: float, linear_ns: float, alternative_ns: float) -> float:
    saving = linear_ns - alternative_ns
    return preparation_ns / saving if saving > 0 else float("inf")


def _complete_run(
    strategy: str,
    data: np.ndarray,
    queries: np.ndarray,
    seed: int,
) -> Callable[[], int]:
    """Build a full-run callable; dataset analysis and exact checking are included."""
    if strategy == "linear":
        algorithm = ALGORITHMS["linear"]
    elif strategy == "sort_binary":
        algorithm = ALGORITHMS["binary_branchless"]
    elif strategy == "hash":
        algorithm = ALGORITHMS["hash"]
    else:
        raise ValueError(f"Unknown amortization strategy: {strategy!r}")
    runner = block_runner(algorithm)

    def run() -> int:
        extract_features(data, seed=seed)
        if strategy == "sort_binary":
            prepared = sort_copy(data)
        elif strategy == "hash":
            prepared = build_hash_table(data)
        else:
            prepared = data
        return int(runner(prepared, queries, new_counter()))

    return run


def validate_model(summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Measure selected full strategy runs using exactly the grid dataset/workload."""
    selected = summary[
        summary["n"].isin(VALIDATION_SIZES[mode])
    ]
    rows = []
    settings = VALIDATION_TIMING[mode]
    for _, scenario in selected.iterrows():
        config = DatasetConfig(
            n=int(scenario["n"]),
            distribution=str(scenario["distribution"]),
            order=str(scenario["order"]),
            swap_fraction=float(scenario["swap_fraction"]),
            duplicate_share=float(scenario["duplicate_share"]),
            structure=str(scenario["structure"]),
            seed=int(scenario["seed"]),
        )
        data = generate_dataset(config)
        query_pool = generate_queries(
            data,
            QueryConfig(count=2_000, hit_ratio=1.0, position="uniform", seed=config.seed),
        )
        for q in VALIDATION_QS[mode]:
            queries = np.resize(query_pool, q)
            for strategy, (preprocessing, query_cost, _, _) in STRATEGIES.items():
                run = _complete_run(strategy, data, queries, config.seed)
                run()  # compile analysis/search/preparation paths before timing
                samples = time_call(run, settings)
                model_ns = float(scenario["analysis_ns"])
                if preprocessing is not None:
                    model_ns += float(scenario[preprocessing])
                model_ns += q * float(scenario[query_cost])
                rows.extend(
                    {
                        "dataset_id": int(scenario["dataset_id"]),
                        "n": int(scenario["n"]),
                        "seed": config.seed,
                        "query_id": f"{int(scenario['dataset_id'])}:hits_uniform",
                        "query_seed": config.seed,
                        "query_count": q,
                        "strategy": strategy,
                        "repetition": repetition,
                        "model_ns": model_ns,
                        "actual_ns": float(actual_ns),
                    }
                    for repetition, actual_ns in enumerate(samples)
                )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, mode: str) -> None:
    medians = summary.groupby("n")[
        ["analysis_ns", "sort_ns", "hash_build_ns", "linear_query_ns", "binary_query_ns", "hash_query_ns"]
    ].median()
    break_even = pd.DataFrame(
        {
            "sort_binary": {
                n: _break_even(row["sort_ns"], row["linear_query_ns"], row["binary_query_ns"])
                for n, row in medians.iterrows()
            },
            "hash": {
                n: _break_even(row["hash_build_ns"], row["linear_query_ns"], row["hash_query_ns"])
                for n, row in medians.iterrows()
            },
        }
    )
    break_even.index.name = "n"
    break_even.to_csv(result_path(NAME + "_break_even", mode))

    plots.apply_style()
    panels = [n for n in SIZES[mode] if n in medians.index]
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes_grid = plots.plt.subplots(
        rows,
        columns,
        figsize=(6.3, 2.9 * rows),
        sharey=False,
        squeeze=False,
    )
    axes_list = axes_grid.ravel()
    for axes in axes_list[len(panels) :]:
        fig.delaxes(axes)
    q = np.logspace(0, 7, 200)
    for panel_index, (axes, n) in enumerate(zip(axes_list, panels)):
        measured = medians.loc[n]
        for _, (preprocessing, query, label, style) in STRATEGIES.items():
            fixed = measured["analysis_ns"] + (measured[preprocessing] if preprocessing else 0.0)
            total = fixed + q * measured[query]
            kwargs = plots.style_kwargs(style)
            kwargs.update(marker=None, label=label)
            axes.plot(q, total / 1e6, **kwargs)
        for strategy, key, marker, label in [
            ("sort_binary", "sort_binary", "D", "Sortieren + binäre Suche"),
            ("hash", "hash", "P", "Hash-Lookup"),
        ]:
            crossing_q = float(break_even.loc[n, key])
            if not np.isfinite(crossing_q) or not (q[0] <= crossing_q <= q[-1]):
                continue
            crossing_ms = (
                measured["analysis_ns"] + crossing_q * measured["linear_query_ns"]
            ) / 1e6
            color = plots.ALGORITHM_STYLE[STRATEGIES[strategy][3]][1]
            axes.axvline(crossing_q, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
            axes.scatter(
                crossing_q,
                crossing_ms,
                marker=marker,
                s=48,
                color=color,
                edgecolors="white",
                linewidths=1,
                zorder=5,
                label=f"Schnittpunkt: {label}" if panel_index == 0 else None,
            )
            axes.annotate(
                f"q ≈ {plots.german_number(round(crossing_q, 1))}",
                (crossing_q, crossing_ms),
                xytext=(5, 8),
                textcoords="offset points",
                color=color,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_title(f"n = {plots.german_number(n)}", loc="left")
        axes.set_xlabel("Anzahl Anfragen q")
    axes_list[0].set_ylabel("Gesamtzeit [ms]")
    fig.suptitle("Unsortierte Daten: Gesamtkosten für q Anfragen", x=0.02, ha="left", fontsize=10)
    plots.legend_below(fig, axes_list[0], columns=3)
    plots.save_figure(fig, result_path(NAME, mode, ".pdf"))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    if args.plot_only:
        summary = load_results(NAME + "_summary", args.mode)
    else:
        grid_features_path = result_path("grid_features", args.mode)
        grid_components_path = result_path("grid_components", args.mode)
        if not grid_features_path.exists() or not grid_components_path.exists():
            raise FileNotFoundError(
                f"Run `python experiments/exp_grid.py --mode {args.mode}` before the amortization experiment"
            )
        features = load_results("grid_features", args.mode)
        components = load_results("grid_components", args.mode)
        summary = summarize_grid(features, components, args.mode)
        validation = validate_model(summary, args.mode)
        save_results(summary, NAME + "_summary", args.mode)
        save_results(validation, NAME + "_validation", args.mode, VALIDATION_TIMING[args.mode])
    plot(summary, args.mode)
    print(f"done: {result_path(NAME, args.mode).parent / f'{NAME}*'}")


if __name__ == "__main__":
    main()
