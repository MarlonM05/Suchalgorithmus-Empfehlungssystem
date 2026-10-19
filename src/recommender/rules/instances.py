"""Build evaluation instances from the measurement grid.

An *instance* is one problem of the algorithm selection problem [rice1976]:
a dataset, an access pattern, a number of queries q and the information
whether the user guarantees sortedness. For every instance this module
computes the total cost of every strategy from the measured components
(linear cost model, docs/annahmen.md A22) and the best strategy (the oracle or
virtual best solver [kerschke2019]).
"""

import math

import pandas as pd

from ..benchmark.grid import ACCESS_PATTERNS, PREPROCESSING
from ..features import DatasetFeatures
from .decision import selection_features
from .strategies import Strategy, Workload, strategies_for

QUERY_COUNTS = [1, 3, 10, 30, 100, 300, 1_000, 10_000, 100_000, 1_000_000]


def component_medians(components: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    """Median cost of every component, keyed by (dataset_id, access)."""
    medians = components.groupby(["dataset_id", "access", "component"])["ns"].median()
    result: dict[tuple[int, str], dict[str, float]] = {}
    for (dataset_id, access, component), value in medians.items():
        result.setdefault((dataset_id, access), {})[component] = float(value)
    return result


def strategy_cost(strategy: Strategy, prep: dict, per_query: dict, q: int, sorted_known: bool, true_sorted: bool) -> float:
    """Total cost in ns of ``strategy`` for q queries; NaN if it cannot answer correctly."""
    if strategy.requires_sorted and not true_sorted:
        return math.nan
    cost = 0.0
    if strategy.requires_sorted and not sorted_known:
        cost += prep["verify"]
    if strategy.preprocessing is not None:
        cost += prep[strategy.preprocessing]
    search = "after_sort_binary_branchless" if strategy.preprocessing == "sort" else strategy.search
    return cost + q * per_query[f"q_{search}"]


def dataset_features(row) -> DatasetFeatures:
    """Reconstruct the estimated features of a dataset from the grid table."""
    duplicate = None if row["duplicate_share_est"] < 0 else float(row["duplicate_share_est"])
    return DatasetFeatures(
        n=int(row["est_n"]),
        structure=row["structure"],
        sample_size=int(row["sample_size"]),
        disorder=float(row["disorder"]),
        duplicate_share=duplicate,
        ks_uniform=float(row["ks_uniform"]),
        analysis_time_ns=int(row["analysis_ns"]),
    )


def build_instances(features: pd.DataFrame, components: pd.DataFrame, query_counts=QUERY_COUNTS) -> pd.DataFrame:
    """One row per instance with selection features, costs of all strategies and the oracle."""
    medians = component_medians(components)
    rows = []
    for _, dataset in features.iterrows():
        dataset_id = int(dataset["dataset_id"])
        prep = medians[(dataset_id, PREPROCESSING)]
        data_features = dataset_features(dataset)
        true_sorted = bool(dataset["true_sorted"])
        candidates = strategies_for(dataset["structure"])
        for access, (hit_ratio, position) in ACCESS_PATTERNS.items():
            per_query = medians[(dataset_id, access)]
            for q in query_counts:
                for sorted_known in ([False, True] if true_sorted else [False]):
                    workload = Workload(queries=q, hit_ratio=hit_ratio, position=position, sorted_known=sorted_known)
                    costs = {s.key: strategy_cost(s, prep, per_query, q, sorted_known, true_sorted) for s in candidates}
                    oracle = min((k for k, v in costs.items() if not math.isnan(v)), key=costs.get)
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "access": access,
                            "structure": dataset["structure"],
                            "distribution": dataset["distribution"],
                            "order": dataset["order"],
                            "duplicate_share": dataset["duplicate_share"],
                            "true_sorted": true_sorted,
                            "analysis_ns": float(dataset["analysis_ns"]),
                            "verify_ns": prep["verify"],
                            **selection_features(data_features, workload),
                            **{f"cost_{k}": v for k, v in costs.items()},
                            "oracle": oracle,
                            "oracle_cost": costs[oracle],
                        }
                    )
    return pd.DataFrame(rows)
