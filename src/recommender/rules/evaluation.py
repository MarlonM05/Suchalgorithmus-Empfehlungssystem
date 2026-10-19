"""Evaluation of the selectors against oracle and single best solver.

Measures (following [kerschke2019], [bischl2016]):

* **Trefferquote**: share of instances where the selector picks the oracle's
  strategy, plus the share within 10 % of the optimal cost (near-ties between
  similar algorithms are common and cost almost nothing).
* **Slowdown**: realised cost / oracle cost per instance; summarised by the
  geometric mean [fleming1986], median, 95 % quantile and maximum.
* **Geschlossene Lücke** between single best solver (SBS) and oracle (VBS):
  ``(SBS - selector) / (SBS - VBS)``, on the summed cost (dominated by large
  instances) and on the geometric mean of the slowdowns (every instance
  counts equally). 1 = as good as the oracle, 0 = no better than the SBS.

Realised cost of a selector on an instance:

* its strategy's cost (from the linear cost model),
* plus the analysis cost if the decision needed a sampled feature,
* if the strategy needs sorted data but the data is not sorted (possible
  when the sample overlooks a disorder), the full check fails: its (early
  exit) cost is paid and the selector decides again knowing the data is
  unsorted,
* a strategy for the wrong storage structure (only possible for the tree) is
  replaced by its counterpart (e.g. hash -> list_hash) and counted.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd

from .decision import FEATURE_NAMES_DE, Recommendation
from .strategies import STRATEGIES

FEATURE_KEYS = list(FEATURE_NAMES_DE)
WITHIN = 1.10

# Counterparts for strategies predicted for the wrong structure.
COUNTERPART = {
    "linked_list": {"linear": "list_linear", "hash": "list_hash", "sort_binary": "list_hash"},
    "array": {"list_linear": "linear", "list_hash": "hash", "list_jump": "binary_branchless"},
}

Decide = Callable[[dict[str, float]], Recommendation]


def instance_features(row) -> dict[str, float]:
    return {key: float(row[key]) for key in FEATURE_KEYS}


def realised(row, decide: Decide) -> dict:
    """Strategy and realised cost of a selector on one instance."""
    features = instance_features(row)
    recommendation = decide(features)
    strategy, sampled, wrong_structure, failed_check = recommendation.strategy, recommendation.uses_sampled_features, False, False
    if STRATEGIES[strategy].structure != row["structure"]:
        mapping = COUNTERPART[row["structure"]]
        strategy = mapping.get(strategy, "list_linear" if row["structure"] == "linked_list" else "linear")
        wrong_structure = True
    cost = float(row[f"cost_{strategy}"])
    extra = 0.0
    if math.isnan(cost):  # needs sorted data, but the data is not sorted: check fails
        failed_check = True
        extra += float(row["verify_ns"])
        retry = decide({**features, "looks_sorted": 0.0, "disorder": 1.0})
        strategy = retry.strategy
        if STRATEGIES[strategy].structure != row["structure"] or math.isnan(float(row[f"cost_{strategy}"])):
            strategy = "list_linear" if row["structure"] == "linked_list" else "linear"
        cost = float(row[f"cost_{strategy}"])
    analysis = float(row["analysis_ns"]) if sampled else 0.0
    return {
        "chosen": strategy,
        "cost": cost + extra,
        "cost_with_analysis": cost + extra + analysis,
        "used_sample": sampled,
        "wrong_structure": wrong_structure,
        "failed_check": failed_check,
    }


def evaluate(instances: pd.DataFrame, decide: Decide, name: str) -> pd.DataFrame:
    """Per-instance results of one selector."""
    rows = [realised(row, decide) for _, row in instances.iterrows()]
    result = pd.DataFrame(rows, index=instances.index)
    result.insert(0, "selector", name)
    return result


def fixed_strategy(instances: pd.DataFrame, strategy_by_structure: dict[str, str], name: str) -> pd.DataFrame:
    """A selector that always picks the same strategy (per structure); no analysis."""
    chosen = instances["structure"].map(strategy_by_structure)
    cost = [float(row[f"cost_{s}"]) for (_, row), s in zip(instances.iterrows(), chosen)]
    return pd.DataFrame(
        {
            "selector": name,
            "chosen": chosen,
            "cost": cost,
            "cost_with_analysis": cost,
            "used_sample": False,
            "wrong_structure": False,
            "failed_check": False,
        },
        index=instances.index,
    )


def oracle(instances: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "selector": "Oracle (VBS)",
            "chosen": instances["oracle"],
            "cost": instances["oracle_cost"],
            "cost_with_analysis": instances["oracle_cost"],
            "used_sample": False,
            "wrong_structure": False,
            "failed_check": False,
        },
        index=instances.index,
    )


def single_best(instances: pd.DataFrame) -> dict[str, str]:
    """Per structure the strategy with the best geometric-mean slowdown among those always applicable."""
    best = {}
    for structure, group in instances.groupby("structure"):
        always = [s for s in STRATEGIES.values() if s.structure == structure and not s.requires_sorted]
        scores = {s.key: float(np.exp(np.mean(np.log(group[f"cost_{s.key}"] / group["oracle_cost"])))) for s in always}
        best[structure] = min(scores, key=scores.get)
    return best


def always_binary(instances: pd.DataFrame) -> pd.DataFrame:
    """Baseline "always binary search" on arrays: check, sort if necessary, then binary search."""
    arrays = instances[instances["structure"] == "array"]
    costs = []
    for _, row in arrays.iterrows():
        if row["true_sorted"]:
            costs.append(float(row["cost_binary_branchless"]))  # includes the check unless guaranteed
        else:
            costs.append(float(row["verify_ns"]) + float(row["cost_sort_binary"]))
    chosen = np.where(arrays["true_sorted"], "binary_branchless", "sort_binary")
    return pd.DataFrame(
        {"selector": "Immer binäre Suche", "chosen": chosen, "cost": costs, "cost_with_analysis": costs, "used_sample": False, "wrong_structure": False, "failed_check": False},
        index=arrays.index,
    )


def summarize(results: pd.DataFrame, instances: pd.DataFrame, sbs_name: str, with_analysis: bool) -> pd.DataFrame:
    """One row per selector with all evaluation measures."""
    cost_column = "cost_with_analysis" if with_analysis else "cost"
    rows = []
    sbs = results[results["selector"] == sbs_name]
    for name, group in results.groupby("selector", sort=False):
        base = instances.loc[group.index]
        slowdown = group[cost_column].to_numpy() / base["oracle_cost"].to_numpy()
        gm = float(np.exp(np.mean(np.log(slowdown))))
        sbs_part = sbs.loc[sbs.index.intersection(group.index)]
        sbs_slowdown = sbs_part[cost_column].to_numpy() / instances.loc[sbs_part.index, "oracle_cost"].to_numpy()
        gm_sbs = float(np.exp(np.mean(np.log(sbs_slowdown))))
        sum_sbs, sum_vbs, sum_sel = sbs_part[cost_column].sum(), base["oracle_cost"].sum(), group[cost_column].sum()
        rows.append(
            {
                "Selektor": name,
                "Instanzen": len(group),
                "Trefferquote": float(np.mean(group["chosen"].to_numpy() == base["oracle"].to_numpy())),
                f"≤ {WITHIN:.1f}× Optimum": float(np.mean(slowdown <= WITHIN)),
                "Slowdown geom. Mittel": gm,
                "Slowdown Median": float(np.median(slowdown)),
                "Slowdown 95 %": float(np.quantile(slowdown, 0.95)),
                "Slowdown max": float(np.max(slowdown)),
                "Lücke geschlossen (Summe)": (sum_sbs - sum_sel) / (sum_sbs - sum_vbs) if sum_sbs != sum_vbs else math.nan,
                "Lücke geschlossen (geom.)": (math.log(gm_sbs) - math.log(gm)) / math.log(gm_sbs) if gm_sbs > 1 else math.nan,
                "Stichprobe genutzt": float(group["used_sample"].mean()),
                "Prüfung fehlgeschlagen": int(group["failed_check"].sum()),
                "Falsche Struktur": int(group["wrong_structure"].sum()),
            }
        )
    return pd.DataFrame(rows)
