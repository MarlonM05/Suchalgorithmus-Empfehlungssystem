"""Robust statistics for runtime measurements.

Runtime measurements are usually not normally distributed but skewed or
multimodal [chen2016]. We therefore report

* the **median** with a 95 % **bootstrap** confidence interval
  (percentile method) [chen2016], [hoefler2015],
* the **minimum** as a robust estimate of the undisturbed runtime [chen2016],

and summarise relative comparisons (speedups) with the **geometric mean**,
because the arithmetic mean of ratios depends on the chosen baseline
[fleming1986].
"""

import numpy as np
import pandas as pd

CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000


def bootstrap_median_ci(
    values: np.ndarray,
    confidence: float = CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the median of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(resamples, values.size), replace=True)
    medians = np.median(resampled, axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, [alpha, 1.0 - alpha])
    return float(low), float(high)


def geometric_mean(values) -> float:
    """Geometric mean of positive values (e.g. speedups) [fleming1986]."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("geometric mean needs at least one value, all positive")
    return float(np.exp(np.mean(np.log(values))))


def summarize(frame: pd.DataFrame, group_columns: list[str], value_column: str = "ns_per_query") -> pd.DataFrame:
    """Median, bootstrap CI, minimum and count per group of raw measurements."""
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        values = group[value_column].to_numpy()
        low, high = bootstrap_median_ci(values)
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        row.update(
            median=float(np.median(values)),
            ci_low=low,
            ci_high=high,
            minimum=float(np.min(values)),
            repetitions=int(values.size),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def speedups(summary: pd.DataFrame, config_columns: list[str], baseline: str = "binary") -> pd.DataFrame:
    """Speedup of every algorithm relative to ``baseline`` per configuration.

    ``speedup > 1`` means faster than the baseline. Uses the medians of
    :func:`summarize` (column ``median``, algorithm in column ``algorithm``).
    """
    base = summary[summary["algorithm"] == baseline][config_columns + ["median"]]
    merged = summary.merge(base, on=config_columns, suffixes=("", "_baseline"))
    merged["speedup"] = merged["median_baseline"] / merged["median"]
    return merged.drop(columns="median_baseline")
