"""Cheap, sample-based characteristics of a dataset.

Features for algorithm selection must be cheap to compute [kerschke2019]: a
full O(n) scan would already cost as much as a linear search. Therefore all
features except the size are **estimated on a random sample**, similar to the
runtime statistics estimated in [li2004]. The cost of the analysis is measured
and reported, so the tool can state honestly when analysis plus preprocessing
pay off.

Features
--------
* ``n``: size, O(1).
* ``disorder``: share of adjacent pairs ``(a[i], a[i+1])`` in wrong order
  (``a[i] > a[i+1]``), estimated on randomly chosen pairs. 0 for sorted data,
  about 0.5 for random order.
* ``duplicate_share``: share of keys equal to their predecessor, estimated on
  the same pairs. For sorted data this equals ``(n - distinct) / n``, the
  duplicate share of the generator. For unsorted data equal keys are usually
  not adjacent, and sampling cannot estimate the number of distinct values
  reliably [charikar2000]; the feature is then ``None``.
* ``ks_uniform``: Kolmogorov-Smirnov distance between the sampled values and
  a uniform distribution on [sample min, sample max]. 0 = perfectly uniform,
  close to 1 = strongly skewed. For sorted data this measures how far the
  "index -> value" curve deviates from a straight line, which is exactly the
  assumption of interpolation search [peterson1957].

Sample size: with the default of 1000 pairs/values, the 95 % critical value of
the KS statistic is about 1.36 / sqrt(1000) = 0.043, and the standard error of
the estimated disorder is at most 0.5 / sqrt(1000) = 0.016.

Because the sample cannot prove that data is sorted, :func:`verify_sorted`
offers a full O(n) check whose cost is reported separately. The recommender
calls it before recommending an algorithm that requires sorted data.
"""

import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..data.linked_list import END, LinkedList

DEFAULT_SAMPLE_SIZE = 1000


@dataclass(frozen=True)
class DatasetFeatures:
    """Estimated characteristics of a dataset and the cost of estimating them."""

    n: int
    structure: str
    sample_size: int
    disorder: float
    duplicate_share: float | None
    ks_uniform: float
    analysis_time_ns: int

    @property
    def looks_sorted(self) -> bool:
        """True if no sampled pair was out of order (not a proof, see verify_sorted)."""
        return self.disorder == 0.0


@dataclass(frozen=True)
class SortCheck:
    """Result and cost of the full sortedness check."""

    is_sorted: bool
    time_ns: int


@njit
def ks_distance_to_uniform(sample):
    """Kolmogorov-Smirnov distance of ``sample`` to U[min(sample), max(sample)]."""
    values = np.sort(sample.astype(np.float64))
    m = values.shape[0]
    if m < 2 or values[m - 1] == values[0]:
        return 0.0
    lo = values[0]
    span = values[m - 1] - lo
    distance = 0.0
    for i in range(m):
        u = (values[i] - lo) / span  # CDF of the uniform distribution at values[i]
        distance = max(distance, (i + 1) / m - u, u - i / m)
    return distance


@njit
def _estimate(values, next_index, is_list, sample_size, seed):
    """Compiled core of the feature estimation.

    Returns ``(pairs, pairs_out_of_order, pairs_equal, ks_uniform)``.

    Adjacent pairs: in the array ``(a[i], a[i+1])``; in the linked list a node
    slot and its successor, which also costs O(1) per pair because the node
    pool allows picking random nodes (see docs/annahmen.md). If the data has
    at most ``sample_size`` pairs or values, all of them are used (exact
    result); otherwise positions are drawn uniformly with replacement.
    """
    np.random.seed(seed)  # Numba keeps its own random state, seeded here
    n = values.shape[0]

    population = n if is_list else n - 1
    use_all = population <= sample_size
    draws = max(population, 0) if use_all else sample_size
    pairs = 0
    out_of_order = 0
    equal = 0
    for t in range(draws):
        i = t if use_all else np.random.randint(0, population)
        if is_list:
            j = next_index[i]
            if j == END:
                continue  # the last node has no successor
        else:
            j = i + 1
        pairs += 1
        if values[i] > values[j]:
            out_of_order += 1
        elif values[i] == values[j]:
            equal += 1

    m = n if n <= sample_size else sample_size
    sample = np.empty(m, dtype=np.int64)
    for t in range(m):
        sample[t] = values[t] if n <= sample_size else values[np.random.randint(0, n)]
    return pairs, out_of_order, equal, ks_distance_to_uniform(sample)


_NO_SUCCESSORS = np.empty(0, dtype=np.int64)


def extract_features(
    data: np.ndarray | LinkedList,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> DatasetFeatures:
    """Estimate the characteristics of ``data`` from a random sample.

    The reported ``analysis_time_ns`` covers the complete call except the
    one-time compilation, which is triggered beforehand on a tiny input.
    """
    is_list = isinstance(data, LinkedList)
    values = data.values if is_list else data
    next_index = data.next_index if is_list else _NO_SUCCESSORS
    _estimate(values[:2], next_index[:0], False, 1, 0)  # compile before timing
    if is_list:
        _estimate(values[:0], next_index[:0], True, 1, 0)

    start = time.perf_counter_ns()
    pairs, out_of_order, equal, ks_uniform = _estimate(values, next_index, is_list, sample_size, seed)
    disorder = out_of_order / pairs if pairs > 0 else 0.0
    if disorder == 0.0:
        # In sorted data, (n - distinct) keys equal their predecessor among the
        # n - 1 adjacent pairs; scaling by (n - 1) / n yields (n - distinct) / n.
        n = values.shape[0]
        duplicate_share = equal / pairs * (n - 1) / n if pairs > 0 else 0.0
    else:
        duplicate_share = None
    elapsed = time.perf_counter_ns() - start

    return DatasetFeatures(
        n=int(values.shape[0]),
        structure="linked_list" if is_list else "array",
        sample_size=int(pairs),
        disorder=float(disorder),
        duplicate_share=None if duplicate_share is None else float(duplicate_share),
        ks_uniform=float(ks_uniform),
        analysis_time_ns=elapsed,
    )


@njit
def _array_is_sorted(arr):
    for i in range(arr.shape[0] - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


@njit
def _list_is_sorted(values, next_index, head):
    slot = head
    while slot != END and next_index[slot] != END:
        if values[slot] > values[next_index[slot]]:
            return False
        slot = next_index[slot]
    return True


def verify_sorted(data: np.ndarray | LinkedList) -> SortCheck:
    """Check sortedness exactly with one O(n) pass (stops at the first violation)."""
    # Compile before timing, so the reported cost does not include compilation.
    _array_is_sorted(np.zeros(2, dtype=np.int64))
    _list_is_sorted(np.zeros(1, dtype=np.int64), np.full(1, END, dtype=np.int64), 0)

    start = time.perf_counter_ns()
    if isinstance(data, LinkedList):
        result = _list_is_sorted(*data.as_tuple())
    else:
        result = _array_is_sorted(data)
    return SortCheck(is_sorted=bool(result), time_ns=time.perf_counter_ns() - start)
