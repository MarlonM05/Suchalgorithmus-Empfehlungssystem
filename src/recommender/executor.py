"""Run a recommended strategy on real data (automatic execution of the search).

The executor performs exactly the steps of a strategy (rules/strategies.py):

1. sortedness check, if the strategy needs sorted data and the user did not
   guarantee it; if the check fails, the search is not run,
2. preprocessing (sorting, building the hash table, copying the list),
3. the search for the given key, once with the counting variant (index and
   operation counts) and repeatedly with the timing variant (runtime).

All indices refer to the data as the user provided it: after sorting, a
stable argsort maps the position in the sorted copy back to the original
position. Because the sort is stable, the first occurrence in the sorted copy
is the first occurrence in the original data.
"""

import time
from dataclasses import dataclass

import numpy as np

from .algorithms import ALGORITHMS, LIST_ALGORITHMS, new_counter
from .algorithms.hashing import build_hash_table
from .benchmark.timing import block_runner
from .data import LinkedList, linked_list_to_array
from .features import verify_sorted
from .rules.strategies import STRATEGIES, Strategy

TIMING_REPETITIONS = 1_000  # the key is searched this often to time a single query


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running one strategy for one key.

    Times are in nanoseconds. ``total_ns`` estimates the cost of q queries
    with the linear cost model (preprocessing + q * time per query).
    """

    strategy: str
    index: int
    check_ns: int
    preprocessing_ns: int
    query_ns: float
    comparisons: int
    accesses: int
    check_failed: bool = False

    def total_ns(self, queries: int) -> float:
        return self.check_ns + self.preprocessing_ns + queries * self.query_ns


def applicable_strategies(data: np.ndarray | LinkedList, is_sorted: bool) -> list[Strategy]:
    """Strategies that can answer correctly on this data."""
    structure = "linked_list" if isinstance(data, LinkedList) else "array"
    return [s for s in STRATEGIES.values() if s.structure == structure and (is_sorted or not s.requires_sorted)]


def _timed(function):
    start = time.perf_counter_ns()
    value = function()
    return value, time.perf_counter_ns() - start


def _prepare(strategy: Strategy, data: np.ndarray | LinkedList):
    """Run the preprocessing; returns (prepared data, mapping to original indices or None, ns)."""
    if strategy.preprocessing == "sort":
        start = time.perf_counter_ns()
        order = np.argsort(data, kind="stable")  # sorted position -> original position
        sorted_data = np.ascontiguousarray(data[order])
        return sorted_data, order, time.perf_counter_ns() - start
    if strategy.preprocessing == "hash_build":
        table, elapsed = _timed(lambda: build_hash_table(data))
        return table, None, elapsed
    if strategy.preprocessing == "list_hash_build":
        table, elapsed = _timed(lambda: build_hash_table(linked_list_to_array(data)))
        return table, None, elapsed
    prepared = data.as_tuple() if isinstance(data, LinkedList) else data
    return prepared, None, 0


def execute(strategy_key: str, data: np.ndarray | LinkedList, key: int, sorted_known: bool = False) -> ExecutionResult:
    """Run ``strategy_key`` on ``data`` for ``key`` and measure it."""
    strategy = STRATEGIES[strategy_key]
    check_ns = 0
    if strategy.requires_sorted and not sorted_known:
        check = verify_sorted(data)
        check_ns = check.time_ns
        if not check.is_sorted:
            return ExecutionResult(strategy_key, -1, check_ns, 0, float("nan"), 0, 0, check_failed=True)

    prepared, original_index, preprocessing_ns = _prepare(strategy, data)
    algorithm = (LIST_ALGORITHMS | ALGORITHMS)[strategy.search]

    counted = algorithm.find_counted(prepared, key)
    index = counted.index
    if original_index is not None and index >= 0:
        index = int(original_index[index])

    runner = block_runner(algorithm)
    keys = np.full(TIMING_REPETITIONS, key, dtype=np.int64)
    ops = new_counter()
    runner(prepared, keys[:1], ops)  # compile (first call only)
    _, elapsed = _timed(lambda: runner(prepared, keys, ops))
    return ExecutionResult(
        strategy=strategy_key,
        index=index,
        check_ns=check_ns,
        preprocessing_ns=preprocessing_ns,
        query_ns=elapsed / TIMING_REPETITIONS,
        comparisons=counted.comparisons,
        accesses=counted.accesses,
    )


def expected_index(data: np.ndarray | LinkedList, key: int) -> int:
    """Reference answer: index of the first occurrence of ``key`` or -1.

    Computed independently of all search algorithms with one vectorised NumPy
    comparison over the data in logical order (for lists: list position). The
    user interface uses it to mark every result as correct or wrong. A wrong
    result is possible only if the user guaranteed sortedness for data that
    is not sorted: then the check is skipped and binary search may miss.
    """
    keys = linked_list_to_array(data) if isinstance(data, LinkedList) else np.asarray(data)
    matches = np.flatnonzero(keys == key)
    return int(matches[0]) if matches.size else -1


def compare_all(data: np.ndarray | LinkedList, key: int, sorted_known: bool = False) -> list[ExecutionResult]:
    """Run every applicable strategy on the same data (comparison mode)."""
    is_sorted = sorted_known or verify_sorted(data).is_sorted
    return [execute(s.key, data, key, sorted_known) for s in applicable_strategies(data, is_sorted)]


def warm_up() -> None:
    """Compile all search functions once on tiny inputs (Numba compiles on first use)."""
    from .data import build_linked_list

    tiny = np.arange(4, dtype=np.int64)
    linked = build_linked_list(tiny, np.random.default_rng(0))
    for data in (tiny, linked):
        compare_all(data, 2, sorted_known=False)
