"""Timing of searches and preprocessing steps.

Methodology (see the chapter "Experimentelle Analyse"):

* **Blocks of queries.** A single search takes nanoseconds, below the
  resolution of the timer. Each measurement therefore times a block of
  queries and divides by the block length [chen2016]. The block length is
  calibrated by doubling until one block takes at least ``min_block_ns``.
  The whole block runs inside one compiled function, so Python overhead is
  paid once per block, not once per query.
* **Warm-up.** The first call compiles the function (Numba). It and further
  ``warmup`` repetitions are discarded [kalibera2013], [georges2007].
* **Fresh inputs per repetition.** Every repetition draws a new random block
  of keys from the query pool. Repeating identical inputs lets the CPU learn
  branch and access patterns and makes results look too good (observed in M2,
  see docs/annahmen.md A17) [mytkowicz2009].
* **All raw values are kept**; statistics are computed afterwards.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numba import njit

from ..algorithms import SearchAlgorithm, new_counter


@dataclass(frozen=True)
class TimingSettings:
    """Parameters of one measurement.

    Attributes:
        repetitions: measured repetitions per configuration.
        warmup: additional repetitions at the start that are discarded.
        min_block_ns: minimum duration of one timed block (default 1 ms,
            about 25 000 times the timer resolution on our machine).
        max_block_queries: upper limit for the block length.
        count_queries: number of queries used to count operations (the
            counts are deterministic, so a moderate number suffices).
    """

    repetitions: int = 30
    warmup: int = 2
    min_block_ns: int = 1_000_000
    max_block_queries: int = 100_000
    count_queries: int = 200


_block_runners: dict[str, Callable] = {}


def block_runner(algorithm: SearchAlgorithm) -> Callable:
    """Compiled loop that runs the (non-counting) search for every key of a block.

    The sum of all results is returned, so the compiler cannot remove the
    searches as dead code.
    """
    if algorithm.key not in _block_runners:
        search = algorithm.search

        @njit
        def run_block(data, keys, ops):
            checksum = 0
            for i in range(keys.shape[0]):
                checksum += search(data, keys[i], ops)
            return checksum

        _block_runners[algorithm.key] = run_block
    return _block_runners[algorithm.key]


def _time_block(runner: Callable, data: Any, keys: np.ndarray, ops: np.ndarray) -> int:
    start = time.perf_counter_ns()
    runner(data, keys, ops)
    return time.perf_counter_ns() - start


def calibrate_block_length(runner: Callable, data: Any, query_pool: np.ndarray, settings: TimingSettings) -> int:
    """Smallest power of two whose block takes at least ``min_block_ns``."""
    ops = new_counter()
    length = 1
    while length < settings.max_block_queries:
        keys = np.resize(query_pool, length)  # repeats the pool if it is shorter
        if _time_block(runner, data, keys, ops) >= settings.min_block_ns:
            break
        length *= 2
    return min(length, settings.max_block_queries)


def time_search(
    algorithm: SearchAlgorithm,
    data: Any,
    query_pool: np.ndarray,
    settings: TimingSettings,
    rng: np.random.Generator,
    query_blocks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Measure the runtime per query of ``algorithm`` on prepared ``data``.

    If ``query_blocks`` is given, its per-repetition query sequence is reused.
    This makes timing comparisons paired across algorithms while retaining
    fresh query blocks between repetitions.

    Returns the raw nanoseconds per query of every measured repetition and
    the block length used.
    """
    if query_pool.size == 0:
        raise ValueError("query_pool must contain at least one query")
    expected_blocks = settings.warmup + settings.repetitions
    if query_blocks is not None and (
        len(query_blocks) != expected_blocks
        or any(block.ndim != 1 or block.size < settings.max_block_queries for block in query_blocks)
    ):
        raise ValueError("query_blocks must provide one sufficiently large 1-D block per warmup and repetition")

    runner = block_runner(algorithm)
    ops = new_counter()
    runner(data, query_pool[:1], ops)  # compile (discarded)
    length = calibrate_block_length(runner, data, query_pool, settings)

    samples = []
    for repetition in range(settings.warmup + settings.repetitions):
        keys = query_blocks[repetition][:length] if query_blocks is not None else rng.choice(query_pool, length)
        elapsed = _time_block(runner, data, keys, ops)
        if repetition >= settings.warmup:
            samples.append(elapsed / length)
    return np.array(samples), length


def count_operations(algorithm: SearchAlgorithm, data: Any, query_pool: np.ndarray, settings: TimingSettings) -> tuple[float, float]:
    """Mean comparisons and accesses per query (counting variant)."""
    keys = query_pool[: settings.count_queries]
    results = [algorithm.find_counted(data, int(key)) for key in keys]
    return (
        float(np.mean([r.comparisons for r in results])),
        float(np.mean([r.accesses for r in results])),
    )


def time_call(function: Callable[[], Any], settings: TimingSettings) -> np.ndarray:
    """Raw nanoseconds of repeated calls of ``function`` (e.g. a preprocessing step).

    Used for steps that take much longer than the timer resolution (sorting,
    building a hash table), so each call is timed individually.
    """
    samples = []
    for repetition in range(settings.warmup + settings.repetitions):
        start = time.perf_counter_ns()
        function()
        elapsed = time.perf_counter_ns() - start
        if repetition >= settings.warmup:
            samples.append(elapsed)
    return np.array(samples, dtype=np.float64)


@njit
def sort_copy(arr):
    """Sorted copy of ``arr``, compiled like the search algorithms for a fair comparison."""
    return np.sort(arr)
