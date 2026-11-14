"""Search keys (queries) that follow a given access pattern.

The access pattern is described by a :class:`QueryConfig`:

* ``count``: number of queries q.
* ``hit_ratio``: share of queries whose key is contained in the data.
* ``position``: where the hits are located in the stored order:
  ``"uniform"`` (every position equally likely) or ``"front"`` (positions near
  the front are much more likely). For ``"front"`` the position follows an
  exponential distribution with mean ``front_mean * n``, truncated to the
  array. With the default 0.01 about 63 % of the hits lie in the first 1 %
  and 99 % in the first 4.6 % of the positions.

Misses are drawn uniformly from the value range of the data (so they fall into
gaps between stored keys); keys that happen to be present are redrawn. If the
range contains no gaps, misses are taken from outside the range.
"""

from dataclasses import dataclass

import numpy as np

from .linked_list import LinkedList, linked_list_to_array

POSITIONS = ("uniform", "front")


@dataclass(frozen=True)
class QueryConfig:
    """Access pattern of a workload."""

    count: int
    hit_ratio: float = 1.0
    position: str = "uniform"
    front_mean: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must not be negative")
        if not 0.0 <= self.hit_ratio <= 1.0:
            raise ValueError("hit_ratio must lie in [0, 1]")
        if self.position not in POSITIONS:
            raise ValueError(f"unknown position {self.position!r}")
        if not 0.0 < self.front_mean <= 1.0:
            raise ValueError("front_mean must lie in (0, 1]")


def _hit_positions(n: int, config: QueryConfig, count: int, rng: np.random.Generator) -> np.ndarray:
    if config.position == "uniform":
        return rng.integers(0, n, count)
    positions = np.floor(rng.exponential(config.front_mean * n, count)).astype(np.int64)
    return np.minimum(positions, n - 1)


def _miss_keys(sorted_unique: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``count`` keys that are not contained in ``sorted_unique``."""
    lo, hi = int(sorted_unique[0]), int(sorted_unique[-1])
    has_gaps = hi - lo + 1 > sorted_unique.size
    misses = np.empty(0, dtype=np.int64)
    while has_gaps and misses.size < count:
        candidates = rng.integers(lo, hi, count, endpoint=True)
        positions = np.searchsorted(sorted_unique, candidates)
        absent = sorted_unique[np.minimum(positions, sorted_unique.size - 1)] != candidates
        misses = np.concatenate([misses, candidates[absent]])
    if not has_gaps:  # dense key range: only keys outside the range can miss
        below = lo - 1 - rng.integers(0, 1000, count)
        above = hi + 1 + rng.integers(0, 1000, count)
        misses = np.where(rng.random(count) < 0.5, below, above)
    return misses[:count]


def random_hit_key(keys: np.ndarray, rng: np.random.Generator) -> int:
    """A key that is contained in ``keys`` (0 for empty data)."""
    if keys.size == 0:
        return 0
    return int(keys[rng.integers(keys.size)])


def random_miss_key(keys: np.ndarray, rng: np.random.Generator, attempts: int = 64) -> int:
    """A key that is **not** contained in ``keys``, preferably inside its value range.

    Each candidate is checked against all keys with one vectorised comparison
    (O(n), a few milliseconds for 10^7 keys), so the result is exact for any
    size. If the value range has no gaps, a key just outside the range is
    returned, which is a miss as well.
    """
    if keys.size == 0:
        return 0
    lo, hi = int(keys.min()), int(keys.max())
    for _ in range(attempts):
        candidate = int(rng.integers(lo, hi, endpoint=True))
        if not np.any(keys == candidate):
            return candidate
    # Dense range (or very unlucky draws): a neighbour of the range is always absent.
    return hi + 1 if hi < np.iinfo(np.int64).max else lo - 1


def generate_queries(data: np.ndarray | LinkedList, config: QueryConfig) -> np.ndarray:
    """Generate ``config.count`` search keys for ``data`` in random order."""
    keys = linked_list_to_array(data) if isinstance(data, LinkedList) else np.asarray(data, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    hit_count = round(config.hit_ratio * config.count) if keys.size > 0 else 0
    miss_count = config.count - hit_count

    hits = keys[_hit_positions(keys.size, config, hit_count, rng)]
    if miss_count == 0:
        misses = np.empty(0, dtype=np.int64)
    elif keys.size == 0:
        misses = rng.integers(-1000, 1000, miss_count)
    else:
        misses = _miss_keys(np.unique(keys), miss_count, rng)
    return rng.permutation(np.concatenate([hits, misses]).astype(np.int64))
