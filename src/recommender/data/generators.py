"""Synthetic datasets with controlled characteristics.

A dataset is described by a :class:`DatasetConfig` along the dimensions of the
task: size, value distribution, degree of sortedness, share of duplicates and
storage structure. Generation is deterministic for a given seed [mytkowicz2009].

Construction in four steps:

1. Draw a pool of ``n - round(p * n)`` **distinct** integer keys from the
   chosen distribution (``p`` = duplicate share).
2. Fill up to ``n`` keys by drawing additional copies from the pool. Thus the
   duplicate share, defined as ``(n - distinct) / n``, is exactly ``p``.
3. Arrange the keys: sorted, nearly sorted (sorted, then ``round(k * n)``
   swaps of two random positions) or unsorted (random permutation).
4. Store them as an array or as a linked list.
"""

from dataclasses import dataclass

import numpy as np

from .linked_list import LinkedList, build_linked_list

DISTRIBUTIONS = ("uniform", "normal", "exponential", "lognormal", "clustered")
ORDERS = ("sorted", "nearly_sorted", "unsorted")
STRUCTURES = ("array", "linked_list")

# Continuous samples are scaled by these factors and rounded down to int64.
# The factors spread the values over a large integer range (up to about 2^52),
# so that rounding creates practically no unintended duplicates.
_SCALE = {
    "uniform": 2.0**52,  # U(0, 1)
    "normal": 2.0**40,  # N(0, 1), |x| < 8 practically always
    "exponential": 2.0**40,  # Exp(1), x < 40 practically always
    "lognormal": 2.0**32,  # LogN(0, 2), x < 2^18 practically always
    "clustered": 2.0**52,  # mixture of narrow normals inside (0, 1)
}

LOGNORMAL_MU = 0.0  # parameters as in [vansandt2019]
LOGNORMAL_SIGMA = 2.0
CLUSTER_COUNT = 10
CLUSTER_STD = 0.001


@dataclass(frozen=True)
class DatasetConfig:
    """Characteristics of a synthetic dataset.

    Attributes:
        n: number of keys.
        distribution: one of :data:`DISTRIBUTIONS`.
        order: one of :data:`ORDERS`.
        swap_fraction: for ``nearly_sorted``: number of random swaps relative to n.
        duplicate_share: share of keys that repeat an earlier key, in [0, 1).
        structure: one of :data:`STRUCTURES`.
        seed: random seed; the same config always yields the same dataset.
    """

    n: int
    distribution: str = "uniform"
    order: str = "sorted"
    swap_fraction: float = 0.01
    duplicate_share: float = 0.0
    structure: str = "array"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must not be negative")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"unknown distribution {self.distribution!r}")
        if self.order not in ORDERS:
            raise ValueError(f"unknown order {self.order!r}")
        if self.structure not in STRUCTURES:
            raise ValueError(f"unknown structure {self.structure!r}")
        if not 0.0 <= self.duplicate_share < 1.0:
            raise ValueError("duplicate_share must lie in [0, 1)")
        if not 0.0 <= self.swap_fraction <= 1.0:
            raise ValueError("swap_fraction must lie in [0, 1]")


def sample_distribution(distribution: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` integer keys from ``distribution`` (may contain duplicates)."""
    if distribution == "uniform":
        x = rng.uniform(0.0, 1.0, size)
    elif distribution == "normal":
        x = rng.normal(0.0, 1.0, size)
    elif distribution == "exponential":
        x = rng.exponential(1.0, size)
    elif distribution == "lognormal":
        x = rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA, size)
    elif distribution == "clustered":
        # Fixed cluster centres (independent of the rng) keep the shape of the
        # distribution identical across seeds; only the members vary.
        centres = np.random.default_rng(12345).uniform(0.05, 0.95, CLUSTER_COUNT)
        x = rng.normal(rng.choice(centres, size), CLUSTER_STD)
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return np.floor(x * _SCALE[distribution]).astype(np.int64)


def distinct_pool(distribution: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exactly ``size`` distinct keys from ``distribution`` (in random order)."""
    pool = np.unique(sample_distribution(distribution, size, rng))
    while pool.size < size:  # replace the rare rounding collisions
        extra = sample_distribution(distribution, size - pool.size, rng)
        pool = np.unique(np.concatenate([pool, extra]))
    return rng.permutation(pool)[:size]


def generate_keys(config: DatasetConfig) -> np.ndarray:
    """Generate the keys of a dataset as an int64 array in the configured order."""
    rng = np.random.default_rng(config.seed)
    distinct = config.n - round(config.duplicate_share * config.n)
    pool = distinct_pool(config.distribution, distinct, rng)
    copies = rng.choice(pool, config.n - distinct) if distinct > 0 else pool[:0]
    keys = np.concatenate([pool, copies])

    if config.order == "unsorted":
        return rng.permutation(keys)
    keys = np.sort(keys)
    if config.order == "nearly_sorted":
        swap_count = round(config.swap_fraction * config.n)
        first = rng.integers(0, config.n, swap_count)
        second = rng.integers(0, config.n, swap_count)
        for i, j in zip(first, second):
            keys[i], keys[j] = keys[j], keys[i]
    return keys


def generate_dataset(config: DatasetConfig) -> np.ndarray | LinkedList:
    """Generate a dataset in the configured storage structure."""
    keys = generate_keys(config)
    if config.structure == "linked_list":
        return build_linked_list(keys, np.random.default_rng(config.seed + 1))
    return keys


def rescale_monotone(keys: np.ndarray, target_max: int) -> np.ndarray:
    """Map keys onto the small range 0 ... ~target_max for display, keeping their structure.

    The generator spreads values over a huge range (up to 2^52, see _SCALE),
    which is correct but hard to read in the user interface. This mapping

    * is linear in the value (the shape of the distribution is kept),
    * is strictly monotone on distinct values: order, sortedness and the
      exact duplicate share stay the same,
    * moves a value up by the minimum needed when rounding would merge two
      distinct values (dense regions of skewed data); the maximum can
      therefore exceed ``target_max`` slightly.
    """
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size == 0:
        return keys.copy()
    unique, inverse = np.unique(keys, return_inverse=True)
    low, span = float(unique[0]), max(float(unique[-1]) - float(unique[0]), 1.0)
    scaled = np.floor((unique.astype(np.float64) - low) / span * target_max).astype(np.int64)
    ranks = np.arange(unique.size, dtype=np.int64)
    # s'_i = max(s_i, s'_{i-1} + 1): keeps distinct values distinct with minimal shifts
    scaled = np.maximum.accumulate(scaled - ranks) + ranks
    return scaled[inverse]
