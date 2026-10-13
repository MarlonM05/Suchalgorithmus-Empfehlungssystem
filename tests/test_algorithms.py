"""Correctness tests for all search algorithms (milestone M1).

Every test runs for every algorithm and for both compiled variants (timing
variant without counters, counting variant). Reference for sorted data is
``np.searchsorted(side="left")``, which returns the lower bound and therefore
the first occurrence of a key.
"""

import math

import numpy as np
import pytest

from recommender.algorithms import ALGORITHMS, SearchAlgorithm, as_search_array, new_counter
from recommender.algorithms.hashing import build_hash_table
from recommender.algorithms.interpolation import GUARD_SIZE

INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max

ALL_ALGORITHMS = list(ALGORITHMS.values())
UNSORTED_ALGORITHMS = [a for a in ALL_ALGORITHMS if not a.requires_sorted]


def search(algorithm: SearchAlgorithm, values, key, counted: bool) -> int:
    """Prepare ``values`` and run one search with the selected variant."""
    data = algorithm.prepare(as_search_array(values))
    if counted:
        return algorithm.find_counted(data, key).index
    return algorithm.find(data, key)


def expected_index(values, key) -> int:
    """Reference answer: index of the first occurrence of ``key`` or -1."""
    matches = np.flatnonzero(np.asarray(values, dtype=np.int64) == key)
    return int(matches[0]) if matches.size else -1


def searchsorted_reference(arr: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Expected answers for sorted ``arr``, computed with np.searchsorted."""
    positions = np.searchsorted(arr, keys, side="left")
    inside = positions < arr.size
    found = np.zeros(keys.size, dtype=bool)
    found[inside] = arr[positions[inside]] == keys[inside]
    return np.where(found, positions, -1)


@pytest.fixture(params=ALL_ALGORITHMS, ids=lambda a: a.key)
def algorithm(request) -> SearchAlgorithm:
    return request.param


@pytest.fixture(params=[False, True], ids=["timed", "counted"])
def counted(request) -> bool:
    return request.param


# --- Basic cases --------------------------------------------------------------


def test_hit(algorithm, counted):
    values = [2, 5, 8, 13, 21, 34, 55]
    for i, key in enumerate(values):
        assert search(algorithm, values, key, counted) == i


@pytest.mark.parametrize("key", [-10, 1, 6, 30, 100], ids=["below", "below_min", "gap", "gap2", "above"])
def test_miss(algorithm, counted, key):
    assert search(algorithm, [2, 5, 8, 13, 21, 34, 55], key, counted) == -1


def test_empty_array(algorithm, counted):
    assert search(algorithm, [], 42, counted) == -1


@pytest.mark.parametrize("key, expected", [(7, 0), (3, -1), (9, -1)], ids=["hit", "below", "above"])
def test_single_element(algorithm, counted, key, expected):
    assert search(algorithm, [7], key, counted) == expected


def test_duplicates_return_first_occurrence(algorithm, counted):
    values = [1, 3, 3, 3, 3, 7, 9, 9, 12, 12, 12, 12, 12, 20]
    for key in set(values):
        assert search(algorithm, values, key, counted) == expected_index(values, key)


def test_all_elements_equal(algorithm, counted):
    values = [5] * 100
    assert search(algorithm, values, 5, counted) == 0
    assert search(algorithm, values, 4, counted) == -1
    assert search(algorithm, values, 6, counted) == -1


def test_extreme_values(algorithm, counted):
    values = [INT64_MIN, INT64_MIN + 1, -1, 0, 1, INT64_MAX - 1, INT64_MAX]
    for i, key in enumerate(values):
        assert search(algorithm, values, key, counted) == i
    assert search(algorithm, values[1:], INT64_MIN, counted) == -1
    assert search(algorithm, values[:-1], INT64_MAX, counted) == -1


# --- Randomised comparison with np.searchsorted -----------------------------------

# Sizes around powers of two and Fibonacci numbers hit the boundary cases of
# binary, exponential and Fibonacci search.
SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 20, 21, 22, 33, 34, 35, 63, 64, 65, 100, 144, 1000, 4097]


def random_sorted_array(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    """Sorted test arrays with different value distributions."""
    if kind == "uniform":
        values = rng.integers(-(10**6), 10**6, size=n)
    elif kind == "duplicates":  # few distinct values -> long runs of duplicates
        values = rng.integers(0, max(1, n // 10), size=n)
    elif kind == "skewed":  # exponential growth: worst case for interpolation
        values = np.floor(np.exp(rng.uniform(0, 40, size=n)))
    elif kind == "lognormal":  # mu = 0, sigma = 2 as in [vansandt2019]
        values = np.floor(rng.lognormal(0.0, 2.0, size=n) * 1000)
    else:
        raise ValueError(kind)
    return np.sort(as_search_array(values))


@pytest.mark.parametrize("kind", ["uniform", "duplicates", "skewed", "lognormal"])
def test_matches_searchsorted(algorithm, counted, kind):
    rng = np.random.default_rng(20261013)
    for n in SIZES:
        arr = random_sorted_array(rng, n, kind)
        # Every stored key, every gap between neighbours, and keys outside the range.
        keys = np.concatenate([arr, arr - 1, arr + 1, [arr[0] - 100, arr[-1] + 100]])
        expected = searchsorted_reference(arr, keys)
        data = algorithm.prepare(arr)
        if counted:
            actual = [algorithm.find_counted(data, int(k)).index for k in keys]
        else:
            actual = [algorithm.find(data, int(k)) for k in keys]
        assert actual == expected.tolist(), f"n={n}, kind={kind}"


@pytest.mark.parametrize("unsorted_algorithm", UNSORTED_ALGORITHMS, ids=lambda a: a.key)
def test_unsorted_data(unsorted_algorithm, counted):
    rng = np.random.default_rng(7)
    for n in [1, 2, 10, 100, 1000]:
        values = rng.integers(0, n, size=n)  # unsorted, with duplicates
        for key in range(-1, n + 1):
            assert search(unsorted_algorithm, values, key, counted) == expected_index(values, key)


# --- Operation counting -----------------------------------------------------------


def test_timed_variant_does_not_count(algorithm):
    data = algorithm.prepare(as_search_array([1, 2, 3, 4, 5]))
    ops = new_counter()
    algorithm.search(data, np.int64(4), ops)
    assert ops.tolist() == [0, 0]


def test_counted_variant_counts(algorithm):
    data = algorithm.prepare(as_search_array([1, 2, 3, 4, 5]))
    result = algorithm.find_counted(data, 4)
    assert result.index == 3
    assert result.comparisons >= 1
    assert result.accesses >= 1


def test_linear_search_counts_every_element():
    linear = ALGORITHMS["linear"]
    data = linear.prepare(as_search_array(range(100)))
    assert linear.find_counted(data, 41).comparisons == 42
    assert linear.find_counted(data, -5).comparisons == 100


@pytest.mark.parametrize("key", ["binary", "binary_branchless", "fibonacci"])
def test_logarithmic_comparison_bound(key):
    """Binary and Fibonacci search need O(log n) comparisons.

    Fibonacci search needs at most about log_phi(n) = 1.44 log2(n) probes,
    plus the final equality check.
    """
    algorithm = ALGORITHMS[key]
    for n in [10, 1000, 100_000]:
        arr = as_search_array(range(0, 2 * n, 2))
        bound = math.ceil(1.45 * math.log2(n + 1)) + 2
        for k in [-1, 0, 1, n, 2 * n - 2, 2 * n]:
            assert algorithm.find_counted(arr, k).comparisons <= bound


def test_jump_search_comparison_bound():
    """Jump search needs at most about 2 * sqrt(n) comparisons."""
    jump = ALGORITHMS["jump"]
    n = 10_000
    arr = as_search_array(range(n))
    worst = max(jump.find_counted(arr, k).comparisons for k in range(0, n, 37))
    assert worst <= 2 * math.isqrt(n) + 2


def test_exponential_search_is_cheap_near_the_front():
    exponential = ALGORITHMS["exponential"]
    arr = as_search_array(range(1_000_000))
    assert exponential.find_counted(arr, 3).comparisons <= 6
    assert exponential.find_counted(arr, 999_999).comparisons > 20


def test_interpolation_beats_binary_on_uniform_data():
    """On uniform data interpolation needs fewer probes than binary search [peterson1957]."""
    rng = np.random.default_rng(1)
    arr = np.sort(as_search_array(rng.integers(0, 10**12, size=1_000_000)))
    keys = rng.choice(arr, size=200)
    interpolation, binary = ALGORITHMS["interpolation"], ALGORITHMS["binary"]
    mean_interpolation = np.mean([interpolation.find_counted(arr, int(k)).comparisons for k in keys])
    mean_binary = np.mean([binary.find_counted(arr, int(k)).comparisons for k in keys])
    assert mean_interpolation < mean_binary


def test_interpolation_guard_switches_to_sequential_scan():
    """Arrays smaller than the guard window are scanned sequentially right away."""
    interpolation = ALGORITHMS["interpolation"]
    arr = as_search_array(range(GUARD_SIZE))
    result = interpolation.find_counted(arr, GUARD_SIZE - 2)
    assert result.index == GUARD_SIZE - 2
    assert result.comparisons <= GUARD_SIZE + 3


# --- Hash table -------------------------------------------------------------------


def test_hash_table_load_factor_at_most_one_half():
    for n in [0, 1, 5, 1000, 4097]:
        slot_keys, slot_index, shift = build_hash_table(as_search_array(range(n)))
        capacity = slot_keys.size
        assert capacity & (capacity - 1) == 0, "capacity must be a power of two"
        assert capacity >= 2 * n
        assert shift == 64 - int(math.log2(capacity))
        assert np.count_nonzero(slot_index != -1) == n


def test_hash_lookup_is_constant_on_average():
    hashing = ALGORITHMS["hash"]
    rng = np.random.default_rng(3)
    values = rng.permutation(1_000_000)
    table = hashing.prepare(values)
    probes = [hashing.find_counted(table, int(k)).accesses for k in rng.integers(0, 2_000_000, size=1000)]
    assert np.mean(probes) < 3
