"""Tests for the search algorithms on linked lists (milestone M2)."""

import math

import numpy as np
import pytest

from recommender.algorithms import LIST_ALGORITHMS, SearchAlgorithm, new_counter
from recommender.data import DatasetConfig, build_linked_list, generate_keys

ALL_LIST_ALGORITHMS = list(LIST_ALGORITHMS.values())


def expected_position(keys, key) -> int:
    matches = np.flatnonzero(keys == key)
    return int(matches[0]) if matches.size else -1


def run(algorithm: SearchAlgorithm, keys, key, counted: bool) -> int:
    data = algorithm.prepare(build_linked_list(np.asarray(keys, dtype=np.int64), np.random.default_rng(1)))
    return algorithm.find_counted(data, key).index if counted else algorithm.find(data, key)


@pytest.fixture(params=ALL_LIST_ALGORITHMS, ids=lambda a: a.key)
def algorithm(request) -> SearchAlgorithm:
    return request.param


@pytest.fixture(params=[False, True], ids=["timed", "counted"])
def counted(request) -> bool:
    return request.param


def test_empty_and_single(algorithm, counted):
    assert run(algorithm, [], 1, counted) == -1
    assert run(algorithm, [7], 7, counted) == 0
    assert run(algorithm, [7], 6, counted) == -1
    assert run(algorithm, [7], 8, counted) == -1


@pytest.mark.parametrize("n", [2, 3, 4, 9, 10, 17, 100, 1000])
def test_sorted_lists_match_reference(algorithm, counted, n):
    keys = generate_keys(DatasetConfig(n=n, duplicate_share=0.3, seed=n))
    data = algorithm.prepare(build_linked_list(keys, np.random.default_rng(n)))
    for key in np.concatenate([keys, keys - 1, keys + 1]):
        result = algorithm.find_counted(data, int(key)).index if counted else algorithm.find(data, int(key))
        assert result == expected_position(keys, key)


def test_linear_search_on_unsorted_list(counted):
    linear = LIST_ALGORITHMS["list_linear"]
    keys = generate_keys(DatasetConfig(n=300, order="unsorted", duplicate_share=0.3))
    for key in np.concatenate([keys, keys + 1]):
        assert run(linear, keys, int(key), counted) == expected_position(keys, key)


def test_timed_variant_does_not_count(algorithm):
    data = algorithm.prepare(build_linked_list(np.arange(10), np.random.default_rng(0)))
    ops = new_counter()
    algorithm.search(data, np.int64(5), ops)
    assert ops.tolist() == [0, 0]


def test_jump_search_saves_comparisons_not_node_visits():
    """On a list, jump search needs O(sqrt n) comparisons but O(n) node visits."""
    n = 10_000
    keys = np.arange(n, dtype=np.int64)
    data = LIST_ALGORITHMS["list_jump"].prepare(build_linked_list(keys, np.random.default_rng(0)))
    linear_data = LIST_ALGORITHMS["list_linear"].prepare(build_linked_list(keys, np.random.default_rng(0)))
    step = math.isqrt(n)
    for key in [0, 1234, n - 1, n + 5]:
        jump = LIST_ALGORITHMS["list_jump"].find_counted(data, key)
        assert jump.comparisons <= 2 * step + 2
        assert jump.accesses <= n + step + 2
    jump_far = LIST_ALGORITHMS["list_jump"].find_counted(data, n - 1)
    linear_far = LIST_ALGORITHMS["list_linear"].find_counted(linear_data, n - 1)
    assert jump_far.comparisons < linear_far.comparisons / 10
    assert jump_far.accesses >= linear_far.accesses * 0.9
