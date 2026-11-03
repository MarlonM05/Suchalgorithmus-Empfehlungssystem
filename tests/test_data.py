"""Tests for dataset generators, query generation and the linked list (milestone M2)."""

import numpy as np
import pytest

from recommender.data import (
    DISTRIBUTIONS,
    ORDERS,
    DatasetConfig,
    LinkedList,
    QueryConfig,
    build_linked_list,
    generate_dataset,
    generate_keys,
    generate_queries,
    linked_list_to_array,
)
from recommender.features import ks_distance_to_uniform


def out_of_order_pairs(keys: np.ndarray) -> int:
    return int(np.count_nonzero(keys[:-1] > keys[1:]))


# --- DatasetConfig -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": 10, "distribution": "pareto"},
        {"n": 10, "order": "reversed"},
        {"n": 10, "structure": "tree"},
        {"n": 10, "duplicate_share": 1.0},
        {"n": 10, "swap_fraction": 1.5},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


# --- Keys ----------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("order", ORDERS)
def test_shape_and_dtype(distribution, order):
    keys = generate_keys(DatasetConfig(n=1000, distribution=distribution, order=order))
    assert keys.shape == (1000,)
    assert keys.dtype == np.int64


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_sizes(n):
    assert generate_keys(DatasetConfig(n=n)).size == n
    assert len(generate_dataset(DatasetConfig(n=n, structure="linked_list"))) == n


def test_same_seed_same_data_different_seed_different_data():
    config = DatasetConfig(n=1000, distribution="normal", order="unsorted", seed=5)
    assert np.array_equal(generate_keys(config), generate_keys(config))
    other = DatasetConfig(n=1000, distribution="normal", order="unsorted", seed=6)
    assert not np.array_equal(generate_keys(config), generate_keys(other))


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_sorted_order(distribution):
    keys = generate_keys(DatasetConfig(n=10_000, distribution=distribution, order="sorted"))
    assert out_of_order_pairs(keys) == 0


def test_unsorted_order_is_random():
    keys = generate_keys(DatasetConfig(n=10_000, order="unsorted"))
    assert 0.45 < out_of_order_pairs(keys) / (keys.size - 1) < 0.55


def test_nearly_sorted_order():
    """Each of the 100 swaps breaks at most 4 adjacent pairs."""
    keys = generate_keys(DatasetConfig(n=10_000, order="nearly_sorted", swap_fraction=0.01))
    assert 0 < out_of_order_pairs(keys) <= 4 * 100


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_duplicate_share_is_exact(distribution, share):
    n = 10_000
    keys = generate_keys(DatasetConfig(n=n, distribution=distribution, duplicate_share=share))
    assert (n - np.unique(keys).size) / n == pytest.approx(share)


def test_actual_duplicate_share_reflects_rounding_for_small_datasets():
    keys = generate_keys(DatasetConfig(n=16, duplicate_share=0.1))
    assert 16 - np.unique(keys).size == 2
    assert (16 - np.unique(keys).size) / 16 == pytest.approx(0.125)


def test_distribution_shapes():
    def ks(distribution):
        return ks_distance_to_uniform(generate_keys(DatasetConfig(n=10_000, distribution=distribution)))

    assert ks("uniform") < 0.03
    assert ks("lognormal") > 0.7  # heavily skewed [vansandt2019]
    assert ks("uniform") < ks("normal") < ks("exponential") < ks("lognormal")


def test_normal_distribution_is_centred():
    keys = generate_keys(DatasetConfig(n=10_000, distribution="normal"))
    assert abs(np.median(keys)) < 0.05 * np.std(keys)


def test_clustered_distribution_has_gaps():
    """Most of the value range of clustered data is empty."""
    keys = generate_keys(DatasetConfig(n=10_000, distribution="clustered"))
    counts, _ = np.histogram(keys, bins=100)
    assert np.count_nonzero(counts == 0) > 50


# --- Linked list ---------------------------------------------------------------------


def test_linked_list_round_trip():
    keys = generate_keys(DatasetConfig(n=1000, order="unsorted", duplicate_share=0.1))
    linked_list = build_linked_list(keys, np.random.default_rng(0))
    assert np.array_equal(linked_list_to_array(linked_list), keys)


def test_linked_list_nodes_are_scattered_in_memory():
    linked_list = build_linked_list(np.arange(1000), np.random.default_rng(0))
    assert not np.array_equal(linked_list.values, np.arange(1000))
    # Every slot is reached exactly once when following the pointers.
    seen, slot = set(), linked_list.head
    while slot != -1:
        seen.add(int(slot))
        slot = linked_list.next_index[slot]
    assert seen == set(range(1000))


def test_generate_dataset_linked_list():
    config = DatasetConfig(n=500, structure="linked_list", seed=3)
    linked_list = generate_dataset(config)
    assert isinstance(linked_list, LinkedList)
    assert np.array_equal(linked_list_to_array(linked_list), generate_keys(config))


def test_empty_linked_list():
    linked_list = build_linked_list(np.empty(0, dtype=np.int64), np.random.default_rng(0))
    assert linked_list.head == -1
    assert linked_list_to_array(linked_list).size == 0


# --- Queries -------------------------------------------------------------------------


def test_invalid_query_config_is_rejected():
    with pytest.raises(ValueError):
        QueryConfig(count=10, hit_ratio=1.5)
    with pytest.raises(ValueError):
        QueryConfig(count=10, position="back")


@pytest.mark.parametrize("hit_ratio", [0.0, 0.3, 1.0])
def test_hit_ratio_is_exact(hit_ratio):
    keys = generate_keys(DatasetConfig(n=10_000, distribution="clustered", duplicate_share=0.1))
    queries = generate_queries(keys, QueryConfig(count=1000, hit_ratio=hit_ratio))
    assert queries.size == 1000
    assert np.count_nonzero(np.isin(queries, keys)) == round(hit_ratio * 1000)


def test_misses_fall_into_gaps():
    keys = generate_keys(DatasetConfig(n=10_000))
    misses = generate_queries(keys, QueryConfig(count=500, hit_ratio=0.0))
    assert not np.isin(misses, keys).any()
    assert (misses >= keys.min()).all() and (misses <= keys.max()).all()


def test_misses_on_dense_data_lie_outside():
    keys = np.arange(100, dtype=np.int64)  # no gaps
    misses = generate_queries(keys, QueryConfig(count=200, hit_ratio=0.0))
    assert ((misses < 0) | (misses > 99)).all()


def test_front_positions_are_concentrated():
    keys = np.arange(100_000, dtype=np.int64)  # key == position
    queries = generate_queries(keys, QueryConfig(count=5000, position="front"))
    assert np.median(queries) < 0.01 * keys.size
    assert np.quantile(queries, 0.99) < 0.05 * keys.size


def test_uniform_positions_are_spread():
    keys = np.arange(100_000, dtype=np.int64)
    queries = generate_queries(keys, QueryConfig(count=5000, position="uniform"))
    assert 0.4 * keys.size < np.median(queries) < 0.6 * keys.size


def test_queries_for_empty_data_are_misses():
    queries = generate_queries(np.empty(0, dtype=np.int64), QueryConfig(count=10, hit_ratio=1.0))
    assert queries.size == 10


def test_queries_for_linked_list_use_list_positions():
    config = DatasetConfig(n=10_000, structure="linked_list")
    linked_list = generate_dataset(config)
    queries = generate_queries(linked_list, QueryConfig(count=500, position="front"))
    keys = generate_keys(config)  # sorted keys in list order
    positions = np.searchsorted(keys, queries)
    assert np.median(positions) < 0.01 * keys.size


def test_queries_are_deterministic():
    keys = generate_keys(DatasetConfig(n=1000))
    config = QueryConfig(count=100, hit_ratio=0.5, seed=9)
    assert np.array_equal(generate_queries(keys, config), generate_queries(keys, config))
