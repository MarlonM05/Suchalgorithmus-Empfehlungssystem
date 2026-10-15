"""Tests for the sample-based feature extraction (milestone M2)."""

import numpy as np
import pytest

from recommender.data import DatasetConfig, build_linked_list, generate_dataset, generate_keys
from recommender.features import extract_features, ks_distance_to_uniform, verify_sorted


def as_list(keys: np.ndarray):
    return build_linked_list(keys, np.random.default_rng(0))


# --- KS distance ---------------------------------------------------------------------


def test_ks_of_evenly_spaced_values_is_tiny():
    m = 1000
    assert ks_distance_to_uniform(np.arange(m, dtype=np.int64)) <= 1 / m + 1e-12


def test_ks_of_constant_or_tiny_sample_is_zero():
    assert ks_distance_to_uniform(np.full(10, 7, dtype=np.int64)) == 0.0
    assert ks_distance_to_uniform(np.array([3], dtype=np.int64)) == 0.0


def test_ks_of_skewed_values_is_large():
    skewed = (2 ** np.arange(50)).astype(np.int64)  # exponential growth
    # At 2^44: empirical CDF 45/50 = 0.9, uniform CDF about 1/32 -> distance 0.87
    assert ks_distance_to_uniform(skewed) > 0.85


# --- Feature estimation --------------------------------------------------------------


@pytest.mark.parametrize("structure", ["array", "linked_list"])
def test_sorted_data(structure):
    data = generate_dataset(DatasetConfig(n=100_000, duplicate_share=0.1, structure=structure))
    features = extract_features(data)
    assert features.structure == structure
    assert features.disorder == 0.0
    assert features.looks_sorted
    assert features.duplicate_share == pytest.approx(0.1, abs=0.04)  # about 4 standard errors
    assert features.sample_size == 1000


@pytest.mark.parametrize("structure", ["array", "linked_list"])
def test_unsorted_data(structure):
    data = generate_dataset(DatasetConfig(n=100_000, order="unsorted", structure=structure))
    features = extract_features(data)
    assert 0.43 < features.disorder < 0.57
    assert not features.looks_sorted
    assert features.duplicate_share is None  # not estimable without sorting


def test_nearly_sorted_data():
    data = generate_dataset(DatasetConfig(n=100_000, order="nearly_sorted", swap_fraction=0.05))
    assert 0.0 < extract_features(data).disorder < 0.2


@pytest.mark.parametrize("structure", ["array", "linked_list"])
def test_small_data_is_analysed_exactly(structure):
    keys = generate_keys(DatasetConfig(n=500, order="unsorted", seed=4))
    data = as_list(keys) if structure == "linked_list" else keys
    features = extract_features(data, sample_size=1000)
    assert features.sample_size == 499  # all adjacent pairs
    assert features.disorder == np.count_nonzero(keys[:-1] > keys[1:]) / 499
    assert features.ks_uniform == pytest.approx(ks_distance_to_uniform(keys))


def test_exact_duplicate_share_for_small_sorted_data():
    keys = generate_keys(DatasetConfig(n=500, duplicate_share=0.2))
    assert extract_features(keys).duplicate_share == pytest.approx(0.2)


def test_uniformity_separates_distributions():
    def ks(distribution):
        return extract_features(generate_keys(DatasetConfig(n=100_000, distribution=distribution))).ks_uniform

    assert ks("uniform") < 0.06  # below the 95 % critical value 1.36/sqrt(1000) plus slack
    assert ks("lognormal") > 0.7


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_sizes(n):
    features = extract_features(np.arange(n, dtype=np.int64))
    assert features.n == n
    assert features.disorder == 0.0
    assert features.sample_size == 0


def test_sample_size_is_configurable_and_deterministic():
    keys = generate_keys(DatasetConfig(n=100_000, order="unsorted"))
    small = extract_features(keys, sample_size=100, seed=1)
    assert small.sample_size == 100
    assert small.disorder == extract_features(keys, sample_size=100, seed=1).disorder


def test_analysis_time_is_measured():
    features = extract_features(generate_keys(DatasetConfig(n=10_000)))
    assert features.analysis_time_ns > 0


# --- Full sortedness check -----------------------------------------------------------


@pytest.mark.parametrize("structure", ["array", "linked_list"])
def test_verify_sorted(structure):
    def check(keys):
        data = as_list(keys) if structure == "linked_list" else keys
        return verify_sorted(data)

    sorted_keys = generate_keys(DatasetConfig(n=10_000, duplicate_share=0.5))
    assert check(sorted_keys).is_sorted
    one_swap = sorted_keys.copy()
    one_swap[[100, 9000]] = one_swap[[9000, 100]]
    assert not check(one_swap).is_sorted
    assert check(np.empty(0, dtype=np.int64)).is_sorted
    assert check(np.array([5], dtype=np.int64)).is_sorted
    assert check(sorted_keys).time_ns > 0


def test_sample_can_miss_a_single_violation():
    """The reason for verify_sorted: a sample cannot prove sortedness."""
    keys = np.arange(1_000_000, dtype=np.int64)
    keys[[500_000, 500_001]] = keys[[500_001, 500_000]]
    assert extract_features(keys).looks_sorted  # the sample misses the swap ...
    assert not verify_sorted(keys).is_sorted  # ... the full check does not
