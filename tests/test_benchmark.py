"""Tests for the measurement environment (milestone M3)."""

import numpy as np
import pandas as pd
import pytest

from recommender.algorithms import ALGORITHMS, LIST_ALGORITHMS
from recommender.benchmark import (
    Configuration,
    TimingSettings,
    applicable_algorithms,
    bootstrap_median_ci,
    collect_environment,
    geometric_mean,
    read_results,
    run_configurations,
    sort_copy,
    speedups,
    summarize,
    time_call,
    time_search,
    write_results,
)
from recommender.data import DatasetConfig, QueryConfig, generate_keys

FAST = TimingSettings(repetitions=3, warmup=1, min_block_ns=10_000, max_block_queries=1_000, count_queries=10)


# --- Statistics ----------------------------------------------------------------------


def test_bootstrap_ci_contains_median_and_is_deterministic():
    values = np.random.default_rng(0).lognormal(0, 1, 200)
    low, high = bootstrap_median_ci(values)
    assert low <= np.median(values) <= high
    assert (low, high) == bootstrap_median_ci(values)


def test_bootstrap_ci_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    small = np.diff(bootstrap_median_ci(rng.normal(0, 1, 20)))[0]
    large = np.diff(bootstrap_median_ci(rng.normal(0, 1, 2000)))[0]
    assert large < small


def test_geometric_mean():
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    # Baseline independence [fleming1986]: inverting all ratios inverts the mean.
    ratios = np.array([0.5, 3.0, 1.2])
    assert geometric_mean(1 / ratios) == pytest.approx(1 / geometric_mean(ratios))
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])


def test_summarize_and_speedups():
    raw = pd.DataFrame(
        {
            "n": [10] * 6,
            "algorithm": ["binary"] * 3 + ["linear"] * 3,
            "ns_per_query": [10.0, 12.0, 11.0, 5.0, 6.0, 5.5],
        }
    )
    summary = summarize(raw, ["n", "algorithm"])
    binary = summary[summary["algorithm"] == "binary"].iloc[0]
    assert binary["median"] == 11.0
    assert binary["minimum"] == 10.0
    assert binary["repetitions"] == 3
    relative = speedups(summary, ["n"], baseline="binary").set_index("algorithm")
    assert relative.loc["binary", "speedup"] == 1.0
    assert relative.loc["linear", "speedup"] == pytest.approx(2.0)


# --- Environment and result files ------------------------------------------------------


def test_environment_contains_required_fields():
    environment = collect_environment()
    for key in ["date", "cpu", "ram_gb", "os", "python", "numpy", "numba", "git_commit", "power_source", "low_power_mode"]:
        assert environment[key]
    assert environment["low_power_mode"] in {"on", "off", "unknown"}


def test_results_round_trip(tmp_path):
    frame = pd.DataFrame({"n": [1, 2], "ns_per_query": [3.5, 4.5]})
    path = tmp_path / "result.csv"
    write_results(frame, path, {"seed": "42", "cpu": "Test CPU: fast"})
    loaded, metadata = read_results(path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert metadata == {"seed": "42", "cpu": "Test CPU: fast"}


# --- Timing ----------------------------------------------------------------------------


def test_time_search_returns_one_value_per_repetition():
    keys = generate_keys(DatasetConfig(n=1000))
    samples, block = time_search(ALGORITHMS["binary"], keys, keys[:100], FAST, np.random.default_rng(0))
    assert samples.shape == (FAST.repetitions,)
    assert (samples > 0).all()
    assert 1 <= block <= FAST.max_block_queries


def test_slower_algorithm_measures_slower():
    """Sanity check of the harness: linear search on 100 000 keys is far slower than binary search."""
    keys = generate_keys(DatasetConfig(n=100_000))
    rng = np.random.default_rng(0)
    linear, _ = time_search(ALGORITHMS["linear"], keys, keys, FAST, rng)
    binary, _ = time_search(ALGORITHMS["binary"], keys, keys, FAST, rng)
    assert np.median(linear) > 10 * np.median(binary)


def test_time_call_and_sort_copy():
    data = generate_keys(DatasetConfig(n=1000, order="unsorted"))
    samples = time_call(lambda: sort_copy(data), FAST)
    assert samples.shape == (FAST.repetitions,)
    assert np.array_equal(sort_copy(data), np.sort(data))
    assert not np.array_equal(data, np.sort(data))  # input stays unchanged


# --- Runner ----------------------------------------------------------------------------


def test_applicable_algorithms():
    sorted_keys = {a.key for a in applicable_algorithms(DatasetConfig(n=10))}
    assert sorted_keys == set(ALGORITHMS)
    unsorted = {a.key for a in applicable_algorithms(DatasetConfig(n=10, order="unsorted"))}
    assert unsorted == {"linear", "hash"}
    linked = {a.key for a in applicable_algorithms(DatasetConfig(n=10, structure="linked_list"))}
    assert linked == set(LIST_ALGORITHMS)
    linked_unsorted = {a.key for a in applicable_algorithms(DatasetConfig(n=10, order="unsorted", structure="linked_list"))}
    assert linked_unsorted == {"list_linear"}


def test_run_configurations_collects_raw_values():
    configurations = [
        Configuration(DatasetConfig(n=100, seed=s), QueryConfig(count=50, seed=s)) for s in range(2)
    ] + [Configuration(DatasetConfig(n=100, structure="linked_list"), QueryConfig(count=50))]
    frame = run_configurations(configurations, FAST, seed=1, algorithms=["binary", "linear", "list_jump"], progress=False)
    assert set(frame["algorithm"]) == {"binary", "linear", "list_jump"}
    assert len(frame) == (2 * 2 + 1) * FAST.repetitions
    assert {"n", "distribution", "ns_per_query", "comparisons", "accesses", "block_queries"} <= set(frame.columns)
    assert (frame["ns_per_query"] > 0).all()
