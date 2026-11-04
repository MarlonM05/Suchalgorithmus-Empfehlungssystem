"""Tests for the logic behind the user interface (milestone M5):
executor, decision graphs and step-by-step traces."""

import numpy as np
import pytest

from recommender.algorithms import ALGORITHMS, as_search_array
from recommender.data import DatasetConfig, generate_dataset, generate_keys, linked_list_to_array
from recommender.executor import applicable_strategies, compare_all, execute
from recommender.features import extract_features
from recommender.rules import RuleBook, Workload, selection_features
from recommender.rules.learned import TREE_FEATURES, LearnedSelector, train_tree
from recommender.viz.graph import rulebook_dot, tree_dot
from recommender.viz.stepper import MAX_LEARNING_SIZE, TRACERS, trace


def first_occurrence(values, key) -> int:
    matches = np.flatnonzero(np.asarray(values) == key)
    return int(matches[0]) if matches.size else -1


# --- Executor ------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["sorted", "unsorted"])
@pytest.mark.parametrize("structure", ["array", "linked_list"])
def test_every_applicable_strategy_finds_the_first_occurrence(order, structure):
    data = generate_dataset(DatasetConfig(n=2000, order=order, duplicate_share=0.3, structure=structure, seed=5))
    keys_in_order = linked_list_to_array(data) if structure == "linked_list" else data
    for key in [int(keys_in_order[0]), int(keys_in_order[1234]), int(keys_in_order.max()) + 1]:
        expected = first_occurrence(keys_in_order, key)
        results = compare_all(data, key)
        assert {r.strategy for r in results} == {s.key for s in applicable_strategies(data, order == "sorted")}
        for result in results:
            assert result.index == expected, result.strategy
            assert result.query_ns > 0


def test_failed_sortedness_check_skips_the_search():
    data = generate_keys(DatasetConfig(n=100, order="unsorted"))
    result = execute("binary_branchless", data, int(data[0]))
    assert result.check_failed and result.index == -1


def test_guaranteed_sortedness_skips_the_check():
    data = generate_keys(DatasetConfig(n=1000))
    assert execute("binary_branchless", data, int(data[10]), sorted_known=True).check_ns == 0
    assert execute("binary_branchless", data, int(data[10])).check_ns > 0


def test_total_cost_model():
    data = generate_keys(DatasetConfig(n=1000, order="unsorted"))
    result = execute("hash", data, int(data[3]))
    assert result.total_ns(10) == pytest.approx(result.preprocessing_ns + 10 * result.query_ns)


# --- Decision graphs -------------------------------------------------------------------


def test_rulebook_graph_highlights_the_active_path():
    rulebook = RuleBook.load()
    recommendation = rulebook.decide(selection_features(extract_features(generate_keys(DatasetConfig(n=100_000))), Workload(queries=100)))
    dot = rulebook_dot(rulebook, recommendation)
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")
    for key in list(rulebook.nodes) + list(rulebook.leaves):
        assert f'"{key}"' in dot
    highlighted = [line for line in dot.splitlines() if "#2a78d6" in line and "->" not in line]
    assert len(highlighted) == len(recommendation.path) + 1  # visited nodes plus leaf
    assert "Stichprobe: sortiert = ja" in dot  # measured value shown at the visited node


def test_rulebook_graph_without_recommendation():
    dot = rulebook_dot(RuleBook.load())
    assert "#2a78d6" not in dot


def test_tree_graph_highlights_the_active_path():
    rng = np.random.default_rng(0)
    import pandas as pd

    frame = pd.DataFrame({name: rng.uniform(0, 3, 300) for name in TREE_FEATURES})
    frame["oracle"] = np.where(frame["log10_queries"] > 1.5, "hash", "linear")
    selector = LearnedSelector.from_sklearn(train_tree(frame))
    recommendation = selector.decide({name: 2.0 for name in TREE_FEATURES})
    dot = tree_dot(selector, recommendation)
    assert recommendation.leaf in dot
    active_edges = [line for line in dot.splitlines() if "->" in line and "#2a78d6" in line]
    assert len(active_edges) == len(recommendation.path)


# --- Step-by-step traces ---------------------------------------------------------------


@pytest.mark.parametrize("algorithm", list(TRACERS))
def test_traces_end_like_the_compiled_algorithms(algorithm):
    rng = np.random.default_rng(1)
    compiled = ALGORITHMS[algorithm]
    for n in [0, 1, 2, 5, 9, 16, 33, MAX_LEARNING_SIZE]:
        for kind in ("dense", "duplicates", "skewed"):
            if kind == "dense":
                values = np.sort(rng.integers(0, 3 * n + 1, n))
            elif kind == "duplicates":
                values = np.sort(rng.integers(0, max(1, n // 4), n))
            else:
                values = np.sort((2 ** rng.uniform(0, 40, n)).astype(np.int64))
            arr = as_search_array(values)
            for key in list(arr[: min(n, 12)]) + [-1, int(arr.max()) + 1 if n else 5, int(arr[n // 2]) + 1 if n else 0]:
                steps = trace(algorithm, arr, key)
                assert steps[-1].done
                assert not any(step.done for step in steps[:-1])
                assert steps[-1].result == compiled.find(arr, int(key)), (n, kind, key)


def test_trace_steps_are_explained_and_ranges_shrink():
    arr = as_search_array(range(0, 128, 2))
    steps = trace("binary", arr, 50)
    assert all(step.text for step in steps)
    widths = [step.hi - step.lo for step in steps[:-1]]
    assert widths == sorted(widths, reverse=True)
    assert steps[-1].result == 25
