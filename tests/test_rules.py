"""Tests for strategies, rule book, learned tree and evaluation (milestone M4)."""

import math

import numpy as np
import pandas as pd
import pytest

from recommender.data import DatasetConfig, generate_dataset
from recommender.features import DatasetFeatures, extract_features
from recommender.rules import STRATEGIES, RuleBook, Workload, selection_features
from recommender.rules.evaluation import fixed_strategy, oracle, realised, summarize
from recommender.rules.instances import strategy_cost
from recommender.rules.learned import TREE_FEATURES, LearnedSelector, train_tree


def features_for(n=1_000_000, structure="array", disorder=0.0, ks=0.02, duplicates=0.0) -> DatasetFeatures:
    return DatasetFeatures(n=n, structure=structure, sample_size=1000, disorder=disorder, duplicate_share=duplicates if disorder == 0 else None, ks_uniform=ks, analysis_time_ns=50_000)


# --- Workload ----------------------------------------------------------------------------


def test_workload_scan_fraction():
    assert Workload(queries=10).scan_fraction == 0.5
    assert Workload(queries=10, hit_ratio=0.0).scan_fraction == 1.0
    assert Workload(queries=10, position="front").scan_fraction == pytest.approx(0.01)
    assert Workload(queries=10, hit_ratio=0.5).linear_scans == pytest.approx(10 * 0.75)


def test_workload_validation():
    with pytest.raises(ValueError):
        Workload(queries=0)
    with pytest.raises(ValueError):
        Workload(queries=5, position="back")


# --- Rule book ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rulebook() -> RuleBook:
    return RuleBook.load()


def test_every_leaf_is_reachable(rulebook):
    reachable, todo = set(), [rulebook.start]
    while todo:
        key = todo.pop()
        if key in reachable:
            continue
        reachable.add(key)
        if key in rulebook.nodes:
            todo += [rulebook.nodes[key].yes, rulebook.nodes[key].no]
    assert set(rulebook.leaves) <= reachable
    assert set(rulebook.nodes) <= reachable


def test_every_leaf_and_node_has_a_reason(rulebook):
    assert all(leaf.reason for leaf in rulebook.leaves.values())
    assert all(node.reason and node.question for node in rulebook.nodes.values())


@pytest.mark.parametrize(
    "features, workload, expected",
    [
        (features_for(n=10), Workload(queries=5), "linear"),
        (features_for(), Workload(queries=1), "linear"),
        (features_for(), Workload(queries=1, sorted_known=True), "binary_branchless"),
        (features_for(), Workload(queries=100), "binary_branchless"),
        (features_for(), Workload(queries=10_000_000), "hash"),
        (features_for(disorder=0.5), Workload(queries=100), "hash"),
        (features_for(disorder=0.5), Workload(queries=10), "linear"),
        (features_for(disorder=0.5), Workload(queries=100, position="front"), "linear"),
        (features_for(structure="linked_list"), Workload(queries=1), "list_linear"),
        (features_for(structure="linked_list"), Workload(queries=1, sorted_known=True), "list_jump"),
        (features_for(structure="linked_list"), Workload(queries=10), "list_hash"),
    ],
)
def test_rule_decisions(rulebook, features, workload, expected):
    recommendation = rulebook.decide(selection_features(features, workload))
    assert recommendation.strategy == expected
    assert STRATEGIES[recommendation.strategy].structure == features.structure
    assert recommendation.reason


def test_sample_is_only_needed_when_a_sampled_feature_is_used(rulebook):
    cheap = rulebook.decide(selection_features(features_for(), Workload(queries=1)))
    assert not cheap.uses_sampled_features  # decided from n, q and the user's input alone
    needs_sample = rulebook.decide(selection_features(features_for(), Workload(queries=100)))
    assert needs_sample.uses_sampled_features
    assert "looks_sorted" in [step.feature for step in needs_sample.path]


def test_decision_path_is_consistent(rulebook):
    recommendation = rulebook.decide(selection_features(features_for(), Workload(queries=100)))
    keys = [step.node for step in recommendation.path]
    assert keys[0] == rulebook.start
    for step, following in zip(recommendation.path, keys[1:] + [recommendation.leaf]):
        node = rulebook.nodes[step.node]
        assert following == (node.yes if step.answer else node.no)


def test_real_dataset_end_to_end(rulebook):
    data = generate_dataset(DatasetConfig(n=100_000, order="unsorted"))
    recommendation = rulebook.decide(selection_features(extract_features(data), Workload(queries=1000)))
    assert recommendation.strategy == "hash"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"nodes": {"is_list": {"yes": "nowhere"}}}, "unknown target"),
        ({"nodes": {"is_list": {"op": "~"}}}, "unknown operator"),
        ({"nodes": {"is_list": {"feature": "colour"}}}, "unknown feature"),
        ({"leaves": {"leaf_hash": {"strategy": "quantum"}}}, "unknown strategy"),
        ({"nodes": {"list_sorted_known": {"yes": "is_list"}}}, "cycle"),
    ],
)
def test_invalid_rule_files_are_rejected(tmp_path, change, message):
    import yaml

    from recommender.rules.rulebook import DEFAULT_RULES

    raw = yaml.safe_load(DEFAULT_RULES.read_text(encoding="utf-8"))
    for section, entries in change.items():
        for key, fields in entries.items():
            raw[section][key].update(fields)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        RuleBook.load(path)


# --- Cost model --------------------------------------------------------------------------

PREP = {"verify": 1000.0, "sort": 50_000.0, "hash_build": 5_000.0}
PER_QUERY = {"q_linear": 500.0, "q_binary_branchless": 10.0, "q_hash": 2.0, "q_after_sort_binary_branchless": 10.0}


def test_strategy_cost():
    assert strategy_cost(STRATEGIES["linear"], PREP, PER_QUERY, 10, False, False) == 5000.0
    assert math.isnan(strategy_cost(STRATEGIES["binary_branchless"], PREP, PER_QUERY, 10, False, False))
    assert strategy_cost(STRATEGIES["binary_branchless"], PREP, PER_QUERY, 10, False, True) == 1000.0 + 100.0
    assert strategy_cost(STRATEGIES["binary_branchless"], PREP, PER_QUERY, 10, True, True) == 100.0
    assert strategy_cost(STRATEGIES["hash"], PREP, PER_QUERY, 10, False, False) == 5020.0
    assert strategy_cost(STRATEGIES["sort_binary"], PREP, PER_QUERY, 10, False, False) == 50_100.0


# --- Evaluation --------------------------------------------------------------------------


def instance(true_sorted: bool, disorder: float, queries: int = 100) -> pd.Series:
    features = selection_features(features_for(disorder=disorder), Workload(queries=queries))
    costs = {key: strategy_cost(STRATEGIES[key], PREP, PER_QUERY, queries, False, true_sorted) for key in ["linear", "binary_branchless", "hash", "sort_binary"]}
    finite = {k: v for k, v in costs.items() if not math.isnan(v)}
    best = min(finite, key=finite.get)
    return pd.Series({**features, "structure": "array", "true_sorted": true_sorted, "analysis_ns": 50_000.0, "verify_ns": 300.0, **{f"cost_{k}": v for k, v in costs.items()}, "oracle": best, "oracle_cost": finite[best]})


def test_failed_check_falls_back(rulebook):
    """The sample overlooks the disorder: the check fails, its cost is paid and the decision repeated."""
    row = instance(true_sorted=False, disorder=0.0)
    result = realised(row, rulebook.decide)
    assert result["failed_check"]
    assert result["chosen"] == "hash"
    assert result["cost"] == pytest.approx(300.0 + row["cost_hash"])
    assert result["cost_with_analysis"] == pytest.approx(result["cost"] + 50_000.0)


def test_summary_gap_is_one_for_oracle_and_zero_for_sbs():
    instances = pd.DataFrame([instance(True, 0.0, q) for q in (1, 10, 100, 1000)] + [instance(False, 0.5, q) for q in (1, 10, 100, 1000)])
    results = pd.concat([oracle(instances), fixed_strategy(instances, {"array": "linear"}, "SBS")])
    table = summarize(results, instances, "SBS", with_analysis=False).set_index("Selektor")
    assert table.loc["Oracle (VBS)", "Trefferquote"] == 1.0
    assert table.loc["Oracle (VBS)", "Lücke geschlossen (geom.)"] == pytest.approx(1.0)
    assert table.loc["SBS", "Lücke geschlossen (geom.)"] == pytest.approx(0.0)
    assert table.loc["SBS", "Slowdown geom. Mittel"] > 1.0


# --- Learned tree ------------------------------------------------------------------------


def test_learned_tree_matches_sklearn_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({name: rng.uniform(0, 3, 400) for name in TREE_FEATURES})
    frame["oracle"] = np.where(frame["log10_queries"] > 1.5, "hash", "linear")
    tree = train_tree(frame)
    selector = LearnedSelector.from_sklearn(tree)
    selector.to_json(tmp_path / "tree.json")
    loaded = LearnedSelector.from_json(tmp_path / "tree.json")
    predictions = tree.predict(frame[TREE_FEATURES].to_numpy())
    for (_, row), expected in zip(frame.iterrows(), predictions):
        recommendation = loaded.decide({name: row[name] for name in TREE_FEATURES})
        assert recommendation.strategy == expected
        assert recommendation.path  # the root split is always visited
    assert loaded.decide({**{n: 1.0 for n in TREE_FEATURES}, "log10_queries": 2.5}).strategy == "hash"
