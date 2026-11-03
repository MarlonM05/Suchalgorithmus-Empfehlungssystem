import pandas as pd
import pytest

from recommender.benchmark import experiment
from experiments.exp_amortization import _grid_scenarios, summarize_grid, validate_model
from experiments.exp_grid import REPLICATED_CASES, SEED, SEED_REPLICATES, SEED_STEP, configurations
from experiments.exp_size import SIZES as SIZE_SERIES


def test_custom_results_dir_overrides_mode_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["experiment", "--mode", "full", "--results-dir", str(tmp_path)],
    )

    args = experiment.parse_args("test")

    assert args.mode == "full"
    assert experiment.result_path("grid_features", args.mode) == tmp_path / "grid_features.csv"


def test_amortization_plot_marks_both_break_even_intersections(monkeypatch):
    from experiments import exp_amortization

    captured = {}

    def capture_figure(fig, path):
        captured["fig"] = fig
        captured["path"] = path
        return path

    monkeypatch.setattr(exp_amortization.plots, "save_figure", capture_figure)
    summary = pd.DataFrame(
        [
            {
                "n": 1_000,
                "analysis_ns": 10.0,
                "sort_ns": 100.0,
                "hash_build_ns": 50.0,
                "linear_query_ns": 20.0,
                "binary_query_ns": 5.0,
                "hash_query_ns": 2.0,
            }
        ]
    )

    exp_amortization.plot(summary, "quick")

    axes = captured["fig"].axes[0]
    assert len(axes.collections) == 2
    assert [text.get_text() for text in axes.texts] == ["q ≈ 6,7", "q ≈ 2,8"]
    assert len(axes.lines) == 5
    assert captured["fig"].get_size_inches()[0] == pytest.approx(6.3)
    assert captured["path"] == experiment.result_path("amortization", "quick", ".pdf")
    exp_amortization.plots.plt.close(captured["fig"])


def test_full_grid_and_size_series_include_requested_small_sizes_and_replicates():
    cases = configurations("full")
    assert {16, 30, 32} <= set(SIZE_SERIES["full"])
    assert {16, 30, 32} <= {config.n for config, _ in cases}

    for n, distribution, order, duplicate_share, structure in REPLICATED_CASES:
        matching = [
            (config, replicate)
            for config, replicate in cases
            if (
                config.n,
                config.distribution,
                config.order,
                config.duplicate_share,
                config.structure,
            )
            == (n, distribution, order, duplicate_share, structure)
        ]
        assert len(matching) == SEED_REPLICATES
        assert [replicate for _, replicate in matching] == list(range(SEED_REPLICATES))
        assert [config.seed for config, _ in matching] == [
            SEED + SEED_STEP * r for r in range(SEED_REPLICATES)
        ]


def test_amortization_uses_grid_medians_without_irrelevant_sortedness_check():
    features = pd.DataFrame(
        [
            {
                "dataset_id": 12,
                "seed_replicate": 0,
                "n": 1_000,
                "distribution": "uniform",
                "order": "unsorted",
                "duplicate_share": 0.0,
                "structure": "array",
                "seed": 42,
                "analysis_ns": 30.0,
                "swap_fraction": 0.01,
            }
        ]
    )
    components = pd.DataFrame(
        [
            {
                "dataset_id": 12,
                "access": access,
                "component": component,
                "query_id": "" if access == "prep" else "12:hits_uniform",
                "query_seed": 42,
                "ns": value,
            }
            for access, component, value in [
                ("prep", "verify", 10.0),
                ("prep", "sort", 100.0),
                ("prep", "hash_build", 50.0),
                ("hits_uniform", "q_linear", 20.0),
                ("hits_uniform", "q_after_sort_binary_branchless", 5.0),
                ("hits_uniform", "q_hash", 2.0),
            ]
        ]
    )

    summary = summarize_grid(features, components, "quick")

    assert summary.loc[0, "inspection_ns"] == 30
    assert summary.loc[0, "sort_break_even_q"] == pytest.approx(100 / 15)
    assert summary.loc[0, "hash_break_even_q"] == pytest.approx(50 / 18)
    validation = validate_model(summary, "quick")
    assert len(validation) == 3 * 3 * 3
    assert set(validation["strategy"]) == {"linear", "sort_binary", "hash"}
    assert (validation["actual_ns"] > 0).all()
    assert (validation["model_ns"] > 0).all()
    linear_q1 = validation[(validation["strategy"] == "linear") & (validation["query_count"] == 1)]
    assert linear_q1["model_ns"].iloc[0] == pytest.approx(30 + 20)


def test_amortization_fails_when_exact_grid_component_is_missing():
    features = pd.DataFrame(
        [
            {
                "dataset_id": 12,
                "seed_replicate": 0,
                "n": 1_000,
                "distribution": "uniform",
                "order": "unsorted",
                "duplicate_share": 0.0,
                "structure": "array",
                "seed": 42,
                "analysis_ns": 30.0,
                "swap_fraction": 0.01,
            }
        ]
    )
    components = pd.DataFrame(
        columns=["dataset_id", "access", "component", "query_id", "query_seed", "ns"]
    )

    with pytest.raises(ValueError, match="missing exact-match component"):
        summarize_grid(features, components, "quick")


def test_amortization_explains_when_grid_schema_is_stale():
    legacy_features = pd.DataFrame(
        columns=["dataset_id", "seed", "n", "distribution", "order", "duplicate_share", "structure"]
    )

    with pytest.raises(ValueError, match="regenerate them"):
        _grid_scenarios(legacy_features, "full")


def test_amortization_summary_and_validation_include_every_seed_replicate():
    features = pd.DataFrame(
        [
            {
                "dataset_id": 20 + replicate,
                "seed_replicate": replicate,
                "n": 1_000,
                "distribution": "uniform",
                "order": "unsorted",
                "duplicate_share": 0.0,
                "structure": "array",
                "seed": 100 + replicate,
                "analysis_ns": 30.0 + replicate,
                "swap_fraction": 0.01,
            }
            for replicate in range(3)
        ]
    )
    component_values = [
        ("prep", "verify", 10.0),
        ("prep", "sort", 100.0),
        ("prep", "hash_build", 50.0),
        ("hits_uniform", "q_linear", 20.0),
        ("hits_uniform", "q_after_sort_binary_branchless", 5.0),
        ("hits_uniform", "q_hash", 2.0),
    ]
    components = pd.DataFrame(
        [
            {
                "dataset_id": 20 + replicate,
                "access": access,
                "component": component,
                "query_id": "" if access == "prep" else f"{20 + replicate}:hits_uniform",
                "query_seed": 100 + replicate,
                "ns": (
                    value * (replicate + 1)
                    if component in {"sort", "hash_build"}
                    else value
                ),
            }
            for replicate in range(3)
            for access, component, value in component_values
        ]
    )

    scenarios = _grid_scenarios(features, "quick")
    summary = summarize_grid(features, components, "quick")
    validation = validate_model(summary, "quick")

    assert scenarios["seed_replicate"].tolist() == [0, 1, 2]
    assert summary["seed_replicate"].tolist() == [0, 1, 2]
    assert summary["seed_count"].eq(3).all()
    assert summary["sort_break_even_q_seed_min"].to_list() == pytest.approx([100 / 15] * 3)
    assert summary["sort_break_even_q_seed_max"].to_list() == pytest.approx([300 / 15] * 3)
    assert summary["hash_break_even_q_seed_min"].to_list() == pytest.approx([50 / 18] * 3)
    assert summary["hash_break_even_q_seed_max"].to_list() == pytest.approx([150 / 18] * 3)
    assert validation["dataset_id"].nunique() == 3
    assert set(validation["seed"]) == {100, 101, 102}
    assert len(validation) == 3 * 3 * 3 * 3
