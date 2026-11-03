"""Experiment: evaluation of the recommender (M4).

Reads the measurement grid (exp_grid.py), builds the instances and compares:

* Oracle / virtual best solver (VBS) and single best solver (SBS) [kerschke2019]
* the baselines "always linear search" and "always binary search"
* the rule book (rules.yaml)
* the learned decision tree
* the sampling meta-algorithm of [vansandt2019] (sorted arrays only)

Fair split: the datasets of the grid are divided into a calibration half and
a test half (stratified by size and structure, grouped by dataset, so no
dataset appears in both). The tree is trained, the single best solver chosen
and the rule thresholds checked only on the calibration half; all measures
are reported on the test half. Every measure is given with and without the
cost of the feature analysis.

Outputs (results/<mode>/):
    recommender_summary.csv / .tex       measures per selector and scope
    recommender_sampling.csv / .tex      comparison on sorted arrays incl. meta-algorithm
    recommender_choices.csv              chosen vs. best strategy on the test half (counts)
    recommender_split.csv                which dataset belongs to which half
    recommender_slowdown.pdf             distribution of the slowdowns
    learned_tree.json / .txt / .pdf      tree trained on the calibration half (the one
                                         evaluated here and used by the app, A30/A38)
"""

import sys

import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree

from recommender.benchmark.experiment import SEED, load_results, parse_args, result_path
from recommender.rules import RuleBook
from recommender.rules.decision import FEATURE_NAMES_DE
from recommender.rules.evaluation import always_binary, evaluate, fixed_strategy, oracle, single_best, summarize
from recommender.rules.instances import build_instances
from recommender.rules.learned import TREE_FEATURES, LearnedSelector, train_tree
from recommender.viz import plots

SBS = "Single Best (SBS)"
RULES = "Regelwerk"
TREE = "Entscheidungsbaum"
META = "Stichproben-Meta-Algorithmus"


def split_datasets(features: pd.DataFrame) -> pd.Series:
    """Assign every dataset to "Kalibrierung" or "Test", half of each (n, structure) stratum."""
    rng = np.random.default_rng(SEED)
    half = pd.Series("Test", index=features["dataset_id"].to_numpy(), name="half")
    for _, stratum in features.groupby(["n", "structure"]):
        ids = rng.permutation(stratum["dataset_id"].to_numpy())
        half.loc[ids[: len(ids) // 2]] = "Kalibrierung"
    return half


def sampling_meta(instances: pd.DataFrame, sampling: pd.DataFrame) -> pd.DataFrame:
    """Meta-algorithm of [vansandt2019] on sorted arrays with guaranteed sortedness."""
    subset = instances[(instances["structure"] == "array") & (instances["sorted_known"] == 1.0)]
    trials = sampling.groupby(["dataset_id", "access"])
    rows = []
    for index, row in subset.iterrows():
        group = trials.get_group((row["dataset_id"], row["access"]))
        costs = [probe + float(row[f"cost_{chosen}"]) for chosen, probe in zip(group["chosen"], group["probe_ns"])]
        rows.append(
            {
                "selector": META,
                "chosen": group["chosen"].mode().iloc[0],
                "cost": float(np.mean(costs)),
                "cost_with_analysis": float(np.mean(costs)),
                "used_sample": False,
                "wrong_structure": False,
                "failed_check": False,
            }
        )
    return pd.DataFrame(rows, index=subset.index)


def write_table(frame: pd.DataFrame, name: str, mode: str) -> None:
    frame.to_csv(result_path(name, mode), index=False)
    percent = [c for c in frame.columns if c in ("Trefferquote", "≤ 1.1× Optimum", "Stichprobe genutzt") or c.startswith("Lücke")]
    numbers = [c for c in frame.columns if c.startswith("Slowdown")]
    styler = frame.style.hide(axis="index").format("{:.0%}", subset=percent, na_rep="–").format("{:.2f}", subset=numbers)
    result_path(name, mode, ".tex").write_text(styler.to_latex(hrules=True), encoding="utf-8")


def plot_slowdowns(results: pd.DataFrame, instances: pd.DataFrame, mode: str) -> None:
    styles = {
        RULES: {"color": plots.PALETTE[0], "linestyle": "-"},
        TREE: {"color": plots.PALETTE[1], "linestyle": "-"},
        SBS: {"color": plots.INK_MUTED, "linestyle": "--"},
        "Immer lineare Suche": {"color": plots.INK_MUTED, "linestyle": ":"},
    }
    plots.apply_style()
    fig, axes = plots.plt.subplots()
    for name, style in styles.items():
        group = results[results["selector"] == name]
        slowdown = np.sort(group["cost_with_analysis"].to_numpy() / instances.loc[group.index, "oracle_cost"].to_numpy())
        share = np.arange(1, slowdown.size + 1) / slowdown.size
        axes.step(slowdown, share, where="post", label=name, linewidth=1.5, **style)
    axes.set_xscale("log")
    axes.set_xlim(left=1)
    axes.set_xlabel("Slowdown gegenüber dem Oracle (inkl. Analysekosten)")
    axes.set_ylabel("Anteil der Instanzen")
    axes.set_title("Wie weit liegen die Empfehlungen vom Optimum entfernt?", loc="left")
    plots.legend_below(fig, axes, columns=2)
    plots.save_figure(fig, result_path("recommender_slowdown", mode, ".pdf"))


def export_tree(calibration: pd.DataFrame, mode: str) -> None:
    """Export the tree trained on the calibration half (the one evaluated and used by the app)."""
    tree = train_tree(calibration)
    LearnedSelector.from_sklearn(tree).to_json(result_path("learned_tree", mode, ".json"))
    names = [FEATURE_NAMES_DE[f] for f in TREE_FEATURES]
    result_path("learned_tree", mode, ".txt").write_text(export_text(tree, feature_names=names), encoding="utf-8")
    plots.apply_style()
    fig, axes = plots.plt.subplots(figsize=(20, 7))
    labels = plot_tree(tree, feature_names=names, class_names=list(tree.classes_), impurity=False, fontsize=6, ax=axes, rounded=True)
    for label in labels:  # drop the class-count vector, keep condition, samples and class
        label.set_text("\n".join(line for line in label.get_text().split("\n") if not line.startswith("value")))
    plots.save_figure(fig, result_path("learned_tree", mode, ".pdf"))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    features = load_results("grid_features", args.mode)
    components = load_results("grid_components", args.mode)
    sampling = load_results("grid_sampling", args.mode)
    all_instances = build_instances(features, components)
    half = split_datasets(features)
    half.rename_axis("dataset_id").reset_index().to_csv(result_path("recommender_split", args.mode), index=False)
    calibration = all_instances[all_instances["dataset_id"].map(half) == "Kalibrierung"]
    instances = all_instances[all_instances["dataset_id"].map(half) == "Test"]
    print(f"{len(all_instances)} instances from {len(features)} datasets; calibration {len(calibration)}, test {len(instances)}", flush=True)

    sbs = single_best(calibration)
    tree = LearnedSelector.from_sklearn(train_tree(calibration))
    print(f"single best solver (calibration): {sbs}")
    results = pd.concat(
        [
            oracle(instances),
            fixed_strategy(instances, sbs, SBS),
            fixed_strategy(instances, {"array": "linear", "linked_list": "list_linear"}, "Immer lineare Suche"),
            always_binary(instances),
            evaluate(instances, RuleBook.load().decide, RULES),
            evaluate(instances, tree.decide, TREE),
        ]
    )

    scopes = {
        "alle": instances.index,
        "Array": instances.index[instances["structure"] == "array"],
        "Liste": instances.index[instances["structure"] == "linked_list"],
    }
    tables = []
    for scope, index in scopes.items():
        part = results[results.index.isin(index)]
        for with_analysis in (False, True):
            table = summarize(part, instances, SBS, with_analysis)
            table.insert(0, "Analysekosten", "ja" if with_analysis else "nein")
            table.insert(0, "Bereich", scope)
            tables.append(table)
    summary = pd.concat(tables, ignore_index=True)
    write_table(summary, "recommender_summary", args.mode)

    meta = sampling_meta(instances, sampling)
    subset = results[results.index.isin(meta.index)]
    write_table(summarize(pd.concat([subset, meta]), instances, SBS, with_analysis=True), "recommender_sampling", args.mode)

    choices = results[results["selector"].isin([RULES, TREE])].assign(oracle=instances.loc[results.index[results["selector"].isin([RULES, TREE])], "oracle"].to_numpy())
    choices.groupby(["selector", "oracle", "chosen"]).size().rename("count").reset_index().to_csv(result_path("recommender_choices", args.mode), index=False)

    plot_slowdowns(results, instances, args.mode)
    export_tree(calibration, args.mode)

    pd.set_option("display.width", 250)
    columns = ["Bereich", "Analysekosten", "Selektor", "Trefferquote", "≤ 1.1× Optimum", "Slowdown geom. Mittel", "Slowdown 95 %", "Lücke geschlossen (geom.)"]
    print(summary[summary["Bereich"] == "alle"][columns].round(3).to_string(index=False))
    print(f"done: {result_path('recommender_summary', args.mode).parent / 'recommender_*'} and learned_tree.*")


if __name__ == "__main__":
    main()
