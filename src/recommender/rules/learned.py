"""Learned selector: a shallow decision tree trained on the measurement grid.

Trained with scikit-learn using the information-gain criterion
(``criterion="entropy"``), following ID3 [quinlan1986]. The depth is limited to
3-4 levels so the tree stays interpretable and can be drawn [vilasboas2021].

Labels are the oracle strategies of the training instances. The tree is
stored as plain arrays (JSON), so the application can use it without
scikit-learn and draw its decision path like the rule book.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .decision import FEATURE_NAMES_DE, DecisionStep, Recommendation
from .strategies import STRATEGIES

# Inputs of the tree: the same information the rule book has (no generator settings).
TREE_FEATURES = [
    "log10_n",
    "log10_queries",
    "is_list",
    "sorted_known",
    "disorder",
    "ks_uniform",
    "duplicate_share",
    "position_front",
    "hit_ratio",
    "log10_linear_scans",
    "queries_per_n",
]
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 20


def train_tree(instances: pd.DataFrame, max_depth: int = MAX_DEPTH, seed: int = 0) -> DecisionTreeClassifier:
    """Fit the decision tree on instances (features -> oracle strategy)."""
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed)
    tree.fit(instances[TREE_FEATURES].to_numpy(), instances["oracle"].to_numpy())
    return tree


@dataclass(frozen=True)
class LearnedSelector:
    """A trained tree as plain arrays; sklearn's convention: go left if x <= threshold."""

    feature: list[int]  # -2 marks a leaf
    threshold: list[float]
    left: list[int]
    right: list[int]
    leaf_class: list[str]
    leaf_samples: list[int]
    leaf_purity: list[float]
    feature_names: list[str]

    @classmethod
    def from_sklearn(cls, tree: DecisionTreeClassifier) -> "LearnedSelector":
        t = tree.tree_
        counts = t.value[:, 0, :]  # class weights (or fractions) per node
        classes = tree.classes_
        return cls(
            feature=[int(f) for f in t.feature],
            threshold=[float(x) for x in t.threshold],
            left=[int(x) for x in t.children_left],
            right=[int(x) for x in t.children_right],
            leaf_class=[str(classes[int(np.argmax(c))]) for c in counts],
            leaf_samples=[int(s) for s in t.n_node_samples],
            leaf_purity=[float(np.max(c) / np.sum(c)) for c in counts],
            feature_names=list(TREE_FEATURES),
        )

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "LearnedSelector":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def is_leaf(self, node: int) -> bool:
        return self.left[node] == -1

    def decide(self, features: dict[str, float]) -> Recommendation:
        """Follow the tree for the given selection features and record the path."""
        node, path = 0, []
        while not self.is_leaf(node):
            name = self.feature_names[self.feature[node]]
            value = features[name]
            threshold = self.threshold[node]
            answer = value <= threshold
            question = f"Ist {FEATURE_NAMES_DE.get(name, name)} ≤ {threshold:.3g}?"
            path.append(DecisionStep(f"node_{node}", question, name, value, threshold, answer))
            node = self.left[node] if answer else self.right[node]
        strategy = self.leaf_class[node]
        reason = (
            f"Der Entscheidungsbaum hat dieses Blatt aus {self.leaf_samples[node]} Trainingsfällen gelernt; "
            f"in {self.leaf_purity[node]:.0%} davon war „{STRATEGIES[strategy].name_de}“ die beste Strategie."
        )
        return Recommendation(strategy, reason, tuple(path), f"node_{node}", ("quinlan1986", "vilasboas2021"))
