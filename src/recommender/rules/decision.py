"""Common result format of both selectors (rule book and learned tree).

Both selectors return a :class:`Recommendation` that contains the chosen
strategy and the **decision path**: the visited nodes with the measured
feature value and the outcome of each test. The user interface draws this
path into the decision graph, so both selectors can be shown the same way.
"""

import math
from dataclasses import dataclass, field

from ..features import DatasetFeatures
from .strategies import Workload

# Features that are estimated on a sample and therefore cost analysis time.
# All other features are known for free (size, user input).
SAMPLED_FEATURES = frozenset({"disorder", "looks_sorted", "ks_uniform", "duplicate_share"})

# German names of the features (user interface, decision graph).
FEATURE_NAMES_DE = {
    "n": "Datengröße n",
    "log10_n": "log₁₀(n)",
    "queries": "Anzahl Anfragen q",
    "log10_queries": "log₁₀(q)",
    "is_list": "Verkettete Liste",
    "sorted_known": "Sortierung garantiert",
    "looks_sorted": "Stichprobe: sortiert",
    "disorder": "Anteil falsch geordneter Paare",
    "ks_uniform": "KS-Abstand zur Gleichverteilung",
    "duplicate_share": "Duplikatanteil",
    "position_front": "Treffer vorne gehäuft",
    "hit_ratio": "Trefferquote",
    "linear_scans": "Erwartete Durchläufe der linearen Suche",
    "log10_linear_scans": "log₁₀(erwartete Durchläufe)",
    "queries_per_n": "Anfragen pro Element (q/n)",
}


def selection_features(features: DatasetFeatures, workload: Workload) -> dict[str, float]:
    """All inputs of the selectors as one flat dictionary of numbers.

    ``duplicate_share`` is -1 if it could not be estimated (unsorted data).
    """
    return {
        "n": float(features.n),
        "log10_n": math.log10(max(features.n, 1)),
        "queries": float(workload.queries),
        "log10_queries": math.log10(workload.queries),
        "is_list": float(features.structure == "linked_list"),
        "sorted_known": float(workload.sorted_known),
        "looks_sorted": float(features.looks_sorted),
        "disorder": features.disorder,
        "ks_uniform": features.ks_uniform,
        "duplicate_share": -1.0 if features.duplicate_share is None else features.duplicate_share,
        "position_front": float(workload.position == "front"),
        "hit_ratio": workload.hit_ratio,
        "linear_scans": workload.linear_scans,
        "log10_linear_scans": math.log10(workload.linear_scans),
        "queries_per_n": workload.queries / max(features.n, 1),
    }


@dataclass(frozen=True)
class DecisionStep:
    """One visited decision node."""

    node: str
    question: str
    feature: str
    value: float
    threshold: float | None
    answer: bool
    reason: str = ""


@dataclass(frozen=True)
class Recommendation:
    """Chosen strategy plus the path that led to it."""

    strategy: str
    reason: str
    path: tuple[DecisionStep, ...]
    leaf: str
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_sampled_features(self) -> bool:
        """True if the path needed a sampled feature, i.e. the analysis cost is due."""
        return any(step.feature in SAMPLED_FEATURES for step in self.path)
