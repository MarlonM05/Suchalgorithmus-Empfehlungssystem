"""Rule-based selector: evaluates the decision graph defined in ``rules.yaml``.

The rules are data, not code: every decision node holds a question, a test
``feature <op> threshold``, the next node for "yes" and "no", a justification
and source keys; every leaf holds the recommended strategy, a one-sentence
justification and source keys. This keeps the rules readable, quotable in the
paper and separate from the program logic.
"""

import operator
from dataclasses import dataclass
from pathlib import Path

import yaml

from .decision import FEATURE_NAMES_DE, DecisionStep, Recommendation
from .strategies import STRATEGIES

DEFAULT_RULES = Path(__file__).with_name("rules.yaml")

OPERATORS = {
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class DecisionNode:
    key: str
    question: str
    feature: str
    op: str
    threshold: float
    yes: str
    no: str
    reason: str
    sources: tuple[str, ...]

    def test(self, value: float) -> bool:
        return OPERATORS[self.op](value, self.threshold)


@dataclass(frozen=True)
class Leaf:
    key: str
    strategy: str
    reason: str
    sources: tuple[str, ...]


class RuleBook:
    """The decision graph of the rule-based selector."""

    def __init__(self, start: str, nodes: dict[str, DecisionNode], leaves: dict[str, Leaf]):
        self.start = start
        self.nodes = nodes
        self.leaves = leaves
        self._validate()

    @classmethod
    def load(cls, path: Path = DEFAULT_RULES) -> "RuleBook":
        """Read and validate a rule file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        nodes = {
            key: DecisionNode(
                key=key,
                question=spec["question"],
                feature=spec["feature"],
                op=spec["op"],
                threshold=float(spec["threshold"]),
                yes=spec["yes"],
                no=spec["no"],
                reason=spec.get("reason", ""),
                sources=tuple(spec.get("sources", [])),
            )
            for key, spec in raw["nodes"].items()
        }
        leaves = {
            key: Leaf(key=key, strategy=spec["strategy"], reason=spec["reason"], sources=tuple(spec.get("sources", [])))
            for key, spec in raw["leaves"].items()
        }
        return cls(raw["start"], nodes, leaves)

    def _validate(self) -> None:
        """Every reference must exist, operators and features must be known, no cycles."""
        targets = set(self.nodes) | set(self.leaves)
        if self.start not in targets:
            raise ValueError(f"start node {self.start!r} does not exist")
        for node in self.nodes.values():
            if node.op not in OPERATORS:
                raise ValueError(f"{node.key}: unknown operator {node.op!r}")
            if node.feature not in FEATURE_NAMES_DE:
                raise ValueError(f"{node.key}: unknown feature {node.feature!r}")
            for target in (node.yes, node.no):
                if target not in targets:
                    raise ValueError(f"{node.key}: unknown target {target!r}")
        for leaf in self.leaves.values():
            if leaf.strategy not in STRATEGIES:
                raise ValueError(f"{leaf.key}: unknown strategy {leaf.strategy!r}")
        self._check_acyclic(self.start, ())

    def _check_acyclic(self, key: str, trail: tuple[str, ...]) -> None:
        if key in trail:
            raise ValueError(f"cycle in rule graph: {' -> '.join(trail + (key,))}")
        if key in self.nodes:
            node = self.nodes[key]
            self._check_acyclic(node.yes, trail + (key,))
            self._check_acyclic(node.no, trail + (key,))

    def decide(self, features: dict[str, float]) -> Recommendation:
        """Follow the graph from the start node for the given selection features."""
        path = []
        key = self.start
        while key in self.nodes:
            node = self.nodes[key]
            value = features[node.feature]
            answer = node.test(value)
            path.append(DecisionStep(node.key, node.question, node.feature, value, node.threshold, answer, node.reason))
            key = node.yes if answer else node.no
        leaf = self.leaves[key]
        return Recommendation(leaf.strategy, leaf.reason, tuple(path), leaf.key, leaf.sources)

    def edges(self) -> list[tuple[str, str, bool]]:
        """All edges ``(from, to, answer)`` of the graph, for drawing it."""
        return [(n.key, n.yes, True) for n in self.nodes.values()] + [(n.key, n.no, False) for n in self.nodes.values()]
