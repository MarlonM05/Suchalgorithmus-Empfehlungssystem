"""Decision graphs as Graphviz DOT source, with the active path highlighted.

Both selectors are drawn the same way: decision nodes as boxes with the
question, leaves as rounded boxes with the recommended strategy. Nodes and
edges on the decision path of the current recommendation are coloured; every
visited node additionally shows the measured feature value and the answer.

The DOT source is rendered in the app with ``st.graphviz_chart`` (in the
browser, no local Graphviz needed) and can be exported for the paper.
"""

from ..rules.decision import FEATURE_NAMES_DE, Recommendation
from ..rules.learned import LearnedSelector
from ..rules.rulebook import RuleBook
from ..rules.strategies import STRATEGIES
from .formatting import german

ACTIVE = "#2a78d6"  # blue: the active path
ACTIVE_FILL = "#e3eefb"
INACTIVE = "#898781"  # muted ink for the rest of the graph
INACTIVE_FILL = "#ffffff"
LEAF_FILL = "#f4f3ef"
FONT = "Helvetica"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _format_value(feature: str, value: float) -> str:
    if feature in ("is_list", "sorted_known", "looks_sorted", "position_front"):
        return "ja" if value >= 0.5 else "nein"
    return german(value)


def _wrap(text: str, width: int = 34) -> str:
    """Break long questions into lines (Graphviz does not wrap)."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\\n".join(_escape(part) for part in lines)


def _header() -> list[str]:
    return [
        "digraph Entscheidungsweg {",
        "  rankdir=TB;",
        f'  graph [fontname="{FONT}", nodesep=0.25, ranksep=0.35];',
        f'  node [fontname="{FONT}", fontsize=10, shape=box, style="filled", color="{INACTIVE}", fontcolor="#52514e", fillcolor="{INACTIVE_FILL}"];',
        f'  edge [fontname="{FONT}", fontsize=9, color="{INACTIVE}", fontcolor="#52514e"];',
    ]


def _node_style(active: bool, leaf: bool) -> str:
    fill = ACTIVE_FILL if active else (LEAF_FILL if leaf else INACTIVE_FILL)
    shape = 'shape=box, style="rounded,filled"' if leaf else 'shape=box, style="filled"'
    if active:
        return f'{shape}, color="{ACTIVE}", penwidth=2.2, fontcolor="#0b0b0b", fillcolor="{fill}"'
    return f'{shape}, fillcolor="{fill}"'


def _edge_style(active: bool) -> str:
    return f'color="{ACTIVE}", fontcolor="{ACTIVE}", penwidth=2.2' if active else ""


def path_dot(recommendation: Recommendation, alternatives: dict[str, str]) -> str:
    """Compact view: only the decision path, each question with its untaken branch.

    ``alternatives`` maps every visited node to the label of the branch that
    was *not* taken (a further question or a recommendation), drawn in grey.
    """
    lines = _header()
    keys = [step.node for step in recommendation.path] + [recommendation.leaf]
    for step, target in zip(recommendation.path, keys[1:]):
        label = f"{_wrap(step.question)}\\n\\n{_escape(FEATURE_NAMES_DE.get(step.feature, step.feature))} = {_format_value(step.feature, step.value)}"
        lines.append(f'  "{step.node}" [label="{label}", {_node_style(True, leaf=False)}];')
        other = f"{step.node}__other"
        lines.append(f'  "{other}" [label="{_wrap(alternatives[step.node], 30)}", shape=box, style="rounded,dashed", fontsize=9];')
        answer, other_answer = ("ja", "nein") if step.answer else ("nein", "ja")
        lines.append(f'  "{step.node}" -> "{target}" [label="{answer}", {_edge_style(True)}];')
        lines.append(f'  "{step.node}" -> "{other}" [label="{other_answer}", style=dashed];')
        lines.append(f'  {{ rank=same; "{target}"; "{other}"; }}')
    strategy = STRATEGIES[recommendation.strategy].name_de
    lines.append(f'  "{recommendation.leaf}" [label="Empfehlung:\\n{_wrap(strategy, 28)}", {_node_style(True, leaf=True)}];')
    lines.append("}")
    return "\n".join(lines)


def rulebook_alternatives(rulebook: RuleBook, recommendation: Recommendation) -> dict[str, str]:
    """Label of the untaken branch for every visited node of the rule book."""
    labels = {}
    for step in recommendation.path:
        node = rulebook.nodes[step.node]
        other = node.no if step.answer else node.yes
        labels[step.node] = (
            f"→ {STRATEGIES[rulebook.leaves[other].strategy].name_de}" if other in rulebook.leaves else f"→ weitere Frage: {rulebook.nodes[other].question}"
        )
    return labels


def tree_alternatives(selector: LearnedSelector, recommendation: Recommendation) -> dict[str, str]:
    """Label of the untaken branch for every visited node of the tree."""
    labels = {}
    for step in recommendation.path:
        node = int(step.node.removeprefix("node_"))
        other = selector.right[node] if step.answer else selector.left[node]
        if selector.is_leaf(other):
            labels[step.node] = f"→ {STRATEGIES[selector.leaf_class[other]].name_de}"
        else:
            name = selector.feature_names[selector.feature[other]]
            labels[step.node] = f"→ weitere Frage: {FEATURE_NAMES_DE.get(name, name)}"
    return labels


def rulebook_dot(rulebook: RuleBook, recommendation: Recommendation | None = None) -> str:
    """DOT source of the rule book; the path of ``recommendation`` is highlighted."""
    steps = {step.node: step for step in recommendation.path} if recommendation else {}
    active_edges = set()
    if recommendation:
        targets = [step.node for step in recommendation.path[1:]] + [recommendation.leaf]
        active_edges = {(step.node, target) for step, target in zip(recommendation.path, targets)}

    lines = _header()
    for key, node in rulebook.nodes.items():
        label = _wrap(node.question)
        if key in steps:
            step = steps[key]
            label += f"\\n\\n{_escape(FEATURE_NAMES_DE[step.feature])} = {_format_value(step.feature, step.value)}"
        lines.append(f'  "{key}" [label="{label}", {_node_style(key in steps, leaf=False)}];')
    for key, leaf in rulebook.leaves.items():
        active = recommendation is not None and key == recommendation.leaf
        label = _wrap(STRATEGIES[leaf.strategy].name_de, 28)
        lines.append(f'  "{key}" [label="{label}", {_node_style(active, leaf=True)}];')
    for source, target, answer in rulebook.edges():
        style = _edge_style((source, target) in active_edges)
        lines.append(f'  "{source}" -> "{target}" [label="{"ja" if answer else "nein"}"{", " + style if style else ""}];')
    lines.append("}")
    return "\n".join(lines)


def tree_dot(selector: LearnedSelector, recommendation: Recommendation | None = None) -> str:
    """DOT source of the learned tree; the path of ``recommendation`` is highlighted."""
    steps = {step.node: step for step in recommendation.path} if recommendation else {}
    visited = set(steps) | ({recommendation.leaf} if recommendation else set())
    lines = _header()
    for node in range(len(selector.feature)):
        key = f"node_{node}"
        if selector.is_leaf(node):
            strategy = STRATEGIES[selector.leaf_class[node]].name_de
            label = f"{_wrap(strategy, 28)}\\n({selector.leaf_samples[node]} Fälle, {selector.leaf_purity[node]:.0%})"
            lines.append(f'  "{key}" [label="{label}", {_node_style(key in visited, leaf=True)}];')
            continue
        name = selector.feature_names[selector.feature[node]]
        threshold = _format_value(name, selector.threshold[node]) if name not in ("is_list", "sorted_known", "position_front") else "0,5"
        label = _wrap(f"{FEATURE_NAMES_DE.get(name, name)} ≤ {threshold}?")
        if key in steps:
            label += f"\\n\\nWert = {_format_value(name, steps[key].value)}"
        lines.append(f'  "{key}" [label="{label}", {_node_style(key in visited, leaf=False)}];')
        for child, answer in ((selector.left[node], True), (selector.right[node], False)):
            child_key = f"node_{child}"
            active = key in steps and child_key in visited and steps[key].answer == answer
            style = _edge_style(active)
            lines.append(f'  "{key}" -> "{child_key}" [label="{"ja" if answer else "nein"}"{", " + style if style else ""}];')
    lines.append("}")
    return "\n".join(lines)
