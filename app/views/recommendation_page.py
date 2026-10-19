"""Page 4 – Empfehlung: recommended strategy, justification and decision path."""

import pandas as pd
import streamlit as st

from recommender.rules import STRATEGIES
from recommender.rules.decision import FEATURE_NAMES_DE
from recommender.viz.graph import path_dot, rulebook_alternatives, rulebook_dot, tree_alternatives, tree_dot

from .common import RULES, TREE, current_features, format_ns, german, learned_tree, recommend, rulebook


def _value(feature: str, value: float) -> str:
    if feature in ("is_list", "sorted_known", "looks_sorted", "position_front"):
        return "ja" if value >= 0.5 else "nein"
    return german(round(value, 3))


def render() -> None:
    st.title("4 · Empfehlung")
    options = [RULES, TREE] if learned_tree() is not None else [RULES]
    current = st.session_state.get("selector", RULES)
    selector = st.radio("Empfehlung durch", options, index=options.index(current) if current in options else 0, horizontal=True)
    st.session_state.selector = selector  # plain key: survives switching pages, unlike widget state
    if learned_tree() is None:
        st.info("Kein gelernter Entscheidungsbaum gefunden. Erzeuge ihn mit `python experiments/exp_recommender.py --mode full`.")

    recommendation = recommend(selector)
    strategy = STRATEGIES[recommendation.strategy]
    st.success(f"**Empfehlung: {strategy.name_de}**")
    st.write(recommendation.reason)
    sources = ", ".join(f"[{s}]" for s in recommendation.sources)
    features = current_features()
    sample_note = (
        f"Die Entscheidung brauchte die Stichprobe (Analysekosten {format_ns(features.analysis_time_ns)})."
        if recommendation.uses_sampled_features
        else "Die Entscheidung kam ohne Stichprobe aus: Datengröße, Struktur und deine Angaben reichten."
    )
    st.caption(f"{sample_note}" + (f" Quellen: {sources}" if sources else ""))

    st.subheader("Entscheidungsweg")
    full = st.toggle("Vollständigen Graphen zeigen", value=False, help="Alle Fragen und Empfehlungen; der Weg dieser Empfehlung bleibt blau hervorgehoben.")
    if full:
        dot = rulebook_dot(rulebook(), recommendation) if selector == RULES else tree_dot(learned_tree(), recommendation)
        caption = "Blau: der Weg dieser Empfehlung mit den gemessenen Merkmalswerten. Grau: die übrigen Fragen und Empfehlungen. Vollbild über das Symbol oben rechts am Graphen."
    else:
        alternatives = rulebook_alternatives(rulebook(), recommendation) if selector == RULES else tree_alternatives(learned_tree(), recommendation)
        dot = path_dot(recommendation, alternatives)
        caption = "Blau: der Weg dieser Empfehlung mit den gemessenen Merkmalswerten. Gestrichelt: was bei der anderen Antwort passiert wäre."
    st.graphviz_chart(dot, use_container_width=True)
    st.caption(caption)

    rows = [
        {
            "Frage": step.question,
            "Merkmal": FEATURE_NAMES_DE.get(step.feature, step.feature),
            "Wert": _value(step.feature, step.value),
            "Antwort": "ja" if step.answer else "nein",
            "Begründung": step.reason,
        }
        for step in recommendation.path
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Graph als DOT-Datei (für die Ausarbeitung)", dot, file_name="entscheidungsweg.dot", mime="text/vnd.graphviz")
