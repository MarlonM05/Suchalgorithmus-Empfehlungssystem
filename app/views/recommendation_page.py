"""Page 4 – Empfehlung: recommended strategy, justification and decision path."""

import pandas as pd
import streamlit as st

from recommender.rules import STRATEGIES
from recommender.rules.decision import FEATURE_NAMES_DE
from recommender.viz.graph import path_dot, rulebook_alternatives, rulebook_dot, tree_alternatives, tree_dot

from .common import RULES, TREE, current_features, format_ns, german, learned_tree, next_step, page_header, recommend, rulebook, strategy_properties

BOOLEAN_FEATURES = ("is_list", "sorted_known", "looks_sorted", "position_front")


def _value(feature: str, value: float) -> str:
    if feature in BOOLEAN_FEATURES:
        return "ja" if value >= 0.5 else "nein"
    return german(round(value, 3))


def _strategy_card(recommendation) -> None:
    strategy = STRATEGIES[recommendation.strategy]
    with st.container(border=True):
        st.markdown(f"#### :material/recommend: Empfehlung: {strategy.name_de}")
        st.write(recommendation.reason)
        properties = strategy_properties(recommendation.strategy)
        columns = st.columns(3)
        for column, label in zip(columns, ("Laufzeit pro Anfrage", "Vorverarbeitung", "Braucht sortierte Daten")):
            column.markdown(f"<small>{label}</small><br>**{properties[label]}**", unsafe_allow_html=True)
        features = current_features()
        sample_note = (
            f"Die Entscheidung brauchte die Stichprobe (Analysekosten {format_ns(features.analysis_time_ns)})."
            if recommendation.uses_sampled_features
            else "Die Entscheidung kam ohne Stichprobe aus: Datengröße, Struktur und deine Angaben reichten."
        )
        sources = ", ".join(f"[{s}]" for s in recommendation.sources)
        st.caption(sample_note + (f" Quellen: {sources}" if sources else ""))


def _second_opinion(selector: str, recommendation) -> None:
    """Compare with the other selector, so the user sees whether both agree."""
    if learned_tree() is None:
        return
    other_name = TREE if selector == RULES else RULES
    other = recommend(other_name)
    if other.strategy == recommendation.strategy:
        st.caption(f":material/done_all: Der {'gelernte Entscheidungsbaum' if other_name == TREE else 'Regelwerk'} empfiehlt dasselbe.")
    else:
        st.caption(
            f":material/compare_arrows: Zum Vergleich: {'Der gelernte Entscheidungsbaum' if other_name == TREE else 'Das Regelwerk'} "
            f"empfiehlt „{STRATEGIES[other.strategy].name_de}“. Welche Wahl schneller ist, zeigt der Vergleichsmodus auf Seite 5."
        )


def render() -> None:
    page_header(4, "Das Empfehlungssystem wählt eine Strategie (Suchverfahren samt Vorverarbeitung) und zeigt, welche Fragen zu dieser Wahl geführt haben.")
    options = [RULES, TREE] if learned_tree() is not None else [RULES]
    current = st.session_state.get("selector", RULES)
    selector = st.segmented_control(
        "Empfehlung durch",
        options,
        default=current if current in options else RULES,
        help="Regelwerk: von Hand formulierte Regeln mit Schwellen aus den Messungen. Entscheidungsbaum: aus den Messungen gelernt (scikit-learn, Tiefe 4).",
    ) or RULES
    st.session_state.selector = selector  # plain key: survives switching pages, unlike widget state
    if learned_tree() is None:
        st.info("Kein gelernter Entscheidungsbaum gefunden. Erzeuge ihn mit `python experiments/exp_recommender.py --mode full`.")

    recommendation = recommend(selector)
    _strategy_card(recommendation)
    _second_opinion(selector, recommendation)

    st.subheader("Entscheidungsweg")
    full = st.toggle("Vollständigen Graphen zeigen", value=False, help="Alle Fragen und Empfehlungen; der Weg dieser Empfehlung bleibt blau hervorgehoben.")
    if full:
        dot = rulebook_dot(rulebook(), recommendation) if selector == RULES else tree_dot(learned_tree(), recommendation)
        caption = "Blau: der Weg dieser Empfehlung mit den gemessenen Merkmalswerten. Grau: die übrigen Fragen und Empfehlungen. Vollbild über das Symbol oben rechts am Graphen."
    else:
        alternatives = rulebook_alternatives(rulebook(), recommendation) if selector == RULES else tree_alternatives(learned_tree(), recommendation)
        dot = path_dot(recommendation, alternatives)
        caption = "Blau: der Weg dieser Empfehlung mit den gemessenen Merkmalswerten. Gestrichelt: was bei der anderen Antwort passiert wäre."
    with st.container(border=True):
        st.graphviz_chart(dot, width="stretch")
    st.caption(caption)

    rows = [
        {
            "Frage": step.question,
            "Merkmal": FEATURE_NAMES_DE.get(step.feature, step.feature),
            "Wert": _value(step.feature, step.value),
            "Antwort": "✅ ja" if step.answer else "❌ nein",
            "Begründung": step.reason,
        }
        for step in recommendation.path
    ]
    with st.expander("Entscheidungsweg als Tabelle", expanded=True, icon=":material/table:"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Graph als DOT-Datei (für die Ausarbeitung)", dot, file_name="entscheidungsweg.dot", mime="text/vnd.graphviz", icon=":material/download:")
    next_step(4)
