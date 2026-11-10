"""Page 4 – Empfehlung: recommended strategy, justification, decision path and a race."""

import pandas as pd
import streamlit as st

from recommender.rules import STRATEGIES
from recommender.rules.decision import FEATURE_NAMES_DE
from recommender.viz.graph import path_dot, rulebook_alternatives, rulebook_dot, tree_alternatives, tree_dot

from .common import RULES, TREE, current_features, format_ns, german, keys_in_order, learned_tree, next_step, page_header, recommend, rulebook, strategy_properties
from .race_view import excerpt, render_race, rescale, traceable

BOOLEAN_FEATURES = ("is_list", "sorted_known", "looks_sorted", "position_front")
RACE_SIZE = 24


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


def _cost_model(strategy_key: str, workload) -> None:
    """The additive cost model with the user's number of queries filled in."""
    strategy = STRATEGIES[strategy_key]
    parts = []
    if strategy.requires_sorted and not workload.sorted_known:
        parts.append("Sortierprüfung")
    if strategy.preprocessing is not None:
        parts.append("Vorverarbeitung")
    fixed = " + ".join(parts) if parts else "0"
    st.markdown(
        f"**Kostenmodell (Annahme A22):** T(q) = {fixed} + q · Zeit pro Anfrage, hier mit q = {german(workload.queries)} Anfragen. "
        "Die gemessenen Werte dafür liefert Seite 5 (Ausführung)."
    )


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


def _race_section(strategy_key: str) -> None:
    """Linear search against the recommended method on a small sorted excerpt of the data."""
    st.subheader("Anschaulich: Wettrennen auf einem Ausschnitt deiner Daten")
    values = keys_in_order(st.session_state.data)
    if values.size == 0:
        return
    sample = rescale(excerpt(values, RACE_SIZE, seed=st.session_state.get("data_version", 0)))
    n = int(sample.size)
    competitor = traceable(strategy_key)
    if competitor is None:
        st.write(
            "Der **Hash-Lookup** lässt sich nicht als Suche durch ein geordnetes Array zeigen: Nach dem Aufbau der Tabelle "
            "berechnet er aus dem Schlüssel direkt den Slot und braucht erwartet nur einen Zugriff [richter2015]. "
            "Zum Vergleich läuft hier die lineare gegen die binäre Suche, die Alternativen ohne Hashtabelle."
        )
        algorithms = ["linear", "binary"]
    elif competitor == "linear":
        st.write("Die Empfehlung ist die lineare Suche selbst. Zum Vergleich läuft sie hier gegen die binäre Suche, die sortierte Daten voraussetzt.")
        algorithms = ["linear", "binary"]
    else:
        note = " (gezeigt wird die klassische binäre Suche; die verzweigungsfreie Variante prüft dieselbe Anzahl Positionen, nur ohne bedingten Sprung)" if strategy_key in ("binary_branchless", "sort_binary") else ""
        st.write(f"Lineare Suche gegen das empfohlene Verfahren{note}.")
        algorithms = ["linear", competitor]
    st.caption(
        f"Ausschnitt: {n} zufällig gezogene Werte deiner Daten, sortiert und auf 0–999 skaliert "
        f"(aus n = {german(values.size)}). Gesucht wird ein vorhandener Wert; mit **▶ Start** läuft das Rennen."
    )
    key = int(sample[n // 3])
    render_race(sample, key, algorithms)


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
    _cost_model(recommendation.strategy, st.session_state.workload)

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

    _race_section(recommendation.strategy)
    next_step(4)
