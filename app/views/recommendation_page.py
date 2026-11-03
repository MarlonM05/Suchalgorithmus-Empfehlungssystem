"""Page 4 – Empfehlung: recommended strategy, profile, decision path and a race."""

import numpy as np
import pandas as pd
import streamlit as st

from recommender.algorithms import ALGORITHMS, LIST_ALGORITHMS
from recommender.rules import STRATEGIES
from recommender.rules.decision import FEATURE_NAMES_DE
from recommender.viz.graph import format_value, path_dot, rulebook_alternatives, rulebook_dot, tree_alternatives, tree_dot

from .common import RULES, TREE, current_features, format_ns, german, keys_in_order, learned_tree, recommend, rulebook
from .race_view import excerpt, render_race, rescale, traceable

PREPROCESSING_DE = {
    None: "keine",
    "sort": "Daten sortieren (stabiles argsort, O(n log n))",
    "hash_build": "Hashtabelle aufbauen (O(n), Füllgrad ≤ 0,5)",
    "list_hash_build": "Liste in eine Hashtabelle kopieren (O(n))",
}
RACE_SIZE = 24


def _profile(strategy_key: str, workload) -> None:
    """Short profile of the recommended strategy: what runs, what it needs, what it costs."""
    strategy = STRATEGIES[strategy_key]
    algorithm = (LIST_ALGORITHMS | ALGORITHMS)[strategy.search]
    check = (
        "entfällt (Sortierung garantiert)"
        if workload.sorted_known
        else "vollständige Sortierprüfung, ein Durchlauf O(n), bricht beim ersten Fehler ab"
    )
    rows = [
        ("Suchverfahren", algorithm.name_de),
        ("Komplexität pro Anfrage", algorithm.complexity),
        ("Vorverarbeitung", PREPROCESSING_DE[strategy.preprocessing]),
        ("Voraussetzung", (check if strategy.requires_sorted else "keine Sortierung nötig")),
        ("Speicherstruktur", "verkettete Liste (nur sequenzieller Zugriff)" if strategy.structure == "linked_list" else "Array (Direktzugriff)"),
        ("Quellen", ", ".join(f"[{s}]" for s in algorithm.sources)),
    ]
    st.table(pd.DataFrame(rows, columns=["", "Empfohlene Strategie"]).set_index(""))


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
    st.title("4 · Empfehlung")
    options = [RULES, TREE] if learned_tree() is not None else [RULES]
    current = st.session_state.get("selector", RULES)
    selector = st.radio("Empfehlung durch", options, index=options.index(current) if current in options else 0, horizontal=True)
    st.session_state.selector = selector  # plain key: survives switching pages, unlike widget state
    if learned_tree() is None:
        st.info("Kein gelernter Entscheidungsbaum gefunden. Erzeuge ihn mit `python experiments/exp_recommender.py --mode full`.")

    workload = st.session_state.workload
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

    left, right = st.columns([3, 2])
    with left:
        _profile(recommendation.strategy, workload)
    with right:
        st.markdown("**Warum diese Strategie?**")
        st.markdown(
            "\n".join(
                f"{number}. {step.question} **{'Ja' if step.answer else 'Nein'}** "
                f"({FEATURE_NAMES_DE.get(step.feature, step.feature)} = {format_value(step.feature, step.value)})"
                for number, step in enumerate(recommendation.path, start=1)
            )
        )
        _cost_model(recommendation.strategy, workload)

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
            "Wert": format_value(step.feature, step.value),
            "Antwort": "ja" if step.answer else "nein",
            "Begründung": step.reason,
        }
        for step in recommendation.path
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Graph als DOT-Datei (für die Ausarbeitung)", dot, file_name="entscheidungsweg.dot", mime="text/vnd.graphviz")

    _race_section(recommendation.strategy)
