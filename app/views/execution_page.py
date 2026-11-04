"""Page 5 – Ausführung: run the recommended search automatically; compare all strategies.

Every result is checked against an independent reference (``expected_index``,
one NumPy comparison over all data). A wrong result is only possible if the
user guaranteed sortedness for unsorted data; the page then says so clearly.
"""

import numpy as np
import pandas as pd
import streamlit as st

from recommender.executor import compare_all, execute, expected_index
from recommender.rules import STRATEGIES, selection_features

from . import charts
from .common import compiled, current_features, format_ns, german, index_text, next_step, page_header, recommend, selector_for

WRONG_RESULT = (
    "Das Ergebnis ist **falsch**: Richtig wäre {expected}, geliefert wurde {found}. "
    "Grund: Die Sortierung wurde auf Seite 2 garantiert, die Daten sind aber nicht sortiert. "
    "Ohne Garantie hätte die Sortierprüfung das erkannt."
)


def _run_recommendation(selector: str):
    """Execute the recommendation; if the sortedness check fails, decide again knowing that."""
    workload = st.session_state.workload
    recommendation = recommend(selector)
    first = execute(recommendation.strategy, st.session_state.data, st.session_state.search_key, workload.sorted_known)
    if not first.check_failed:
        return recommendation, first, None
    features = {**selection_features(current_features(), workload), "looks_sorted": 0.0, "disorder": 1.0}
    retry = selector_for(selector).decide(features)
    second = execute(retry.strategy, st.session_state.data, st.session_state.search_key, workload.sorted_known)
    return retry, second, first


def _comparison(state: tuple, recommended: str, expected: int) -> None:
    st.subheader("Vergleichsmodus")
    st.write("Alle anwendbaren Strategien laufen auf denselben Daten mit demselben Schlüssel. So zeigt sich, ob die Empfehlung die beste war.")
    workload = st.session_state.workload
    if st.button("Alle Verfahren vergleichen", type="primary", icon=":material/leaderboard:"):
        with st.spinner("Alle Strategien laufen …"):
            st.session_state.comparison = (state, compare_all(st.session_state.data, st.session_state.search_key, workload.sorted_known))
    comparison = st.session_state.get("comparison")
    if comparison is None or comparison[0] != state:
        return
    results = comparison[1]
    table = pd.DataFrame(
        {
            "strategy": [r.strategy for r in results],
            "Strategie": [STRATEGIES[r.strategy].name_de for r in results],
            "Ergebnis": [index_text(r.index) for r in results],
            "Korrekt": ["✅" if r.index == expected else "❌" for r in results],
            "Vergleiche": [german(r.comparisons) for r in results],
            "Vorbereitung": [format_ns(r.check_ns + r.preprocessing_ns) for r in results],
            "pro Anfrage": [format_ns(r.query_ns) for r in results],
            "total_ns": [r.total_ns(workload.queries) for r in results],
        }
    ).sort_values("total_ns", ignore_index=True)
    table["empfohlen"] = table["strategy"] == recommended
    table[f"Gesamt ({german(workload.queries)} Anfragen)"] = [format_ns(v) for v in table["total_ns"]]

    best = table.iloc[0]
    if best["empfohlen"]:
        st.success("Die Empfehlung war die schnellste Strategie.", icon=":material/emoji_events:")
    elif table["empfohlen"].any():
        factor = table.loc[table["empfohlen"], "total_ns"].iloc[0] / best["total_ns"]
        st.info(f"Am schnellsten war „{best['Strategie']}“. Die Empfehlung war {german(round(factor, 2))}-mal so langsam.", icon=":material/info:")
    if (table["Korrekt"] == "❌").any():
        st.error("Mindestens ein Verfahren lieferte ein falsches Ergebnis, weil die Sortierung garantiert wurde, die Daten aber nicht sortiert sind.", icon=":material/error:")

    table.insert(0, "Empfohlen", np.where(table["empfohlen"], "★", ""))
    st.dataframe(table.drop(columns=["strategy", "total_ns", "empfohlen"]), hide_index=True, width="stretch")
    st.altair_chart(charts.strategy_comparison(table, workload.queries), width="stretch")
    st.caption("Logarithmische Achse: Jeder Gitterschritt ist ein Faktor 10. Die Gesamtzeit folgt dem Kostenmodell Vorbereitung + q · Zeit pro Anfrage.")


def render() -> None:
    page_header(5, "Die empfohlene Strategie wird automatisch auf den aktuellen Daten ausgeführt, gemessen und mit einer unabhängigen Referenz geprüft.")
    compiled()
    selector = st.session_state.get("selector", "Regelwerk")
    workload = st.session_state.workload
    key = st.session_state.search_key
    st.write(f"Gesucht wird **{german(key)}** (ändern auf Seite 2).")

    state = (st.session_state.data_version, key, workload, selector, st.session_state.get("sample_size"))
    if st.session_state.get("execution", (None,))[0] != state:
        with st.spinner("Suche läuft …"):
            st.session_state.execution = (state, *_run_recommendation(selector))
    _, recommendation, result, failed = st.session_state.execution
    expected = expected_index(st.session_state.data, key)

    if failed is not None:
        st.warning(
            f"Die Stichprobe wirkte sortiert, die vollständige Prüfung fand aber eine Unordnung (nach {format_ns(failed.check_ns)}). "
            f"Statt „{STRATEGIES[failed.strategy].name_de}“ wurde deshalb neu entschieden.",
            icon=":material/warning:",
        )
    with st.container(border=True):
        st.markdown(f"#### :material/play_circle: Ausgeführt: {STRATEGIES[result.strategy].name_de}")
        a, b, c = st.columns(3)
        a.metric("Ergebnis", index_text(result.index))
        b.metric("Schlüsselvergleiche", german(result.comparisons))
        c.metric("Gelesene Elemente", german(result.accesses))
        a, b, c, d = st.columns(4)
        a.metric("Sortierprüfung", format_ns(result.check_ns) if result.check_ns else "entfällt")
        b.metric("Vorverarbeitung", format_ns(result.preprocessing_ns) if result.preprocessing_ns else "keine")
        c.metric("Zeit pro Anfrage", format_ns(result.query_ns))
        d.metric(f"Gesamt für {german(workload.queries)} Anfragen", format_ns(result.total_ns(workload.queries)))
    if result.index == expected:
        st.success("Ergebnis geprüft: stimmt mit der Referenz überein (erstes Vorkommen, unabhängig mit NumPy berechnet).", icon=":material/verified:")
    else:
        st.error(WRONG_RESULT.format(expected=index_text(expected), found=index_text(result.index)), icon=":material/error:")
    st.caption(
        "Die Zeit pro Anfrage ist der Mittelwert aus 1.000 Wiederholungen derselben Suche. Weil sich die CPU auf wiederholte Eingaben "
        "einstellt, ist sie eher optimistisch (Annahme A36). Die Gesamtzeit folgt dem Kostenmodell Vorbereitung + q · Zeit pro Anfrage."
    )

    _comparison(state, result.strategy, expected)
    next_step(5)
