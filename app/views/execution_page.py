"""Page 5 – Ausführung: run the recommended search automatically; compare all strategies."""

import numpy as np
import pandas as pd
import streamlit as st

from recommender.executor import compare_all, execute
from recommender.rules import STRATEGIES, selection_features
from recommender.viz import plots

from .common import RULES, compiled, current_features, format_ns, german, learned_tree, recommend, rulebook


def _run_recommendation(selector: str):
    """Execute the recommendation; if the sortedness check fails, decide again knowing that."""
    workload = st.session_state.workload
    recommendation = recommend(selector)
    first = execute(recommendation.strategy, st.session_state.data, st.session_state.search_key, workload.sorted_known)
    if not first.check_failed:
        return recommendation, first, None
    features = {**selection_features(current_features(), workload), "looks_sorted": 0.0, "disorder": 1.0}
    chooser = rulebook() if selector == RULES or learned_tree() is None else learned_tree()
    retry = chooser.decide(features)
    second = execute(retry.strategy, st.session_state.data, st.session_state.search_key, workload.sorted_known)
    return retry, second, first


def _comparison_figure(table: pd.DataFrame, queries: int):
    plots.apply_style()
    fig, axes = plots.plt.subplots(figsize=(6.3, 0.45 * len(table) + 1.2))
    ordered = table.sort_values("total_ns", ascending=False)
    colors = [plots.ALGORITHM_STYLE.get(STRATEGIES[s].search, ("", plots.INK_MUTED))[1] for s in ordered["strategy"]]
    axes.barh(ordered["Strategie"], ordered["total_ns"], color=colors, height=0.6)
    axes.set_xscale("log")
    axes.set_xlabel(f"Geschätzte Gesamtzeit für {german(queries)} Anfragen [ns]")
    axes.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def render() -> None:
    st.title("5 · Ausführung")
    compiled()
    selector = st.session_state.get("selector", RULES)
    workload = st.session_state.workload
    key = st.session_state.search_key
    st.write(f"Die empfohlene Suche wird automatisch auf den aktuellen Daten ausgeführt. Gesucht wird **{german(key)}**.")

    state = (st.session_state.data_version, key, workload, selector)
    if st.session_state.get("execution", (None,))[0] != state:
        with st.spinner("Suche läuft …"):
            st.session_state.execution = (state, *_run_recommendation(selector))
    _, recommendation, result, failed = st.session_state.execution

    if failed is not None:
        st.warning(
            f"Die Stichprobe wirkte sortiert, die vollständige Prüfung fand aber eine Unordnung (nach {format_ns(failed.check_ns)}). "
            f"Statt „{STRATEGIES[failed.strategy].name_de}“ wurde deshalb neu entschieden."
        )
    st.success(f"**Ausgeführt: {STRATEGIES[result.strategy].name_de}**")
    a, b, c = st.columns(3)
    a.metric("Ergebnis", f"Index {german(result.index)}" if result.index >= 0 else "nicht gefunden")
    b.metric("Schlüsselvergleiche", german(result.comparisons))
    c.metric("Gelesene Elemente", german(result.accesses))
    a, b, c, d = st.columns(4)
    a.metric("Sortierprüfung", format_ns(result.check_ns) if result.check_ns else "entfällt")
    b.metric("Vorverarbeitung", format_ns(result.preprocessing_ns) if result.preprocessing_ns else "keine")
    c.metric("Zeit pro Anfrage", format_ns(result.query_ns))
    d.metric(f"Gesamt für {german(workload.queries)} Anfragen", format_ns(result.total_ns(workload.queries)))
    st.caption(
        "Die Zeit pro Anfrage ist der Mittelwert aus 1.000 Wiederholungen derselben Suche. Weil sich die CPU auf wiederholte Eingaben "
        "einstellt, ist sie eher optimistisch (Annahme A17). Die Gesamtzeit folgt dem Kostenmodell Vorbereitung + q · Zeit pro Anfrage."
    )

    st.subheader("Vergleichsmodus")
    st.write("Alle anwendbaren Strategien laufen auf denselben Daten mit demselben Schlüssel. So zeigt sich, ob die Empfehlung die beste war.")
    if st.button("Alle Verfahren vergleichen", type="primary"):
        with st.spinner("Alle Strategien laufen …"):
            st.session_state.comparison = (state, compare_all(st.session_state.data, key, workload.sorted_known))
    comparison = st.session_state.get("comparison")
    if comparison is None or comparison[0] != state:
        return
    results = comparison[1]
    table = pd.DataFrame(
        {
            "strategy": [r.strategy for r in results],
            "Strategie": [STRATEGIES[r.strategy].name_de for r in results],
            "Ergebnis": [german(r.index) if r.index >= 0 else "nicht gefunden" for r in results],
            "Vergleiche": [r.comparisons for r in results],
            "Vorbereitung": [format_ns(r.check_ns + r.preprocessing_ns) for r in results],
            "pro Anfrage": [format_ns(r.query_ns) for r in results],
            "total_ns": [r.total_ns(workload.queries) for r in results],
        }
    ).sort_values("total_ns")
    table[f"Gesamt ({german(workload.queries)} Anfragen)"] = [format_ns(v) for v in table["total_ns"]]
    best = table.iloc[0]
    recommended = table[table["strategy"] == result.strategy]
    if not recommended.empty and recommended.iloc[0]["strategy"] == best["strategy"]:
        st.success("Die Empfehlung war die schnellste Strategie.")
    elif not recommended.empty:
        factor = recommended.iloc[0]["total_ns"] / best["total_ns"]
        st.info(f"Am schnellsten war „{best['Strategie']}“. Die Empfehlung war {german(round(factor, 2))}-mal so langsam.")
    table.insert(0, "", np.where(table["strategy"] == result.strategy, "★ empfohlen", ""))
    st.dataframe(table.drop(columns=["strategy", "total_ns"]), hide_index=True, use_container_width=True)
    st.pyplot(_comparison_figure(table, workload.queries))
