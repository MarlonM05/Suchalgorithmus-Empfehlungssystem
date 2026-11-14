"""Page 2 – Zugriffsmuster: search key and the expected queries."""

import numpy as np
import streamlit as st

from recommender.data import random_hit_key, random_miss_key
from recommender.executor import expected_index
from recommender.rules import Workload

from .common import current_features, german, index_text, invalidate_results, keys_in_order, next_step, page_header, percent

QUERY_PRESETS = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000, 1_000_000, 10_000_000]


def _search_key() -> None:
    st.subheader("Suchschlüssel")
    values = keys_in_order(st.session_state.data)
    rng = np.random.default_rng()  # unseeded: every click draws a new key
    left, middle, right = st.columns([2, 1, 1], vertical_alignment="bottom")
    # The buttons are evaluated before the number field, so the field shows the new key.
    with middle:
        if st.button("Zufälliger Treffer", icon=":material/target:", width="stretch"):
            st.session_state.search_key = random_hit_key(values, rng)
    with right:
        if st.button("Zufälliger Fehlschlag", icon=":material/block:", width="stretch"):
            st.session_state.search_key = random_miss_key(values, rng)
    with left:
        st.session_state.search_key = int(st.number_input("Gesuchter Wert", value=int(st.session_state.search_key), step=1))

    index = expected_index(st.session_state.data, st.session_state.search_key)
    if index >= 0:
        st.caption(f":material/check_circle: Der Wert kommt vor, zum ersten Mal bei {index_text(index)}.")
    else:
        st.caption(":material/cancel: Der Wert kommt in den Daten nicht vor (Fehlschlag). Auch das ist ein wichtiger Fall: Die lineare Suche muss dann alles lesen.")


def render() -> None:
    page_header(2, "Wie die Daten durchsucht werden, entscheidet mit darüber, ob sich eine Vorverarbeitung wie Sortieren oder eine Hashtabelle lohnt.")
    _search_key()

    st.subheader("Erwartete Anfragen")
    old = st.session_state.workload
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            queries = st.select_slider(
                "Anzahl Anfragen q",
                options=QUERY_PRESETS,
                value=min(QUERY_PRESETS, key=lambda v: abs(v - old.queries)),
                format_func=german,
                help="Wie oft wird in denselben Daten gesucht? Viele Anfragen machen eine einmalige Vorverarbeitung lohnend.",
            )
            hit_ratio = st.slider("Anteil der Anfragen, die einen Treffer haben", 0, 100, int(round(old.hit_ratio * 100)), format="%d %%") / 100
        with right:
            position = st.radio(
                "Wo liegen die gesuchten Werte?",
                ["uniform", "front"],
                index=0 if old.position == "uniform" else 1,
                format_func={"uniform": "überall gleich wahrscheinlich", "front": "vorne gehäuft (z. B. neueste Einträge)"}.get,
            )
            sorted_known = st.checkbox(
                "Die Daten sind garantiert sortiert",
                value=old.sorted_known,
                help="Dann entfällt die vollständige Sortierprüfung (ein Durchlauf über alle Daten). Nur ankreuzen, wenn die Sortierung wirklich feststeht, sonst kann die binäre Suche falsche Ergebnisse liefern.",
            )
    workload = Workload(queries=queries, hit_ratio=hit_ratio, position=position, sorted_known=sorted_known)
    if workload != old:
        st.session_state.workload = workload
        invalidate_results()

    if sorted_known and not current_features().looks_sorted:
        st.warning(
            "Die Stichprobe hat Paare in falscher Reihenfolge gefunden: Die Daten sind **nicht sortiert**. "
            "Mit dieser Garantie würde die Sortierprüfung übersprungen, und Verfahren für sortierte Daten können Werte übersehen. "
            "Entferne den Haken oder sortiere die Daten auf Seite 1.",
            icon=":material/warning:",
        )

    st.subheader("Was das für die lineare Suche bedeutet")
    a, b = st.columns(2)
    a.metric("Gelesener Anteil pro Suche (im Mittel)", percent(workload.scan_fraction), border=True)
    b.metric("Erwartete komplette Durchläufe für alle q Anfragen", german(round(workload.linear_scans, 2)), border=True)
    st.caption(
        "Treffer an zufälliger Stelle lesen im Mittel die Hälfte der Daten, Fehlschläge alles. Die Zahl der Durchläufe steuert das Regelwerk: "
        "Eine Hashtabelle kostet in den Messungen etwa 7–34 komplette Durchläufe, die vollständige Sortierprüfung genau einen (Annahme A33)."
    )
    next_step(2)
