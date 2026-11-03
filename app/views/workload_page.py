"""Page 2 – Zugriffsmuster: search key and the expected queries."""

import numpy as np
import streamlit as st

from recommender.rules import Workload

from .common import german, keys_in_order

QUERY_PRESETS = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000, 1_000_000, 10_000_000]


def _random_key(hit: bool) -> int:
    values = keys_in_order(st.session_state.data)
    rng = np.random.default_rng()
    if values.size == 0:
        return 0
    if hit:
        return int(values[rng.integers(values.size)])
    lo, hi = int(values.min()), int(values.max())
    # Draw a batch of candidates from the value range and keep one that is
    # really absent (one pass over all values, so this is exact for any n).
    candidates = rng.integers(lo, hi + 2, 1000)
    absent = candidates[~np.isin(candidates, values)]
    return int(absent[0]) if absent.size else hi + 1


def render() -> None:
    st.title("2 · Zugriffsmuster")
    st.write("Wie die Daten durchsucht werden, entscheidet mit darüber, ob sich eine Vorverarbeitung lohnt.")

    st.subheader("Suchschlüssel")
    left, middle, right = st.columns([2, 1, 1])
    with middle:
        if st.button("Zufälliger Treffer", use_container_width=True):
            st.session_state.search_key = _random_key(hit=True)
    with right:
        if st.button("Zufälliger Fehlschlag", use_container_width=True):
            st.session_state.search_key = _random_key(hit=False)
    with left:
        st.session_state.search_key = int(st.number_input("Gesuchter Wert", value=int(st.session_state.search_key), step=1))

    st.subheader("Erwartete Anfragen")
    old = st.session_state.workload
    left, right = st.columns(2)
    with left:
        queries = st.select_slider("Anzahl Anfragen q", options=QUERY_PRESETS, value=min(QUERY_PRESETS, key=lambda v: abs(v - old.queries)), format_func=german)
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
        for key in ("execution", "comparison"):
            st.session_state.pop(key, None)

    st.subheader("Was das für die lineare Suche bedeutet")
    a, b = st.columns(2)
    a.metric("Gelesener Anteil pro Suche (im Mittel)", f"{german(round(workload.scan_fraction * 100, 1))} %")
    b.metric("Erwartete komplette Durchläufe für alle q Anfragen", german(round(workload.linear_scans, 2)))
    st.caption(
        "Diese Zahl steuert das Regelwerk: Eine Hashtabelle kostet in den Messungen etwa 7–34 komplette Durchläufe, "
        "die vollständige Sortierprüfung genau einen (siehe Annahme A33)."
    )
