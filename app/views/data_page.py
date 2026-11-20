"""Page 1 – Daten: generate a dataset or upload a CSV file."""

import numpy as np
import streamlit as st

from recommender.data import (
    DISTRIBUTIONS,
    ORDERS,
    DatasetConfig,
    build_linked_list,
    column_keys,
    generate_keys,
    numeric_columns,
    read_table,
    rescale_readable,
)

from . import charts
from .common import DISTRIBUTION_LABELS, ORDER_LABELS, STRUCTURE_LABELS, german, keys_in_order, next_step, page_header, set_data, structure_of

SIZES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
MAX_LIST_SIZE = 1_000_000  # one search on a longer list takes seconds
PREVIEW = 15  # number of values listed below the charts


def _short_structure(structure: str) -> str:
    return STRUCTURE_LABELS[structure].split(" (")[0]


def _generate_form() -> None:
    with st.form("generate", border=True):
        left, right = st.columns(2)
        with left:
            n = st.select_slider("Datengröße n", options=SIZES, value=100_000, format_func=german)
            distribution = st.selectbox("Verteilung der Werte", DISTRIBUTIONS, format_func=DISTRIBUTION_LABELS.get)
            duplicates = st.select_slider("Anteil Duplikate", options=[0.0, 0.1, 0.25, 0.5, 0.75], value=0.0, format_func=lambda v: f"{round(v * 100)} %")
        with right:
            order = st.radio("Sortierung", ORDERS, format_func=ORDER_LABELS.get, horizontal=True)
            swaps = st.slider(
                "Vertauschungen bei „fast sortiert“ (Anteil von n)",
                0.001,
                0.1,
                0.01,
                format="%.3f",
                help="Wirkt nur bei „fast sortiert“: so viele zufällige Paare werden nach dem Sortieren vertauscht.",
            )
            structure = st.radio("Speicherstruktur", list(STRUCTURE_LABELS), format_func=STRUCTURE_LABELS.get)
            seed = st.number_input("Zufalls-Seed", min_value=0, value=1, step=1, help="Gleicher Seed und gleiche Einstellungen ergeben immer dieselben Daten.")
        small_values = st.checkbox(
            "Werte auf einen gut lesbaren Bereich umrechnen (ab 0 … 10·n)",
            value=True,
            help="Der Generator streut die Werte über einen riesigen Zahlenbereich (bis 2⁵²). Die Umrechnung erhält Reihenfolge, Sortierung "
            "und Duplikatanteil. Schief verteilte Daten bekommen so viele Stellen, wie ihre Form braucht (Annahme A39). "
            "Die Messreihen der Arbeit verwenden die ursprünglichen Werte.",
        )
        submitted = st.form_submit_button("Datenmenge erzeugen", type="primary", icon=":material/auto_awesome:")
    if not submitted:
        return
    if structure == "linked_list" and n > MAX_LIST_SIZE:
        st.error(f"Verkettete Listen sind auf n ≤ {german(MAX_LIST_SIZE)} begrenzt, weil schon eine Suche sonst Sekunden dauert.")
        return
    config = DatasetConfig(n=n, distribution=distribution, order=order, swap_fraction=swaps, duplicate_share=duplicates, structure=structure, seed=int(seed))
    with st.spinner("Datenmenge wird erzeugt …"):
        keys = generate_keys(config)
        if small_values:
            keys = rescale_readable(keys)
        # same node placement as generate_dataset (seed + 1)
        data = build_linked_list(keys, np.random.default_rng(config.seed + 1)) if structure == "linked_list" else keys
    description = f"{german(n)} Werte, {DISTRIBUTION_LABELS[distribution]}, {ORDER_LABELS[order]}, {round(duplicates * 100)} % Duplikate, {_short_structure(structure)}"
    if small_values:
        description += ", Werte umgerechnet"
    set_data(data, description)
    st.toast(f"Erzeugt: {description}", icon=":material/check_circle:")


def _upload_form() -> None:
    upload = st.file_uploader("CSV- oder Textdatei mit ganzen Zahlen", type=["csv", "txt"])
    st.caption(
        "Eine Spalte mit ganzen Zahlen genügt, mit oder ohne Überschrift. Trennzeichen `;`, `,` oder Tabulator werden erkannt. "
        "Bei mehreren Spalten wählst du unten die richtige aus. Die Reihenfolge der Datei bleibt erhalten."
    )
    if upload is None:
        return
    try:
        frame = read_table(upload.getvalue())
    except ValueError as error:
        st.error(str(error))
        return
    numeric = numeric_columns(frame)
    if not numeric:
        st.error("Die Datei enthält keine Spalte mit Zahlen.")
        return
    left, right = st.columns(2)
    column = left.selectbox("Spalte", numeric)
    structure = right.radio("Speicherstruktur", list(STRUCTURE_LABELS), format_func=STRUCTURE_LABELS.get, key="upload_structure")
    try:
        uploaded = column_keys(frame[column])
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"Vorschau: {german(uploaded.keys.size)} Werte, beginnend mit " + ", ".join(german(int(v)) for v in uploaded.keys[:8]) + (" …" if uploaded.keys.size > 8 else ""))
    if structure == "linked_list" and uploaded.keys.size > MAX_LIST_SIZE:
        st.error(f"Verkettete Listen sind auf n ≤ {german(MAX_LIST_SIZE)} begrenzt.")
        return
    if st.button("Übernehmen", type="primary", icon=":material/upload:"):
        keys = uploaded.keys
        data = build_linked_list(keys, np.random.default_rng(0)) if structure == "linked_list" else keys
        description = f"{german(keys.size)} Werte aus „{upload.name}“ (Spalte {column}), {_short_structure(structure)}"
        set_data(data, description)
        skipped = f" {german(uploaded.skipped)} leere Zellen wurden übersprungen." if uploaded.skipped else ""
        st.success(f"Übernommen: {description}.{skipped}")


def _current_data() -> None:
    st.subheader("Aktuelle Datenmenge")
    st.caption(st.session_state.data_description)
    values = keys_in_order(st.session_state.data)
    if values.size == 0:
        st.info("Die Datenmenge ist leer.")
        return
    # Keys can have up to 19 digits: the value range gets wide columns so nothing is cut off.
    a, b, c = st.columns([1, 1, 2])
    a.metric("Anzahl Werte n", german(values.size), border=True)
    b.metric("Speicherstruktur", _short_structure(structure_of(st.session_state.data)), border=True)
    with c.container(border=True, height="stretch"):
        st.markdown("Wertebereich")
        st.markdown(f"#### {german(int(values.min()))}  bis  {german(int(values.max()))}")

    histogram, positions = st.tabs([":material/bar_chart: Verteilung der Werte", ":material/scatter_plot: Wert über Position"])
    with histogram:
        st.altair_chart(charts.histogram(values), width="stretch")
        st.caption("Wie häufig die Werte in den einzelnen Bereichen vorkommen. Die Form zeigt die Verteilung, unabhängig von der Reihenfolge.")
    with positions:
        st.altair_chart(charts.value_over_position(values), width="stretch")
        st.caption(
            "Jeder Punkt ist ein Element an seiner Position (bei großen Daten gleichmäßig ausgedünnt). Sortierte Daten ergeben eine steigende Kurve. "
            "Ist die Kurve fast eine Gerade, schätzt die Interpolationssuche die Position gut. Unsortierte Daten ergeben eine Punktwolke."
        )
    st.caption(f"Die ersten Werte in gespeicherter Reihenfolge: {', '.join(german(int(v)) for v in values[:PREVIEW])}" + (" …" if values.size > PREVIEW else ""))


def render() -> None:
    page_header(1, "Erzeuge eine Datenmenge mit gezielten Eigenschaften oder lade eigene Daten hoch. Alle weiteren Schritte arbeiten mit dieser Datenmenge.")
    source = st.segmented_control("Quelle", ["Erzeugen", "CSV hochladen"], default="Erzeugen", label_visibility="collapsed") or "Erzeugen"
    if source == "Erzeugen":
        _generate_form()
    else:
        _upload_form()
    _current_data()
    next_step(1)
