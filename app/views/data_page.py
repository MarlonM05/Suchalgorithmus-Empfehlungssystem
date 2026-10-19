"""Page 1 – Daten: generate a dataset or upload a CSV file."""

import numpy as np
import pandas as pd
import streamlit as st

from recommender.data import DISTRIBUTIONS, ORDERS, DatasetConfig, build_linked_list, generate_dataset

from .common import DISTRIBUTION_LABELS, ORDER_LABELS, STRUCTURE_LABELS, german, keys_in_order, set_data

SIZES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
MAX_LIST_SIZE = 1_000_000  # one search on a longer list takes seconds


def _generate_form() -> None:
    with st.form("generate"):
        left, right = st.columns(2)
        with left:
            n = st.select_slider("Datengröße n", options=SIZES, value=100_000, format_func=german)
            distribution = st.selectbox("Verteilung der Werte", DISTRIBUTIONS, format_func=DISTRIBUTION_LABELS.get)
            duplicates = st.select_slider("Anteil Duplikate", options=[0.0, 0.1, 0.25, 0.5, 0.75], value=0.0, format_func=lambda v: f"{round(v * 100)} %")
        with right:
            order = st.radio("Sortierung", ORDERS, format_func=ORDER_LABELS.get, horizontal=True)
            swaps = st.slider("Vertauschungen bei „fast sortiert“ (Anteil von n)", 0.001, 0.1, 0.01, format="%.3f")
            structure = st.radio("Speicherstruktur", list(STRUCTURE_LABELS), format_func=STRUCTURE_LABELS.get)
            seed = st.number_input("Zufalls-Seed", min_value=0, value=1, step=1)
        submitted = st.form_submit_button("Datenmenge erzeugen", type="primary")
    if not submitted:
        return
    if structure == "linked_list" and n > MAX_LIST_SIZE:
        st.error(f"Verkettete Listen sind auf n ≤ {german(MAX_LIST_SIZE)} begrenzt, weil schon eine Suche sonst Sekunden dauert.")
        return
    config = DatasetConfig(n=n, distribution=distribution, order=order, swap_fraction=swaps, duplicate_share=duplicates, structure=structure, seed=int(seed))
    with st.spinner("Datenmenge wird erzeugt …"):
        data = generate_dataset(config)
    description = f"{german(n)} Werte, {DISTRIBUTION_LABELS[distribution]}, {ORDER_LABELS[order]}, {round(duplicates * 100)} % Duplikate, {STRUCTURE_LABELS[structure].split(' (')[0]}"
    set_data(data, description)
    st.success(f"Erzeugt: {description}.")


def _upload_form() -> None:
    upload = st.file_uploader("CSV-Datei mit ganzen Zahlen", type=["csv", "txt"])
    if upload is None:
        st.caption("Eine Spalte mit ganzen Zahlen genügt. Bei mehreren Spalten wählst du unten die richtige aus. Die Reihenfolge der Datei bleibt erhalten.")
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as error:  # pandas raises many different parser errors
        st.error(f"Die Datei konnte nicht gelesen werden: {error}")
        return
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        st.error("Die Datei enthält keine Zahlenspalte.")
        return
    column = st.selectbox("Spalte", numeric)
    structure = st.radio("Speicherstruktur", list(STRUCTURE_LABELS), format_func=STRUCTURE_LABELS.get, key="upload_structure")
    values = frame[column].dropna()
    if not np.all(np.isclose(values, np.round(values))):
        st.error("Die Spalte enthält Nachkommastellen. Unterstützt werden nur ganze Zahlen (siehe Annahme A2).")
        return
    if st.button("Übernehmen", type="primary"):
        keys = np.ascontiguousarray(np.round(values).astype(np.int64))
        if structure == "linked_list" and keys.size > MAX_LIST_SIZE:
            st.error(f"Verkettete Listen sind auf n ≤ {german(MAX_LIST_SIZE)} begrenzt.")
            return
        data = build_linked_list(keys, np.random.default_rng(0)) if structure == "linked_list" else keys
        dropped = len(frame) - len(values)
        description = f"{german(keys.size)} Werte aus „{upload.name}“ (Spalte {column}), {STRUCTURE_LABELS[structure].split(' (')[0]}"
        set_data(data, description)
        st.success(f"Übernommen: {description}." + (f" {dropped} leere Zeilen wurden übersprungen." if dropped else ""))


def render() -> None:
    st.title("1 · Daten")
    st.write("Erzeuge eine Datenmenge mit gezielten Eigenschaften oder lade eigene Daten hoch. Alle weiteren Seiten arbeiten mit dieser Datenmenge.")
    source = st.radio("Quelle", ["Erzeugen", "CSV hochladen"], horizontal=True, label_visibility="collapsed")
    if source == "Erzeugen":
        _generate_form()
    else:
        _upload_form()

    st.subheader("Aktuelle Datenmenge")
    st.caption(st.session_state.data_description)
    values = keys_in_order(st.session_state.data)
    if values.size == 0:
        st.info("Die Datenmenge ist leer.")
        return
    left, right = st.columns([1, 2])
    with left:
        st.metric("Anzahl Werte n", german(values.size))
        st.metric("Kleinster Wert", german(int(values.min())))
        st.metric("Größter Wert", german(int(values.max())))
    with right:
        st.caption("Verteilung der Werte (Histogramm)")
        counts, edges = np.histogram(values, bins=40)
        st.bar_chart(pd.DataFrame({"Anzahl": counts}, index=[f"{e:.3g}" for e in edges[:-1]]), height=220)
    st.caption("Die ersten Werte in gespeicherter Reihenfolge: " + ", ".join(german(int(v)) for v in values[:15]) + (" …" if values.size > 15 else ""))
