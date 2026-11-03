"""Page 6 – Lernmodus: follow a search on a small array.

Two views:

* **Schritt für Schritt**: one algorithm, one click per comparison.
  Visualisations help mostly when the learner is active [hundhausen2002], so
  this view does not animate by itself.
* **Wettrennen**: several algorithms side by side on the same array, every
  lane advances one comparison per tick (like the popular "linear vs. binary
  search" animations). Shows *why* one method needs fewer steps.
"""

import html

import numpy as np
import streamlit as st

from recommender.data import DISTRIBUTIONS, DatasetConfig, generate_keys
from recommender.viz.race import DEFAULT_RACE, MAX_RACE_SIZE
from recommender.viz.stepper import MAX_LEARNING_SIZE, TRACERS, Step, trace

from .common import DISTRIBUTION_LABELS, german
from .race_view import render_race, rescale

STEP_VIEW = "Schritt für Schritt"
RACE_VIEW = "Wettrennen"

CELL_CSS = """
<style>
.lm-row {display: flex; flex-wrap: wrap; gap: 4px; margin: 0.5rem 0 1rem 0;}
.lm-cell {width: 3.1rem; border: 1px solid #c3c2b7; border-radius: 4px; text-align: center;
          padding: 2px 0; font-family: ui-monospace, monospace; background: #ffffff; color: #0b0b0b;}
.lm-cell .lm-index {font-size: 0.65rem; color: #898781;}
.lm-cell .lm-value {font-size: 0.85rem;}
.lm-out {background: #f0efec; color: #a3a29c; border-color: #e1e0d9;}
.lm-range {background: #e3eefb; border-color: #2a78d6;}
.lm-mark {border: 2px dashed #4a3aa7;}
.lm-probe {background: #fde3d6; border: 2px solid #eb6834; font-weight: 600;}
.lm-found {background: #d7f0d7; border: 2px solid #008300; font-weight: 600;}
.lm-legend span {display: inline-block; margin-right: 1rem; font-size: 0.8rem; color: inherit; opacity: 0.8;}
.lm-legend i {display: inline-block; width: 0.9rem; height: 0.9rem; border-radius: 3px; vertical-align: -2px; margin-right: 0.3rem;}
</style>
"""

LEGEND = (
    '<div class="lm-legend">'
    '<span><i style="background:#e3eefb;border:1px solid #2a78d6"></i>noch möglicher Bereich</span>'
    '<span><i style="background:#fde3d6;border:2px solid #eb6834"></i>gerade verglichen</span>'
    '<span><i style="background:#f0efec;border:1px solid #e1e0d9"></i>ausgeschlossen</span>'
    '<span><i style="background:#d7f0d7;border:2px solid #008300"></i>gefunden</span>'
    "</div>"
)


def _cells(values: np.ndarray, step: Step | None) -> str:
    """The array as a row of boxes, coloured by the state of the current step."""
    parts = ['<div class="lm-row">']
    for i, value in enumerate(values):
        classes = ["lm-cell"]
        if step is not None:
            if step.result is not None and step.result == i:
                classes.append("lm-found")
            elif step.probe == i:
                classes.append("lm-probe")
            elif step.lo <= i < step.hi:
                classes.append("lm-range")
            else:
                classes.append("lm-out")
            if i in step.marks:
                classes.append("lm-mark")
        parts.append(f'<div class="{" ".join(classes)}"><div class="lm-index">{i}</div><div class="lm-value">{html.escape(german(int(value)))}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def _reset() -> None:
    st.session_state.lm_step = 0


def _example_array(n: int, distribution: str, order: str, seed: int) -> np.ndarray:
    """Small example array with readable values 0 ... 999 (shape of the distribution kept)."""
    return rescale(generate_keys(DatasetConfig(n=n, distribution=distribution, order=order, seed=seed)))


def _key_controls(values: np.ndarray, n: int, seed: int) -> int:
    """Search key input with buttons for a random hit and a random miss."""
    rng = np.random.default_rng(seed)
    left, middle, right = st.columns([2, 1, 1])
    if "lm_key" not in st.session_state:
        st.session_state.lm_key = int(values[n // 2])
    with middle:
        if st.button("Zufälliger Treffer", use_container_width=True):
            st.session_state.lm_key = int(values[rng.integers(n)])
            _reset()
    with right:
        if st.button("Zufälliger Fehlschlag", use_container_width=True):
            candidates = sorted(set(range(int(values.min()) - 1, int(values.max()) + 2)) - set(values.tolist()))
            st.session_state.lm_key = int(candidates[rng.integers(len(candidates))])
            _reset()
    with left:
        key = int(st.number_input("Gesuchter Wert", value=int(st.session_state.lm_key), step=1, on_change=_reset))
        st.session_state.lm_key = key
    return key


def _step_view(algorithm: str, values: np.ndarray, key: int, n: int) -> None:
    steps = trace(algorithm, values, key)
    st.session_state.setdefault("lm_step", 0)
    position = min(st.session_state.lm_step, len(steps))

    back, forward, restart, info = st.columns([1, 1, 1, 3])
    if back.button("◀ Zurück", disabled=position == 0, use_container_width=True):
        position -= 1
    if forward.button("Weiter ▶", type="primary", disabled=position >= len(steps), use_container_width=True):
        position += 1
    if restart.button("Neustart", use_container_width=True):
        position = 0
    st.session_state.lm_step = position
    info.markdown(f"Schritt **{position}** von **{len(steps)}**")

    current = steps[position - 1] if position > 0 else None
    st.markdown(CELL_CSS + _cells(values, current) + LEGEND, unsafe_allow_html=True)
    if current is None:
        st.info(f"Start: gesucht wird {german(key)} in {n} Werten. Noch nichts verglichen.")
    elif current.done:
        (st.success if current.result >= 0 else st.warning)(current.text)
    else:
        st.info(current.text)
    if position > 0:
        with st.expander("Bisherige Schritte"):
            for number, step in enumerate(steps[:position], start=1):
                st.markdown(f"{number}. {step.text}")


def render() -> None:
    st.title("6 · Lernmodus")
    view = st.radio("Ansicht", [STEP_VIEW, RACE_VIEW], horizontal=True, label_visibility="collapsed", on_change=_reset)
    if view == STEP_VIEW:
        st.write("Verfolge eine Suche Schritt für Schritt. Klicke auf **Weiter**, um den nächsten Vergleich zu sehen.")
    else:
        st.write(
            "Mehrere Verfahren suchen **denselben Wert im selben Array**. In jedem Takt macht jedes Verfahren einen Schritt; "
            "wer zuerst fertig ist, gewinnt. So sieht man, warum Halbieren oder Springen so viel weniger Schritte braucht als das Durchlaufen."
        )

    left, right = st.columns(2)
    with left:
        if view == STEP_VIEW:
            algorithms = [st.selectbox("Suchverfahren", list(TRACERS), format_func=lambda k: TRACERS[k][0], on_change=_reset)]
        else:
            algorithms = st.multiselect(
                "Verfahren im Rennen",
                list(TRACERS),
                default=list(DEFAULT_RACE),
                format_func=lambda k: TRACERS[k][0],
                help="Alle Verfahren außer der linearen Suche brauchen sortierte Daten; das Beispiel-Array ist deshalb sortiert.",
            )
        n = st.slider("Größe des Beispiel-Arrays", 4, min(MAX_LEARNING_SIZE, MAX_RACE_SIZE), 24, on_change=_reset)
    with right:
        distribution = st.selectbox("Verteilung der Werte", DISTRIBUTIONS, format_func=DISTRIBUTION_LABELS.get, on_change=_reset)
        seed = st.number_input("Seed", min_value=0, value=3, step=1, on_change=_reset)

    needs_sorted = view == RACE_VIEW or TRACERS[algorithms[0]][2]
    order = "sorted" if needs_sorted else st.radio("Reihenfolge (nur lineare Suche)", ["sorted", "unsorted"], format_func={"sorted": "sortiert", "unsorted": "unsortiert"}.get, horizontal=True, on_change=_reset)
    values = _example_array(n, distribution, order, int(seed))
    key = _key_controls(values, n, int(seed))

    if view == STEP_VIEW:
        _step_view(algorithms[0], values, key, n)
    else:
        render_race(values, key, algorithms)
