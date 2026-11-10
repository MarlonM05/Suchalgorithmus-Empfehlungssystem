"""Page 6 – Lernmodus: follow a search on a small array.

Two views:

* **Schritt für Schritt**: one algorithm, one click per comparison.
  Visualisations help mostly when the learner is active [hundhausen2002], so
  this view does not animate by itself: every step needs a click. Below the
  array, a chart compares how many steps every algorithm needs for the same
  key, which connects the single run to the complexity classes.
* **Wettrennen**: several algorithms side by side on the same array, every
  lane advances one comparison per tick (like the popular "linear vs. binary
  search" animations). Shows *why* one method needs fewer steps.
"""

import html

import numpy as np
import pandas as pd
import streamlit as st

from recommender.data import DISTRIBUTIONS, DatasetConfig, generate_keys
from recommender.viz.race import DEFAULT_RACE, MAX_RACE_SIZE
from recommender.viz.stepper import MAX_LEARNING_SIZE, TRACERS, Step, trace

from . import charts
from .common import DISTRIBUTION_LABELS, german, page_header
from .race_view import render_race, rescale

STEP_VIEW = "Schritt für Schritt"
RACE_VIEW = "Wettrennen"

# Semi-transparent colours work on the light and the dark theme alike.
CELL_CSS = """
<style>
.lm-row {display: flex; flex-wrap: wrap; gap: 4px; margin: 0.5rem 0 0.8rem 0;}
.lm-cell {width: 3.1rem; border: 1px solid rgba(128, 128, 128, 0.45); border-radius: 6px; text-align: center;
          padding: 2px 0; font-family: ui-monospace, monospace; background: rgba(128, 128, 128, 0.06);
          transition: all 0.15s ease-in-out;}
.lm-cell .lm-index {font-size: 0.65rem; opacity: 0.6;}
.lm-cell .lm-value {font-size: 0.85rem;}
.lm-out {opacity: 0.3;}
.lm-range {background: rgba(42, 120, 214, 0.16); border-color: #2a78d6;}
.lm-mark {border-bottom: 3px solid #4a3aa7;}
.lm-probe {background: rgba(235, 104, 52, 0.25); border: 2px solid #eb6834; font-weight: 700; transform: translateY(-3px);}
.lm-found {background: rgba(0, 131, 0, 0.22); border: 2px solid #008300; font-weight: 700; transform: translateY(-3px);}
.lm-legend span {display: inline-block; margin-right: 1rem; font-size: 0.8rem; opacity: 0.85;}
.lm-legend i {display: inline-block; width: 0.9rem; height: 0.9rem; border-radius: 3px; vertical-align: -2px; margin-right: 0.3rem;}
</style>
"""

LEGEND_ITEMS = [
    ("background:rgba(42,120,214,0.16);border:1px solid #2a78d6", "noch möglicher Bereich"),
    ("background:rgba(235,104,52,0.25);border:2px solid #eb6834", "gerade verglichen"),
    ("background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.45);opacity:0.4", "ausgeschlossen"),
    ("background:rgba(0,131,0,0.22);border:2px solid #008300", "gefunden"),
]
JUMP_LEGEND = ("border-bottom:3px solid #4a3aa7", "Blockende (Sprungsuche)")


def _legend(with_blocks: bool) -> str:
    items = LEGEND_ITEMS + ([JUMP_LEGEND] if with_blocks else [])
    return '<div class="lm-legend">' + "".join(f'<span><i style="{style}"></i>{label}</span>' for style, label in items) + "</div>"


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
    """Small example array with readable values 0 ... 999 (shape of the distribution kept, duplicates can arise)."""
    return rescale(generate_keys(DatasetConfig(n=n, distribution=distribution, order=order, seed=seed)))


def _step_comparison(values: np.ndarray, key: int, algorithm: str, is_sorted: bool) -> None:
    rows = [
        {"Verfahren": name, "Schritte": len(trace(k, values, key)), "aktiv": k == algorithm}
        for k, (name, _, needs_sorted) in TRACERS.items()
        if is_sorted or not needs_sorted
    ]
    st.markdown("**Wie viele Schritte brauchen die Verfahren für diesen Schlüssel?**")
    st.altair_chart(charts.step_counts(pd.DataFrame(rows)), width="stretch")
    st.caption(
        "Ein Schritt ist ein Vergleich mit einem gespeicherten Wert (bei der Interpolationssuche zählt die Schätzung als ein Schritt). "
        "Probiere Treffer und Fehlschläge, kleine und große Arrays sowie schiefe Verteilungen: Bei schiefen Daten verschätzt sich die Interpolationssuche."
        + ("" if is_sorted else " Auf unsortierten Daten funktioniert nur die lineare Suche.")
    )


def _key_controls(values: np.ndarray, n: int) -> int:
    """Search key input with buttons for a random hit and a random miss."""
    rng = np.random.default_rng()  # unseeded: every click draws a new key
    left, middle, right = st.columns([2, 1, 1], vertical_alignment="bottom")
    with middle:
        if st.button("Zufälliger Treffer", icon=":material/target:", width="stretch"):
            st.session_state.lm_key = int(values[rng.integers(n)])
            _reset()
    with right:
        if st.button("Zufälliger Fehlschlag", icon=":material/block:", width="stretch"):
            candidates = sorted(set(range(int(values.min()) - 1, int(values.max()) + 2)) - set(values.tolist()))
            st.session_state.lm_key = int(candidates[rng.integers(len(candidates))])
            _reset()
    with left:
        key = int(st.number_input("Gesuchter Wert", value=int(st.session_state.lm_key), step=1, on_change=_reset))
        st.session_state.lm_key = key
    return key


def _step_view(algorithm: str, values: np.ndarray, key: int, n: int, is_sorted: bool) -> None:
    steps = trace(algorithm, values, key)
    st.session_state.setdefault("lm_step", 0)
    position = min(st.session_state.lm_step, len(steps))

    back, forward, end, restart = st.columns(4)
    if back.button("Zurück", icon=":material/chevron_left:", disabled=position == 0, width="stretch"):
        position -= 1
    if forward.button("Weiter ▶", type="primary", disabled=position >= len(steps), width="stretch"):
        position += 1
    if end.button("Zum Ende", icon=":material/last_page:", disabled=position >= len(steps), width="stretch"):
        position = len(steps)
    if restart.button("Neustart", icon=":material/restart_alt:", width="stretch"):
        position = 0
    st.session_state.lm_step = position
    st.progress(position / len(steps), text=f"Schritt **{position}** von **{len(steps)}**")

    current = steps[position - 1] if position > 0 else None
    st.markdown(CELL_CSS + _cells(values, current) + _legend(algorithm == "jump"), unsafe_allow_html=True)
    if current is None:
        st.info(f"Start: gesucht wird {german(key)} in {n} Werten. Noch nichts verglichen.", icon=":material/flag:")
    elif current.done:
        if current.result >= 0:
            st.success(current.text, icon=":material/check_circle:")
        else:
            st.warning(current.text, icon=":material/cancel:")
    else:
        st.info(current.text, icon=":material/arrow_forward:")
    if position > 0:
        with st.expander(f"Bisherige Schritte ({position})", icon=":material/history:"):
            for number, step in enumerate(steps[:position], start=1):
                st.markdown(f"{number}. {step.text}")

    st.divider()
    _step_comparison(values, key, algorithm, is_sorted)


def render() -> None:
    page_header(6, "Verfolge eine Suche Schritt für Schritt oder lass mehrere Verfahren im Wettrennen gegeneinander antreten.")
    view = st.radio("Ansicht", [STEP_VIEW, RACE_VIEW], horizontal=True, label_visibility="collapsed", on_change=_reset)
    if view == STEP_VIEW:
        st.write("Klicke auf **Weiter**, um den nächsten Vergleich zu sehen.")
    else:
        st.write(
            "Mehrere Verfahren suchen **denselben Wert im selben Array**. In jedem Takt macht jedes Verfahren einen Schritt; "
            "wer zuerst fertig ist, gewinnt. So sieht man, warum Halbieren oder Springen so viel weniger Schritte braucht als das Durchlaufen."
        )

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            if view == STEP_VIEW:
                algorithms = [st.selectbox("Suchverfahren", list(TRACERS), format_func=lambda k: TRACERS[k][0], on_change=_reset)]
                max_size = MAX_LEARNING_SIZE
            else:
                algorithms = st.multiselect(
                    "Verfahren im Rennen",
                    list(TRACERS),
                    default=list(DEFAULT_RACE),
                    format_func=lambda k: TRACERS[k][0],
                    help="Alle Verfahren außer der linearen Suche brauchen sortierte Daten; das Beispiel-Array ist deshalb sortiert.",
                )
                max_size = min(MAX_LEARNING_SIZE, MAX_RACE_SIZE)
            n = st.slider("Größe des Beispiel-Arrays", 4, max_size, 24, on_change=_reset)
        with right:
            distribution = st.selectbox("Verteilung der Werte", DISTRIBUTIONS, format_func=DISTRIBUTION_LABELS.get, on_change=_reset)
            seed = int(st.number_input("Seed", min_value=0, value=3, step=1, on_change=_reset))
        needs_sorted = view == RACE_VIEW or TRACERS[algorithms[0]][2]
        order = (
            "sorted"
            if needs_sorted
            else st.radio("Reihenfolge (nur lineare Suche)", ["sorted", "unsorted"], format_func={"sorted": "sortiert", "unsorted": "unsortiert"}.get, horizontal=True, on_change=_reset)
        )
    values = _example_array(n, distribution, order, seed)

    # A new example array starts with a key that is contained in it.
    config = (n, distribution, order, seed)
    if st.session_state.get("lm_config") != config:
        st.session_state.lm_config = config
        st.session_state.lm_key = int(values[n // 2])
        _reset()

    key = _key_controls(values, n)
    if view == STEP_VIEW:
        _step_view(algorithms[0], values, key, n, order == "sorted")
    else:
        render_race(values, key, algorithms)
