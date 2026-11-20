"""Shared "Wettrennen" block: the animated race plus a table of exact counts.

Used by the learning mode (free choice of algorithms) and by the
recommendation page (linear search against the recommended method on a
small excerpt of the current data).
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from recommender.data import rescale_monotone
from recommender.viz.race import MAX_RACE_SIZE, build_lanes, race_height, race_html
from recommender.viz.stepper import TRACERS

from .common import german

# Strategies of the recommender -> algorithm shown in the race (None: not traceable).
# The branchless binary search probes a slightly different sequence than the
# classic one; the classic trace is shown as its stand-in and labelled so.
RACE_STAND_IN = {
    "linear": "linear",
    "binary": "binary",
    "binary_branchless": "binary",
    "sort_binary": "binary",
    "jump": "jump",
    "exponential": "exponential",
    "fibonacci": "fibonacci",
    "interpolation": "interpolation",
    "list_linear": "linear",
    "list_jump": "jump",
    "hash": None,
    "list_hash": None,
}


def rescale(values: np.ndarray) -> np.ndarray:
    """Map the values onto about 0 ... 999: keeps order, shape and the exact duplicates (A39)."""
    return rescale_monotone(values, 999)


def excerpt(values: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """A sorted random excerpt of at most ``size`` values (the race needs sorted data)."""
    values = np.asarray(values, dtype=np.int64)
    if values.size > size:
        values = np.random.default_rng(seed).choice(values, size, replace=False)
    return np.sort(values)


def render_race(values: np.ndarray, key: int, algorithms: list[str], autoplay: bool = False) -> None:
    """Animated race of ``algorithms`` on ``values`` plus the exact operation counts."""
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0 or values.size > MAX_RACE_SIZE:
        st.warning(f"Das Wettrennen zeigt 1 bis {MAX_RACE_SIZE} Werte.")
        return
    if not algorithms:
        st.info("Wähle mindestens ein Suchverfahren.")
        return
    lanes = build_lanes(values, key, algorithms)
    page, height = race_html(values, key, lanes, autoplay=autoplay), race_height(lanes, int(values.size))
    # st.iframe replaces components.html (deprecated); older Streamlit versions only have the latter.
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height, scrolling=False)

    order = sorted(lanes, key=lambda lane: lane.step_count)
    table = pd.DataFrame(
        {
            "Platz": range(1, len(order) + 1),
            "Verfahren": [lane.name for lane in order],
            "Schritte im Rennen": [lane.step_count for lane in order],
            "Vergleiche (gezählt)": [lane.comparisons for lane in order],
            "Gelesene Elemente (gezählt)": [lane.accesses for lane in order],
            "Ergebnis": [f"Index {german(lane.result)}" if lane.result >= 0 else "nicht gefunden" for lane in order],
            "Komplexität": [lane.complexity for lane in order],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Ein Schritt im Rennen ist ein Zwischenstand der Suche; die gezählten Vergleiche und Zugriffe stammen aus der "
        "kompilierten Zählvariante desselben Verfahrens (Annahme A3). Bei n Werten braucht die lineare Suche bis zu n Schritte, "
        "die Halbierungsverfahren etwa log₂(n)."
    )


def traceable(strategy_key: str) -> str | None:
    """Algorithm key for the race, or None if the strategy cannot be traced (hash lookup)."""
    algorithm = RACE_STAND_IN.get(strategy_key)
    return algorithm if algorithm in TRACERS else None
