"""Shared state, layout helpers and navigation of the Streamlit pages.

Session state (shared by all pages)
-----------------------------------
* ``data``: the current dataset (int64 array or LinkedList) and a short
  ``data_description``; ``data_version`` changes whenever the data changes,
  so results computed for old data are never reused,
* ``workload``: the user's statements about the queries (Workload),
* ``search_key``: the key searched on the execution page,
* ``sample_size``: sample size of the analysis (default 1 000),
* ``features`` / ``features_sample_size`` / ``sort_check``: results of the
  analysis page, computed on demand,
* ``selector``: rule book or learned tree,
* ``execution`` / ``comparison``: cached results of the execution page,
  stored together with the state they were computed for.

Layout
------
Every page starts with :func:`page_header` (progress bar of the five work
steps plus title) and ends with :func:`next_step` (link to the next page), so
the workflow data -> access pattern -> analysis -> recommendation ->
execution is visible everywhere.
"""

import numpy as np
import streamlit as st

from recommender.algorithms import ALGORITHMS, LIST_ALGORITHMS
from recommender.benchmark.experiment import RESULTS_ROOT
from recommender.data import DatasetConfig, LinkedList, generate_dataset, linked_list_to_array
from recommender.executor import warm_up
from recommender.features import DEFAULT_SAMPLE_SIZE, DatasetFeatures, extract_features
from recommender.rules import STRATEGIES, RuleBook, Workload, selection_features
from recommender.rules.decision import Recommendation
from recommender.rules.learned import LearnedSelector
from recommender.viz.formatting import format_ns, german, percent

__all__ = ["format_ns", "german", "percent"]  # re-exported for the pages

DISTRIBUTION_LABELS = {
    "uniform": "gleichverteilt",
    "normal": "normalverteilt",
    "exponential": "exponentiell",
    "lognormal": "lognormal (µ = 0, σ = 2)",
    "clustered": "geclustert",
}
ORDER_LABELS = {"sorted": "sortiert", "nearly_sorted": "fast sortiert", "unsorted": "unsortiert"}
STRUCTURE_LABELS = {"array": "Array (Direktzugriff)", "linked_list": "Verkettete Liste (nur sequenziell)"}
PREPROCESSING_LABELS = {
    None: "keine",
    "sort": "Kopie sortieren, O(n log n)",
    "hash_build": "Hashtabelle aufbauen, O(n)",
    "list_hash_build": "Liste in Hashtabelle kopieren, O(n)",
}
RULES = "Regelwerk"
TREE = "Gelernter Entscheidungsbaum"

# number -> (title, Material icon, one-line description); 1-5 are the workflow, 6 is separate
STEPS = {
    1: ("Daten", ":material/database:", "Datenmenge erzeugen oder hochladen"),
    2: ("Zugriffsmuster", ":material/manage_search:", "Suchschlüssel und erwartete Anfragen"),
    3: ("Analyse", ":material/query_stats:", "Merkmale per Stichprobe schätzen"),
    4: ("Empfehlung", ":material/account_tree:", "Strategie und Entscheidungsweg"),
    5: ("Ausführung", ":material/play_circle:", "Suche ausführen und vergleichen"),
    6: ("Lernmodus", ":material/school:", "Suchverfahren Schritt für Schritt"),
}
WORKFLOW = [1, 2, 3, 4, 5]

# Filled by streamlit_app.py with the registered st.Page objects. Pages rendered
# on their own (as in the tests) have no navigation, then no links are drawn.
PAGES: dict[int, "st.Page"] = {}


# --- Layout ------------------------------------------------------------------------------

_STYLE = """
<style>
.sr-steps {display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 0.6rem 0;}
.sr-step {font-size: 0.78rem; padding: 0.15rem 0.65rem; border-radius: 999px;
          border: 1px solid rgba(128, 128, 128, 0.35); opacity: 0.75; white-space: nowrap;}
.sr-step.sr-done {border-color: #2a78d6; color: #2a78d6; opacity: 1;}
.sr-step.sr-active {background: #2a78d6; border-color: #2a78d6; color: #ffffff; opacity: 1; font-weight: 600;}
</style>
"""


def page_header(number: int, lead: str) -> None:
    """Progress bar of the workflow, page title and a one-sentence introduction."""
    st.markdown(_STYLE, unsafe_allow_html=True)
    if number in WORKFLOW:
        chips = []
        for step in WORKFLOW:
            state = "sr-active" if step == number else ("sr-done" if step < number else "")
            mark = "✓ " if step < number else ""
            chips.append(f'<span class="sr-step {state}">{mark}{step} · {STEPS[step][0]}</span>')
        st.markdown(f'<div class="sr-steps">{"".join(chips)}</div>', unsafe_allow_html=True)
    st.title(f"{number} · {STEPS[number][0]}")
    st.markdown(lead)  # plain Markdown, so **bold** in the introduction works


def next_step(number: int) -> None:
    """Link to the following work step at the end of a page (only inside the app)."""
    following = number + 1
    if following not in PAGES or following not in STEPS:
        return
    st.divider()
    title, icon, description = STEPS[following]
    st.page_link(PAGES[following], label=f"Weiter zu {following} · {title}: {description}", icon=icon)


def strategy_properties(strategy_key: str) -> dict[str, str]:
    """What the user should know about a strategy (for cards and tables)."""
    strategy = STRATEGIES[strategy_key]
    algorithm = (ALGORITHMS | LIST_ALGORITHMS)[strategy.search]
    return {
        "Suchverfahren": algorithm.name_de,
        "Laufzeit pro Anfrage": algorithm.complexity,
        "Vorverarbeitung": PREPROCESSING_LABELS[strategy.preprocessing],
        "Braucht sortierte Daten": "ja (wird vorher geprüft)" if strategy.requires_sorted else "nein",
        "Quellen": ", ".join(f"[{s}]" for s in algorithm.sources),
    }


# --- Cached resources ----------------------------------------------------------------------


@st.cache_resource(show_spinner="Suchverfahren werden einmalig kompiliert …")
def compiled() -> bool:
    """Compile all search functions once per server process (Numba)."""
    warm_up()
    return True


@st.cache_resource
def rulebook() -> RuleBook:
    return RuleBook.load()


@st.cache_resource
def learned_tree() -> LearnedSelector | None:
    """The tree trained in the evaluation (results/full, else results/quick)."""
    for mode in ("full", "quick"):
        path = RESULTS_ROOT / mode / "learned_tree.json"
        if path.exists():
            return LearnedSelector.from_json(path)
    return None


# --- Session state -------------------------------------------------------------------------


def init_state() -> None:
    """Default dataset and workload on the first visit."""
    if "data" not in st.session_state:
        config = DatasetConfig(n=100_000, distribution="uniform", order="sorted", seed=1)
        set_data(generate_dataset(config), "100.000 gleichverteilte Werte, sortiert, Array (Standard)")
    if "workload" not in st.session_state:
        st.session_state.workload = Workload(queries=100)
    st.session_state.setdefault("sample_size", DEFAULT_SAMPLE_SIZE)


def invalidate_results() -> None:
    """Forget the results of the execution page (after the data or workload changed)."""
    for key in ("execution", "comparison"):
        st.session_state.pop(key, None)


def set_data(data: np.ndarray | LinkedList, description: str) -> None:
    """Replace the dataset and invalidate everything derived from it."""
    st.session_state.data = data
    st.session_state.data_description = description
    st.session_state.data_version = st.session_state.get("data_version", 0) + 1
    for key in ("features", "features_sample_size", "sort_check"):
        st.session_state.pop(key, None)
    invalidate_results()
    values = keys_in_order(data)
    st.session_state.search_key = int(values[len(values) // 3]) if len(values) else 0


def keys_in_order(data: np.ndarray | LinkedList) -> np.ndarray:
    """The keys in logical order (for lists: following the pointers)."""
    return linked_list_to_array(data) if isinstance(data, LinkedList) else data


def structure_of(data) -> str:
    return "linked_list" if isinstance(data, LinkedList) else "array"


def index_text(index: int) -> str:
    """Search result in words: array index or list position (0 = first element)."""
    if index < 0:
        return "nicht gefunden"
    word = "Position" if isinstance(st.session_state.data, LinkedList) else "Index"
    return f"{word} {german(index)}"


def current_features() -> DatasetFeatures:
    """Features of the current data with the chosen sample size.

    Computed on demand and kept until the data or the sample size changes, so
    every page (analysis, recommendation, execution) sees the same values.
    """
    sample_size = st.session_state.get("sample_size", DEFAULT_SAMPLE_SIZE)
    if "features" not in st.session_state or st.session_state.get("features_sample_size") != sample_size:
        st.session_state.features = extract_features(st.session_state.data, sample_size=sample_size)
        st.session_state.features_sample_size = sample_size
    return st.session_state.features


def selector_for(name: str):
    """The rule book or, if available, the learned tree."""
    if name == TREE and learned_tree() is not None:
        return learned_tree()
    return rulebook()


def recommend(selector: str) -> Recommendation:
    """Recommendation of the chosen selector for the current data and workload."""
    features = selection_features(current_features(), st.session_state.workload)
    return selector_for(selector).decide(features)


def sidebar_summary() -> None:
    """Current data and workload in the sidebar, visible on every page."""
    data = st.session_state.data
    workload = st.session_state.workload
    with st.sidebar:
        with st.container(border=True):
            st.markdown("**:material/database: Aktuelle Daten**")
            st.caption(st.session_state.data_description)
            st.markdown("**:material/manage_search: Zugriffsmuster**")
            position = "vorne gehäuft" if workload.position == "front" else "gleichverteilt"
            st.caption(
                f"{german(workload.queries)} Anfragen · Trefferquote {percent(workload.hit_ratio, 0)} · "
                f"Treffer {position}" + (" · Sortierung garantiert" if workload.sorted_known else "")
            )
            # len() works for arrays and LinkedList alike and does not walk the list.
            st.caption(f"Suchschlüssel: {german(st.session_state.search_key)} · n = {german(len(data))}")
