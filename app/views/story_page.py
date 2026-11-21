"""Presentation mode – Geschichten: one everyday story per page, told in six chapters.

Every chapter answers one question of the talk with a picture first and a
sentence second:

1. Die Daten – what do the data look like?
2. Was das Tool erkennt – the profile from the sample
3. Die Entscheidung – the path through the rule book
4. Das Rennen – the race on a small excerpt, plus counts on all data
5. Was wäre wenn? – cost curves over the number of queries, with a slider
6. Die Lehre – the take-away in one or two sentences

All numbers are computed live on the story's data (measured on this
computer), the texts of the stories are checked against the rule book by the
tests (``tests/test_presentation.py``).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from recommender.algorithms import ALGORITHMS, LIST_ALGORITHMS
from recommender.data import LinkedList, QueryConfig, generate_queries, linked_list_to_array
from recommender.executor import ExecutionResult, compare_all
from recommender.features import DatasetFeatures, extract_features
from recommender.rules import STRATEGIES, Workload, selection_features
from recommender.rules.decision import Recommendation
from recommender.viz.graph import path_dot, rulebook_alternatives

from . import charts
from .common import DISTRIBUTION_LABELS, ORDER_LABELS, PAGES, compiled, format_ns, german, rulebook, set_data
from .race_view import excerpt, render_race, rescale
from .stories import STORIES, Story
from .story_texts import agreement_text, answer_parts, data_text, profile_cards

RACE_SIZE = 64
COUNT_QUERIES = 200  # searches per algorithm for the comparison counts on all data
QUERY_STEPS = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 1_000_000, 10_000_000]


@dataclass(frozen=True)
class StoryResults:
    """Everything computed once per story (cached)."""

    data: np.ndarray | LinkedList
    keys: np.ndarray  # logical order
    features: DatasetFeatures
    recommendation: Recommendation
    search_key: int
    measured: tuple[ExecutionResult, ...]
    counts: pd.DataFrame
    race_values: np.ndarray
    race_key: int


def _count_algorithms(story: Story, recommendation: Recommendation):
    """Algorithms for the comparison counts: linear search as baseline, the race, the recommendation and the hash table."""
    is_list = story.dataset.structure == "linked_list"
    registry = LIST_ALGORITHMS if is_list else ALGORITHMS
    baseline = "list_linear" if is_list else "linear"  # always shown: the method without any preparation
    keys = [baseline] + ["list_" + k if is_list else k for k in story.race]
    keys.append(STRATEGIES[recommendation.strategy].search)
    if story.dataset.order != "sorted" or is_list:
        keys.append("hash")
    seen, result = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            result.append((key, (registry | ALGORITHMS)[key]))
    return result


def _counts(story: Story, data, keys: np.ndarray, recommendation: Recommendation) -> pd.DataFrame:
    """Mean comparisons per search over COUNT_QUERIES searches that follow the story's access pattern."""
    workload = story.workload
    queries = generate_queries(keys, QueryConfig(count=COUNT_QUERIES, hit_ratio=workload.hit_ratio, position=workload.position, seed=story.dataset.seed))
    recommended_search = STRATEGIES[recommendation.strategy].search
    rows = []
    for key, algorithm in _count_algorithms(story, recommendation):
        prepared = algorithm.prepare(keys if key == "hash" else data)
        mean = float(np.mean([algorithm.find_counted(prepared, int(q)).comparisons for q in queries]))
        name = algorithm.name_de + (" (nach Kopie der Liste)" if key == "hash" and isinstance(data, LinkedList) else "")
        rows.append({"Verfahren": name, "Vergleiche": mean, "empfohlen": key == recommended_search})
    return pd.DataFrame(rows)


def compute_story_results(key: str, selector) -> StoryResults:
    """Generate, analyse and measure one story (no Streamlit: also used by presentation/build_site.py)."""
    story = STORIES[key]
    data = story.data()
    keys = linked_list_to_array(data) if isinstance(data, LinkedList) else data
    features = extract_features(data)
    recommendation = selector.decide(selection_features(features, story.workload))
    search_key = story.search_key(keys)
    measured = tuple(compare_all(data, search_key, story.workload.sorted_known))
    if story.dataset.order == "sorted" and story.dataset.structure == "array":
        race_values = rescale(excerpt(keys, RACE_SIZE, seed=0))
    else:
        race_values = rescale(keys[:RACE_SIZE])  # stored order: the race shows the linear search walking through it
    race_key = int(race_values[int(story.workload.scan_fraction * (race_values.size - 1))])
    return StoryResults(data, keys, features, recommendation, search_key, measured, _counts(story, data, keys, recommendation), race_values, race_key)


@st.cache_resource(show_spinner="Die Geschichte wird vorbereitet: Daten erzeugen, analysieren und alle Strategien messen …")
def story_results(key: str) -> StoryResults:
    compiled()
    return compute_story_results(key, rulebook())


# --- Chapters ------------------------------------------------------------------------------


def _chapter(number: int, title: str, question: str) -> None:
    st.markdown(f"### {number} · {title}")
    st.caption(question)


def _header(story: Story) -> None:
    dataset, workload = story.dataset, story.workload
    st.title(f"{story.icon} {story.title}")
    st.markdown(f"#### {story.scenario}")
    position = "die gesuchten Einträge liegen vorne" if workload.position == "front" else "jeder Eintrag gleich wahrscheinlich"
    structure = "verkettete Liste" if dataset.structure == "linked_list" else "Array"
    st.markdown(
        f":blue-badge[n = {german(dataset.n)}] :blue-badge[{structure}] :blue-badge[{ORDER_LABELS[dataset.order]}] "
        f":blue-badge[{DISTRIBUTION_LABELS[dataset.distribution]}] :orange-badge[{german(workload.queries)} Anfragen] :orange-badge[{position}]"
    )


def _data_chapter(story: Story, results: StoryResults) -> None:
    _chapter(1, "Die Daten", "Wie sehen die Daten aus, die durchsucht werden sollen?")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(charts.value_over_position(results.keys), width="stretch")
    with right:
        for line in data_text(story):
            st.markdown(line)
        first = ", ".join(german(int(v)) for v in results.keys[:8])
        st.caption(f"Die ersten Werte: {first} …")


def _profile_chapter(story: Story, results: StoryResults) -> None:
    _chapter(2, "Was das Tool erkennt", "Der Steckbrief der Daten: aus einer kleinen Stichprobe geschätzt, dazu deine Angaben zu den Anfragen.")
    features = results.features
    for column, card in zip(st.columns(4), profile_cards(features, story)):
        column.metric(card.label, card.value, border=True, help=card.help or None)
        column.caption(card.verdict)
    used = "gebraucht" if results.recommendation.uses_sampled_features else "nicht gebraucht: Größe und deine Angaben reichten"
    st.caption(f"Die Stichprobe kostete {format_ns(features.analysis_time_ns)} und wurde für die Entscheidung {used}.")


def _answer(step) -> str:
    """The answer of one decision step with the measured value, as a coloured badge."""
    yes, value = answer_parts(step)
    badge = ":green-badge[ja]" if yes else ":gray-badge[nein]"
    return f"{badge} :gray[{value}]" if value else badge


def _decision_chapter(results: StoryResults) -> None:
    _chapter(3, "Die Entscheidung", "Welche Fragen stellt das Regelwerk, und wie werden sie beantwortet?")
    recommendation = results.recommendation
    with st.container(border=True):
        st.markdown(f"Empfehlung: **{STRATEGIES[recommendation.strategy].name_de}**")
        st.markdown(f"##### {recommendation.reason}")
        if recommendation.sources:
            st.caption("Quellen: " + ", ".join(f"[{s}]" for s in recommendation.sources))
    st.markdown("**Der Weg dorthin, Frage für Frage:**")
    for number, step in enumerate(recommendation.path, start=1):
        question, answer = st.columns([3, 2], vertical_alignment="center")
        question.markdown(f"**{number}.** {step.question}")
        answer.markdown(_answer(step))
    st.markdown(f"➜ **{STRATEGIES[recommendation.strategy].name_de}**")
    with st.expander("Als Entscheidungsgraph zeigen"):
        st.graphviz_chart(path_dot(recommendation, rulebook_alternatives(rulebook(), recommendation)), width="stretch")
        st.caption("Blau ist der Weg dieser Geschichte. Gestrichelt steht, was bei der anderen Antwort passiert wäre.")


def _race_chapter(story: Story, results: StoryResults) -> None:
    _chapter(4, "Das Rennen", "Wer findet den Eintrag schneller? Erst anschaulich im Kleinen, dann gezählt auf allen Daten.")
    if story.race:
        st.markdown(f"**Im Kleinen:** {RACE_SIZE if results.race_values.size >= RACE_SIZE else results.race_values.size} Werte aus den Daten, gesucht wird {german(results.race_key)}.")
        render_race(results.race_values, results.race_key, list(story.race), autoplay=True)
    else:
        st.info("Die Hashtabelle springt direkt an die richtige Stelle. Ein Rennen über ein Array lässt sich für sie nicht zeigen, die gezählten Vergleiche unten schon.")
    st.markdown(f"**Auf allen {german(story.dataset.n)} Werten:** Vergleiche pro Suche, gemittelt über {COUNT_QUERIES} Suchen mit dem Zugriffsmuster der Geschichte.")
    st.altair_chart(charts.mean_comparisons(results.counts), width="stretch")
    if story.race:
        st.caption("Das kleine Rennen hängt davon ab, welche Werte im Ausschnitt landen. Belastbar sind die gezählten Vergleiche auf allen Daten.")


def _curve_strategies(story: Story, results: StoryResults) -> list[ExecutionResult]:
    """At most four lines: linear search, the recommendation, the hash table and the best sorted alternative."""
    by_key = {r.strategy: r for r in results.measured if not r.check_failed}
    if story.dataset.structure == "linked_list":
        wanted = ["list_linear", "list_hash"]
    elif story.dataset.order == "sorted":
        wanted = ["linear", "hash", "binary_branchless"]  # sorting already sorted data would be pointless
    else:
        wanted = ["linear", "hash", "sort_binary"]
    wanted.insert(0, results.recommendation.strategy)
    chosen = []
    for key in wanted:
        if key in by_key and by_key[key] not in chosen:
            chosen.append(by_key[key])
    return chosen[:4]


def _what_if_chapter(story: Story, results: StoryResults) -> None:
    _chapter(5, "Was wäre wenn?", "Wie ändert sich die beste Wahl, wenn mehr oder weniger gesucht wird?")
    steps = sorted(set(QUERY_STEPS) | {story.workload.queries})  # the story's own q is always selectable
    queries = st.select_slider("Anzahl Anfragen q", options=steps, value=story.workload.queries, format_func=german, key=f"whatif_{story.key}")
    workload = Workload(queries=queries, hit_ratio=story.workload.hit_ratio, position=story.workload.position, sorted_known=story.workload.sorted_known)
    recommendation = rulebook().decide(selection_features(results.features, workload))
    measured = [r for r in results.measured if not r.check_failed]
    fastest = min(measured, key=lambda r: r.total_ns(queries))
    shown = _curve_strategies(story, results)
    if all(r.strategy != recommendation.strategy for r in shown):
        shown.append(next(r for r in measured if r.strategy == recommendation.strategy))
    frame = pd.DataFrame(
        {
            "Strategie": [STRATEGIES[r.strategy].name_de for r in shown],
            "prep_ns": [r.check_ns + r.preprocessing_ns for r in shown],
            "query_ns": [r.query_ns for r in shown],
            "empfohlen": [r.strategy == recommendation.strategy for r in shown],
        }
    )
    st.altair_chart(charts.cost_curves(frame, queries), width="stretch")
    chosen = next(r for r in measured if r.strategy == recommendation.strategy)
    ratio = chosen.total_ns(queries) / fastest.total_ns(queries)
    agreement = agreement_text(STRATEGIES[recommendation.strategy].name_de, STRATEGIES[fastest.strategy].name_de, ratio)
    verdict, fact = st.columns([3, 2])
    with verdict.container(border=True, height="stretch"):
        st.caption(f"Empfehlung bei q = {german(queries)}")
        st.markdown(f"#### {STRATEGIES[recommendation.strategy].name_de}")
        st.caption(agreement)
    with fact.container(border=True, height="stretch"):
        st.caption("Geschätzte Gesamtzeit")
        st.markdown(f"#### {format_ns(chosen.total_ns(queries))}")
    st.caption(
        "Jede Linie ist Vorbereitung + q · Zeit pro Anfrage, gemessen auf den Daten dieser Geschichte auf diesem Rechner. "
        "Wo sich Linien kreuzen, lohnt sich die Vorbereitung (Break-even)."
    )


def _lesson_chapter(story: Story) -> None:
    _chapter(6, "Die Lehre", "Was diese Geschichte zeigt.")
    with st.container(border=True):
        st.markdown(f"#### 💡 {story.lesson}")


def _open_in_tool(story: Story, results: StoryResults) -> None:
    if 4 not in PAGES:
        return
    if st.button("Diese Geschichte im Werkzeug weiter untersuchen", icon=":material/build:", type="primary"):
        set_data(story.data(), f"Geschichte „{story.title}“: {german(story.dataset.n)} Werte, {DISTRIBUTION_LABELS[story.dataset.distribution]}, {ORDER_LABELS[story.dataset.order]}")
        st.session_state.workload = story.workload
        st.session_state.search_key = results.search_key
        st.switch_page(PAGES[4])


def _story_picker() -> Story:
    keys = list(STORIES)
    current = st.session_state.get("story", keys[0])
    choice = st.pills(
        "Geschichte",
        keys,
        default=current if current in keys else keys[0],
        format_func=lambda k: f"{STORIES[k].icon} {STORIES[k].title}",
        label_visibility="collapsed",
    )
    st.session_state.story = choice or current
    return STORIES[st.session_state.story]


def render() -> None:
    story = _story_picker()
    results = story_results(story.key)
    _header(story)
    st.divider()
    _data_chapter(story, results)
    st.divider()
    _profile_chapter(story, results)
    st.divider()
    _decision_chapter(results)
    st.divider()
    _race_chapter(story, results)
    st.divider()
    _what_if_chapter(story, results)
    st.divider()
    _lesson_chapter(story)
    _open_in_tool(story, results)
