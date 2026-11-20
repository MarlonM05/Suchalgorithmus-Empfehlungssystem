"""Presentation mode – Überblick: the big picture, the key results and the stories.

The first page of the app and the opening of the talk:

1. the question of the project in one sentence,
2. how the tool works, as four steps (data -> profile -> decision -> result),
   which map onto the algorithm selection framework of Rice [rice1976],
3. the key results of the evaluation, read from results/full,
4. the stories as cards; a click opens the story page.
"""

import pandas as pd
import streamlit as st

from recommender.benchmark.experiment import RESULTS_ROOT
from recommender.rules import STRATEGIES

from .common import PAGES, german, percent
from .stories import STORIES

PIPELINE = [
    ("📦", "Daten und Anfragen", "Was liegt vor, und wie oft wird wonach gesucht?", "Problemraum"),
    ("🔍", "Steckbrief", "Eine kleine Stichprobe verrät: sortiert? gleichmäßig? groß?", "Merkmalsraum"),
    ("🧭", "Entscheidung", "Ein Regelwerk aus unseren Messungen wählt die Strategie.", "Auswahlabbildung"),
    ("🏁", "Ergebnis", "Die Suche läuft und wird gemessen, zum Vergleich auch alle anderen.", "Leistungsmaß"),
]


@st.cache_data
def _evaluation() -> pd.DataFrame | None:
    """Rule book and single best solver on the test half, including analysis cost."""
    path = RESULTS_ROOT / "full" / "recommender_summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table = table[(table["Bereich"] == "alle") & (table["Analysekosten"] == "ja")]
    return table.set_index("Selektor")


def _pipeline() -> None:
    st.markdown("### So arbeitet das Tool")
    columns = st.columns([4, 1, 4, 1, 4, 1, 4], vertical_alignment="center")
    for index, (icon, title, text, rice) in enumerate(PIPELINE):
        with columns[2 * index].container(border=True, height="stretch"):
            st.markdown(f"<div style='font-size:2.6rem;line-height:1'>{icon}</div>", unsafe_allow_html=True)
            st.markdown(f"**{index + 1} · {title}**")
            st.caption(text)
            st.caption(f"*Rice: {rice}*")
        if index < len(PIPELINE) - 1:
            columns[2 * index + 1].markdown("<div style='font-size:2rem;text-align:center;opacity:0.6'>➜</div>", unsafe_allow_html=True)


def _results() -> None:
    table = _evaluation()
    if table is None or "Regelwerk" not in table.index:
        return
    rules = table.loc["Regelwerk"]
    sbs = table.loc[[name for name in table.index if name.startswith("Single Best")][0]]
    st.markdown("### Was es bringt")
    hit, slow, gap = st.columns(3)
    hit.metric("Richtige Wahl", percent(rules["Trefferquote"], 0), border=True, help="Anteil der Testfälle, in denen das Regelwerk genau die schnellste Strategie wählt.")
    hit.caption(f"Immer dieselbe Strategie: {percent(sbs['Trefferquote'], 0)}")
    slow.metric("Abstand zum Optimum", f"{german(round(rules['Slowdown geom. Mittel'], 2))}×", border=True, help="Geometrisches Mittel: so viel langsamer als die jeweils beste Strategie, einschließlich der Kosten der Stichprobe.")
    slow.caption(f"Immer dieselbe Strategie: {german(round(sbs['Slowdown geom. Mittel'], 2))}×")
    gap.metric("Lücke geschlossen", percent(rules["Lücke geschlossen (geom.)"], 0), border=True, help="Wie viel des Abstands zwischen 'immer dieselbe Strategie' und dem Optimum das Regelwerk aufholt.")
    gap.caption("zwischen bester Einzelstrategie und Optimum")
    st.caption(f"Bewertet auf {german(int(rules['Instanzen']))} Testfällen des Messrasters, die nicht zum Entwickeln der Regeln verwendet wurden (results/full).")


def _story_cards() -> None:
    st.markdown("### Sechs Geschichten")
    st.caption("Jede Geschichte führt zu einer anderen Empfehlung. Klick auf eine Karte, um sie Schritt für Schritt zu erleben.")
    stories = list(STORIES.values())
    for row in range(0, len(stories), 3):
        for column, story in zip(st.columns(3), stories[row : row + 3]):
            with column.container(border=True, height="stretch"):
                st.markdown(f"<div style='font-size:2.4rem;line-height:1.1'>{story.icon}</div>", unsafe_allow_html=True)
                st.markdown(f"**{story.title}**")
                st.caption(story.hook)
                if st.button("Geschichte ansehen", key=f"open_{story.key}", icon=":material/play_arrow:", width="stretch"):
                    st.session_state.story = story.key
                    if "story" in PAGES:
                        st.switch_page(PAGES["story"])


def _footer() -> None:
    st.divider()
    left, right = st.columns(2)
    if 1 in PAGES:
        left.page_link(PAGES[1], label="Selbst ausprobieren: das Werkzeug in fünf Schritten", icon=":material/build:")
    if 6 in PAGES:
        right.page_link(PAGES[6], label="Suchverfahren Schritt für Schritt verstehen: der Lernmodus", icon=":material/school:")


def render() -> None:
    st.title("Welches Suchverfahren passt zu meinen Daten?")
    st.markdown(
        "#### Linear, binär, Hashtabelle oder etwas ganz anderes? Das hängt von den Daten und davon ab, "
        "wie oft gesucht wird. Dieses Tool schaut sich beides an und empfiehlt die passende Strategie."
    )
    searches = len({s.search for s in STRATEGIES.values()})
    st.caption(f"{searches} Suchverfahren · {len(STRATEGIES)} Strategien mit und ohne Vorbereitung · jede Regel mit Begründung und Quelle")
    _pipeline()
    _results()
    _story_cards()
    _footer()
