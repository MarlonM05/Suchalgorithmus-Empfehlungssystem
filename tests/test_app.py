"""Smoke tests of the Streamlit app (milestone M5).

Every page is rendered with Streamlit's testing framework and must not raise.
The navigation registers pages as functions, which ``AppTest.switch_page``
cannot address, so each page is rendered through a small script that sets
up the shared state and calls the page function directly.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["data_page", "workload_page", "analysis_page", "recommendation_page", "execution_page", "learning_page"]
TIMEOUT = 180  # the first run compiles all search functions


def page_app(page: str, setup: str = "") -> AppTest:
    script = f"""
import sys
sys.path.insert(0, {str(APP_DIR)!r})
from views.common import init_state
from views import {page}
init_state()
{setup}
{page}.render()
"""
    return AppTest.from_string(script, default_timeout=TIMEOUT)


def test_main_app_starts():
    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not app.exception
    assert app.title[0].value == "1 · Daten"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    app = page_app(page).run()
    assert not app.exception, app.exception


def test_execution_finds_the_default_key():
    app = page_app("execution_page").run()
    assert not app.exception
    assert any("Index" in str(metric.value) for metric in app.metric)


@pytest.mark.parametrize(
    "setup",
    [
        # unsorted list, many queries -> list strategies and hash copy
        "from recommender.data import DatasetConfig, generate_dataset\n"
        "from views.common import set_data\n"
        "set_data(generate_dataset(DatasetConfig(n=500, order='unsorted', structure='linked_list')), 'Testliste')",
        # tree selector
        "import streamlit as st\nst.session_state.selector = 'Gelernter Entscheidungsbaum'",
    ],
    ids=["linked_list", "tree"],
)
def test_recommendation_and_execution_variants(setup):
    for page in ("recommendation_page", "execution_page"):
        app = page_app(page, setup).run()
        assert not app.exception, app.exception


def test_comparison_mode():
    app = page_app("execution_page").run()
    app.button[0].click().run()  # "Alle Verfahren vergleichen"
    assert not app.exception
    assert app.dataframe, "comparison table is shown"


def test_learning_mode_steps_forward():
    app = page_app("learning_page").run()
    forward = next(b for b in app.button if "Weiter" in b.label)
    forward.click().run()
    assert not app.exception
    # app.get works in all Streamlit versions (the app.progress shortcut does not exist everywhere)
    assert "Schritt **1**" in app.get("progress")[0].proto.text


def test_learning_mode_new_array_starts_with_a_hit():
    app = page_app("learning_page").run()
    app.slider[0].set_value(40).run()  # array size: a new example array
    assert not app.exception
    assert app.session_state.lm_step == 0
    assert app.session_state.lm_config[0] == 40
    # The key of a new array is contained in it, so the first run ends with a hit.
    next(b for b in app.button if b.label == "Zum Ende").click().run()
    assert app.success, "the search ends with a hit"


def test_guaranteed_but_wrong_sortedness_is_reported():
    setup = (
        "import streamlit as st\n"
        "from recommender.data import DatasetConfig, generate_dataset\n"
        "from recommender.rules import Workload\n"
        "from views.common import set_data\n"
        "set_data(generate_dataset(DatasetConfig(n=100_000, order='unsorted', seed=4)), 'unsortiert')\n"
        "st.session_state.workload = Workload(queries=100, sorted_known=True)"
    )
    workload = page_app("workload_page", setup).run()
    assert not workload.exception
    assert any("nicht sortiert" in w.value for w in workload.warning)
    execution = page_app("execution_page", setup).run()
    assert not execution.exception
    assert any("falsch" in e.value for e in execution.error)


def test_execution_result_is_verified():
    app = page_app("execution_page").run()
    assert any("Ergebnis geprüft" in s.value for s in app.success)


def test_learning_mode_race_view():
    app = page_app("learning_page").run()
    app.radio[0].set_value("Wettrennen").run()
    assert not app.exception, app.exception
    assert app.dataframe, "ranking table of the race is shown"
    assert "Platz" in str(app.dataframe[0].value.columns)


def test_recommendation_page_shows_race():
    app = page_app("recommendation_page").run()
    assert not app.exception, app.exception
    assert any("Wettrennen" in h.value for h in app.subheader)


def test_workload_random_miss_is_really_absent():
    app = page_app("workload_page").run()
    miss = next(b for b in app.button if "Fehlschlag" in b.label)
    miss.click().run()
    assert not app.exception
    data = app.session_state["data"]
    assert int(app.session_state["search_key"]) not in set(data.tolist())
