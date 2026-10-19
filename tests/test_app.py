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
    assert any("Schritt **1**" in m.value for m in app.markdown)
