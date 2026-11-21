"""Tests for the presentation mode: the stories and their pages.

The story texts make claims ("leads to the hash table", "interpolation search
is slower here"). These tests check the claims against the rule book and the
algorithms, so a change elsewhere cannot silently turn a story into a wrong one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from recommender.algorithms import ALGORITHMS  # noqa: E402
from recommender.data import LinkedList, linked_list_to_array  # noqa: E402
from recommender.features import extract_features, ks_distance_to_uniform  # noqa: E402
from recommender.rules import STRATEGIES, RuleBook, selection_features  # noqa: E402
from recommender.viz.race import build_lanes  # noqa: E402
from recommender.viz.stepper import TRACERS  # noqa: E402
from test_app import page_app  # noqa: E402
from views.race_view import excerpt, rescale  # noqa: E402
from views.stories import STORIES  # noqa: E402

ALL_STORIES = list(STORIES.values())


@pytest.fixture(scope="module")
def rulebook() -> RuleBook:
    return RuleBook.load()


@pytest.mark.parametrize("story", ALL_STORIES, ids=lambda s: s.key)
def test_story_leads_to_the_recommendation_its_text_claims(story, rulebook):
    data = story.data()
    recommendation = rulebook.decide(selection_features(extract_features(data), story.workload))
    assert recommendation.strategy == story.expected
    assert STRATEGIES[story.expected].structure == story.dataset.structure


def test_stories_cover_different_recommendations():
    assert len({s.expected for s in ALL_STORIES}) >= 4


@pytest.mark.parametrize("story", ALL_STORIES, ids=lambda s: s.key)
def test_race_is_possible(story):
    """Only traceable algorithms, and anything but linear search needs sorted data."""
    assert all(algorithm in TRACERS for algorithm in story.race)
    if set(story.race) - {"linear"}:
        assert story.dataset.order == "sorted" and story.dataset.structure == "array"


@pytest.mark.parametrize("story", ALL_STORIES, ids=lambda s: s.key)
def test_story_data_is_readable_and_keeps_its_shape(story):
    data = story.data()
    keys = linked_list_to_array(data) if isinstance(data, LinkedList) else data
    assert keys.size == story.dataset.n
    assert keys.max() <= 100 * keys.size  # readable, not up to 2^52
    if story.dataset.distribution == "lognormal":
        positions = np.linspace(0, keys.size - 1, 4000).astype(int)
        assert ks_distance_to_uniform(np.sort(keys)[positions]) > 0.8  # still strongly skewed


def test_skewed_story_shows_the_weakness_of_interpolation_search():
    """The race and the counts must support the lesson of the story 'Messwerte'."""
    story = STORIES["messwerte"]
    keys = story.keys()
    race = rescale(excerpt(keys, 64, seed=0))
    key = int(race[int(story.workload.scan_fraction * (race.size - 1))])
    steps = {lane.key: lane.step_count for lane in build_lanes(race, key, ["binary", "interpolation"])}
    assert steps["interpolation"] > steps["binary"]
    queries = np.random.default_rng(0).choice(keys, 100)
    mean = {name: np.mean([ALGORITHMS[name].find_counted(keys, int(q)).comparisons for q in queries]) for name in ("binary", "interpolation")}
    assert mean["interpolation"] > 5 * mean["binary"]


def test_start_page_shows_all_stories():
    app = page_app("start_page").run()
    assert not app.exception
    assert sum("Geschichte ansehen" in b.label for b in app.button) == len(STORIES)
    assert any("%" in str(m.value) for m in app.metric)  # key results from results/full


@pytest.mark.parametrize("story", ALL_STORIES, ids=lambda s: s.key)
def test_story_page_renders(story):
    app = page_app("story_page", f"import streamlit as st\nst.session_state.story = {story.key!r}").run()
    assert not app.exception, app.exception
    assert app.title[0].value.endswith(story.title)
    assert any(story.lesson[:40] in m.value for m in app.markdown)


def test_what_if_slider_changes_the_recommendation():
    app = page_app("story_page", "import streamlit as st\nst.session_state.story = 'telefonbuch'").run()
    slider = app.select_slider(key="whatif_telefonbuch")
    slider.set_value(1_000_000).run()
    assert not app.exception
    assert any("Hashtabelle" in m.value for m in app.markdown)
