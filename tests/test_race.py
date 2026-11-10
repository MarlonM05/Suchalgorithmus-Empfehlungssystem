"""Tests for the animated race (recommender.viz.race) and its page helpers."""

import json
import re

import numpy as np
import pytest

from recommender.algorithms import ALGORITHMS, as_search_array
from recommender.viz.race import DEFAULT_RACE, LANE_COLOURS, build_lanes, race_height, race_html
from recommender.viz.stepper import TRACERS


@pytest.fixture
def sorted_values() -> np.ndarray:
    return as_search_array(np.sort(np.random.default_rng(4).integers(0, 1000, 24)))


def test_lanes_follow_the_traces_and_the_compiled_counts(sorted_values):
    key = int(sorted_values[7])
    lanes = build_lanes(sorted_values, key, list(TRACERS))
    assert [lane.key for lane in lanes] == list(TRACERS)
    for lane in lanes:
        assert lane.steps[-1].done
        assert lane.steps[-1].result == lane.result == ALGORITHMS[lane.key].find(sorted_values, key)
        counted = ALGORITHMS[lane.key].find_counted(sorted_values, key)
        assert (lane.comparisons, lane.accesses) == (counted.comparisons, counted.accesses)
        assert lane.name == TRACERS[lane.key][0]


def test_binary_search_needs_fewer_steps_than_linear_search(sorted_values):
    key = int(sorted_values[-2])  # near the end: linear search must walk almost everything
    linear, binary = build_lanes(sorted_values, key, DEFAULT_RACE)
    assert linear.step_count == 23
    assert binary.step_count < linear.step_count
    assert binary.step_count <= 6  # ceil(log2(24)) probes plus the final check


def test_miss_is_reported_by_every_lane(sorted_values):
    lanes = build_lanes(sorted_values, int(sorted_values.max()) + 1, list(TRACERS))
    assert all(lane.result == -1 for lane in lanes)


def test_unknown_algorithm_is_rejected(sorted_values):
    with pytest.raises(ValueError, match="hash"):
        build_lanes(sorted_values, 5, ["hash"])


def test_race_html_is_self_contained_and_carries_the_traces(sorted_values):
    key = int(sorted_values[3])
    lanes = build_lanes(sorted_values, key, DEFAULT_RACE)
    document = race_html(sorted_values, key, lanes, autoplay=True)
    assert document.startswith("<!doctype html>")
    assert "http://" not in document and "https://" not in document  # no external resources
    payload = json.loads(re.search(r"const data = (\{.*?\});\n", document, re.S).group(1).replace("<\\/", "</"))
    assert payload["key"] == key and payload["values"] == [int(v) for v in sorted_values]
    assert payload["autoplay"] is True
    assert [len(lane["steps"]) for lane in payload["lanes"]] == [lane.step_count for lane in lanes]
    assert payload["lanes"][0]["colour"] == LANE_COLOURS["linear"]
    assert all(step["result"] is None for step in payload["lanes"][1]["steps"][:-1])
    assert payload["lanes"][1]["steps"][-1]["result"] == lanes[1].result


def test_race_html_escapes_script_terminators():
    """Step texts go through the JSON payload; a '</script>' in them must not end the script."""
    values = as_search_array([1, 2, 3])
    lanes = build_lanes(values, 2, ["linear"])
    document = race_html(values, 2, lanes)
    script = document[document.index("<script>") + len("<script>") : document.rindex("</script>")]
    assert "</script" not in script.lower()
    payload_text = re.search(r"const data = (\{.*?\});\n", document, re.S).group(1)
    assert "</" not in payload_text  # every "</" of the JSON payload is escaped as "<\/"


def test_race_height_grows_with_lanes_and_rows(sorted_values):
    two = build_lanes(sorted_values, 5, DEFAULT_RACE)
    six = build_lanes(sorted_values, 5, list(TRACERS))
    assert race_height(six, 24) > race_height(two, 24)
    assert race_height(two, 64) > race_height(two, 16)
