"""Tests for the German number formatting and the CSV upload (improvements after M5)."""

import math

import numpy as np
import pytest

from recommender.data import DatasetConfig, build_linked_list, column_keys, generate_keys, numeric_columns, random_hit_key, random_miss_key, read_table
from recommender.executor import execute, expected_index
from recommender.viz.formatting import format_ns, german, percent
from recommender.viz.stepper import trace

# --- German number formatting ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (7, "7"),
        (1234567, "1.234.567"),
        (-1234567, "-1.234.567"),
        (100000.0, "100.000"),
        (0.018, "0,018"),
        (12.345, "12,3"),
        (1234.5, "1.234"),  # was "1,23e+03"
        (999.99, "1.000"),  # was "1e+03"
        (2**62 + 1, "4.611.686.018.427.387.905"),  # exact beyond 2^53
        (np.int64(42), "42"),
        (float("nan"), "–"),
        (None, "–"),
        (float("inf"), "∞"),
    ],
)
def test_german_numbers(value, expected):
    assert german(value) == expected


@pytest.mark.parametrize(
    ("ns", "expected"),
    [
        (512, "512 ns"),
        (1_500, "1,5 µs"),
        (176_000, "176 µs"),
        (999_990_000, "1.000 ms"),  # was "1e+03 ms"
        (2.5e9, "2,5 s"),
        (90e9, "1,5 min"),
        (5e13, "13,9 h"),  # was "5e+04 s"
        (math.nan, "–"),
    ],
)
def test_format_ns(ns, expected):
    assert format_ns(ns) == expected


def test_percent():
    assert percent(0.125) == "12,5 %"
    assert percent(1.0, 0) == "100 %"


# --- CSV upload --------------------------------------------------------------------------


def keys_of(text: str, column: int = 0) -> list[int]:
    frame = read_table(text.encode("utf-8"))
    return column_keys(frame[numeric_columns(frame)[column]]).keys.tolist()


def test_file_without_header_keeps_the_first_value():
    assert keys_of("5\n3\n8\n") == [5, 3, 8]  # pandas alone would drop the 5 as column name


def test_file_with_header():
    frame = read_table(b"id,wert\n1,30\n2,10\n")
    assert numeric_columns(frame) == ["id", "wert"]
    assert column_keys(frame["wert"]).keys.tolist() == [30, 10]


@pytest.mark.parametrize("text", ["a;b\n1;2\n3;4\n", "a\tb\n1\t2\n3\t4\n", "1 2\n3 4\n", "﻿a,b\n1,2\n3,4\n"])
def test_separators_and_bom(text):
    assert keys_of(text, column=1) == [2, 4]


def test_empty_cells_are_skipped_and_counted():
    frame = read_table(b"wert\n1\n\n3\n")
    uploaded = column_keys(frame["wert"])
    assert uploaded.keys.tolist() == [1, 3]
    assert uploaded.keys.dtype == np.int64 and uploaded.keys.flags["C_CONTIGUOUS"]


def test_integer_valued_floats_are_accepted():
    assert keys_of("x\n1.0\n-2.0\n") == [1, -2]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x\n1.5\n2\n", "Nachkommastellen"),
        ("x\ninf\n2\n", "unendlich"),
        ("x\n1e19\n2\n", "Wertebereich"),
        ("x\n18446744073709551615\n", "2⁶³"),
    ],
)
def test_invalid_values_are_rejected(text, message):
    frame = read_table(text.encode())
    with pytest.raises(ValueError, match=message):
        column_keys(frame[numeric_columns(frame)[0]])


def test_empty_file_is_rejected():
    with pytest.raises(ValueError, match="leer"):
        read_table(b"  \n")


def test_text_and_yes_no_columns_are_not_offered():
    frame = read_table(b"name,ok,wert\nanna,True,1\nben,False,2\n")
    assert numeric_columns(frame) == ["wert"]


# --- Random search keys ------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        np.arange(10, dtype=np.int64),  # dense: miss must lie outside the range
        np.array([7], dtype=np.int64),
        np.array([np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max], dtype=np.int64),
        generate_keys(DatasetConfig(n=50_000, order="unsorted", duplicate_share=0.5)),
    ],
)
def test_random_keys_hit_and_miss(keys):
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert random_hit_key(keys, rng) in keys
        assert random_miss_key(keys, rng) not in keys


def test_random_miss_key_checks_all_data():
    # The old implementation only looked at the first 2 million values: here
    # they are the even numbers, the odd ones follow later. It returned odd
    # numbers as "misses" although every number of the range is present.
    keys = np.concatenate([np.arange(0, 4_000_000, 2), np.arange(1, 4_000_000, 2)]).astype(np.int64)
    rng = np.random.default_rng(1)
    assert all(random_miss_key(keys, rng) not in keys for _ in range(10))


def test_random_keys_on_empty_data():
    rng = np.random.default_rng(0)
    empty = np.empty(0, dtype=np.int64)
    assert random_hit_key(empty, rng) == 0 and random_miss_key(empty, rng) == 0


# --- Reference result of the executor ------------------------------------------------------


def test_expected_index_array_and_list():
    keys = np.array([5, 3, 5, 9], dtype=np.int64)
    assert expected_index(keys, 5) == 0 and expected_index(keys, 9) == 3 and expected_index(keys, 4) == -1
    linked = build_linked_list(keys, np.random.default_rng(0))
    assert expected_index(linked, 9) == 3


def test_wrong_sortedness_guarantee_is_detectable():
    data = generate_keys(DatasetConfig(n=10_000, order="unsorted", seed=2))
    key = int(data[1234])
    result = execute("binary_branchless", data, key, sorted_known=True)  # check skipped on purpose
    assert result.index != expected_index(data, key)


# --- Learning mode ---------------------------------------------------------------------------


def test_jump_trace_marks_the_block_ends():
    steps = trace("jump", np.arange(0, 32, 2), 20)
    assert all(step.marks == (3, 7, 11, 15) for step in steps[:-1])
