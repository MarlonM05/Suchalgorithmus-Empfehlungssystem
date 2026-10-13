"""Central list of all search algorithms with their metadata.

The registry is the single place where the rest of the project (benchmark,
recommender, executor, user interface) looks up the algorithms, their
preconditions and their sources.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ._common import ACCESSES, COMPARISONS, new_counter
from .binary import (
    binary_search,
    binary_search_counted,
    branchless_binary_search,
    branchless_binary_search_counted,
)
from .exponential import exponential_search, exponential_search_counted
from .fibonacci import fibonacci_search, fibonacci_search_counted
from .hashing import build_hash_table, hash_lookup, hash_lookup_counted
from .interpolation import interpolation_search, interpolation_search_counted
from .jump import jump_search, jump_search_counted
from .linear import linear_search, linear_search_counted


@dataclass(frozen=True)
class SearchResult:
    """Result of one counted search."""

    index: int
    comparisons: int
    accesses: int


def as_search_array(values: Any) -> np.ndarray:
    """Convert input data to the contiguous int64 array all algorithms expect."""
    return np.ascontiguousarray(values, dtype=np.int64)


@dataclass(frozen=True)
class SearchAlgorithm:
    """A search algorithm together with everything needed to run and explain it.

    ``prepare`` turns an int64 array into the data structure the search
    function works on (the array itself, or a hash table). It does not sort:
    sorting is a separate preprocessing step decided by the recommender.
    """

    key: str
    name_de: str
    search: Callable  # compiled without counters, used for timing
    search_counted: Callable  # compiled with counters
    requires_sorted: bool
    complexity: str
    sources: tuple[str, ...]
    prepare: Callable[[np.ndarray], Any] = field(default=as_search_array)

    def find(self, data: Any, key: int) -> int:
        """Run the search on prepared ``data`` and return the index or -1."""
        return int(self.search(data, np.int64(key), new_counter()))

    def find_counted(self, data: Any, key: int) -> SearchResult:
        """Run the counting variant on prepared ``data``."""
        ops = new_counter()
        index = int(self.search_counted(data, np.int64(key), ops))
        return SearchResult(index, int(ops[COMPARISONS]), int(ops[ACCESSES]))


def _prepare_hash_table(values: np.ndarray):
    return build_hash_table(as_search_array(values))


ALGORITHMS: dict[str, SearchAlgorithm] = {
    algorithm.key: algorithm
    for algorithm in [
        SearchAlgorithm(
            key="linear",
            name_de="Lineare Suche",
            search=linear_search,
            search_counted=linear_search_counted,
            requires_sorted=False,
            complexity="O(n)",
            sources=("vansandt2019",),
        ),
        SearchAlgorithm(
            key="binary",
            name_de="Binäre Suche",
            search=binary_search,
            search_counted=binary_search_counted,
            requires_sorted=True,
            complexity="O(log n)",
            sources=("khuong2017",),
        ),
        SearchAlgorithm(
            key="binary_branchless",
            name_de="Binäre Suche (verzweigungsfrei)",
            search=branchless_binary_search,
            search_counted=branchless_binary_search_counted,
            requires_sorted=True,
            complexity="O(log n)",
            sources=("khuong2017",),
        ),
        SearchAlgorithm(
            key="jump",
            name_de="Sprungsuche",
            search=jump_search,
            search_counted=jump_search_counted,
            requires_sorted=True,
            complexity="O(√n)",
            sources=("shneiderman1978",),
        ),
        SearchAlgorithm(
            key="exponential",
            name_de="Exponentielle Suche",
            search=exponential_search,
            search_counted=exponential_search_counted,
            requires_sorted=True,
            complexity="O(log i)",
            sources=("bentley1976",),
        ),
        SearchAlgorithm(
            key="fibonacci",
            name_de="Fibonacci-Suche",
            search=fibonacci_search,
            search_counted=fibonacci_search_counted,
            requires_sorted=True,
            complexity="O(log n)",
            sources=("ferguson1960",),
        ),
        SearchAlgorithm(
            key="interpolation",
            name_de="Interpolationssuche (mit Guard)",
            search=interpolation_search,
            search_counted=interpolation_search_counted,
            requires_sorted=True,
            complexity="O(log log n) erwartet, O(n) im Worst Case",
            sources=("peterson1957", "perl1978", "gonnet1980", "vansandt2019"),
        ),
        SearchAlgorithm(
            key="hash",
            name_de="Hash-Lookup",
            search=hash_lookup,
            search_counted=hash_lookup_counted,
            requires_sorted=False,
            complexity="O(1) erwartet, Aufbau O(n)",
            sources=("richter2015",),
            prepare=_prepare_hash_table,
        ),
    ]
}
