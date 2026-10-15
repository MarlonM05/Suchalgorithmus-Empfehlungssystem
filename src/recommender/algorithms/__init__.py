"""Search algorithms, each compiled with Numba in a timing and a counting variant."""

from ._common import ACCESSES, COMPARISONS, new_counter
from .registry import ALGORITHMS, LIST_ALGORITHMS, SearchAlgorithm, SearchResult, as_search_array

__all__ = [
    "ACCESSES",
    "ALGORITHMS",
    "LIST_ALGORITHMS",
    "COMPARISONS",
    "SearchAlgorithm",
    "SearchResult",
    "as_search_array",
    "new_counter",
]
