"""Linear (sequential) search.

The only method that works on unsorted data without preprocessing, and often
the fastest one on very small arrays [vansandt2019].
Complexity: O(n) comparisons in the worst and average case.
"""

from numba import njit

from ._common import ACCESSES, COMPARISONS


def _make_linear_search(count_ops: bool):
    @njit
    def linear_search(arr, key, ops):
        """Scan the array from left to right; return the first match or -1."""
        for i in range(arr.shape[0]):
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[i] == key:
                return i
        return -1

    return linear_search


linear_search = _make_linear_search(count_ops=False)
linear_search_counted = _make_linear_search(count_ops=True)
