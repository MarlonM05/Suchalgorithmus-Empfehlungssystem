"""Exponential search (unbounded search) [bentley1976].

Doubles a bound (1, 2, 4, ...) until it passes the key, then runs a binary
search in the last interval. Costs O(log i) comparisons, where i is the
position of the key. Pays off when the key is near the front or the size of
the data is unknown. Requires a sorted array.
"""

from numba import njit

from ._common import ACCESSES, COMPARISONS, make_check_candidate
from .binary import make_lower_bound


def _make_exponential_search(count_ops: bool):
    lower_bound = make_lower_bound(count_ops)
    check_candidate = make_check_candidate(count_ops)

    @njit
    def exponential_search(arr, key, ops):
        """Double the bound while ``arr[bound - 1] < key``, then binary search."""
        n = arr.shape[0]
        bound = 1
        while bound <= n:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[bound - 1] >= key:
                break
            bound *= 2
        # The previous check (at bound // 2) showed that every element before
        # index bound // 2 is smaller than the key, so the lower bound lies in
        # [bound // 2, min(bound - 1, n)].
        i = lower_bound(arr, key, bound // 2, min(bound - 1, n), ops)
        return check_candidate(arr, key, i, ops)

    return exponential_search


exponential_search = _make_exponential_search(count_ops=False)
exponential_search_counted = _make_exponential_search(count_ops=True)
