"""Binary search in two variants: classic and branchless.

Both variants search for the lower bound (first index with ``arr[i] >= key``),
which yields the first occurrence of a key in the presence of duplicates.
Complexity: O(log n) comparisons. Requires a sorted array with random access.

The branchless variant follows [khuong2017]: the loop body contains no
data-dependent branch, so the CPU cannot mispredict it. Having both variants
lets us check the claim of [khuong2017] in our own (Numba) environment.
"""

from numba import njit

from ._common import ACCESSES, COMPARISONS, make_check_candidate


def make_lower_bound(count_ops: bool):
    """Build the classic lower-bound loop on the index range ``[lo, hi)``.

    Returns the smallest index ``i`` in ``[lo, hi)`` with ``arr[i] >= key``,
    or ``hi`` if there is none. Also used by exponential search.
    """

    @njit
    def lower_bound(arr, key, lo, hi, ops):
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return lower_bound


def _make_binary_search(count_ops: bool):
    lower_bound = make_lower_bound(count_ops)
    check_candidate = make_check_candidate(count_ops)

    @njit
    def binary_search(arr, key, ops):
        """Classic binary search: halve the interval with an if/else branch."""
        i = lower_bound(arr, key, 0, arr.shape[0], ops)
        return check_candidate(arr, key, i, ops)

    return binary_search


def _make_branchless_binary_search(count_ops: bool):
    check_candidate = make_check_candidate(count_ops)

    @njit
    def branchless_binary_search(arr, key, ops):
        """Branchless binary search [khuong2017].

        ``base`` is the start of the remaining interval and ``length`` its size.
        In every step the interval is halved; whether ``base`` moves is computed
        arithmetically (``half * (arr[...] < key)``) instead of by a branch.
        """
        n = arr.shape[0]
        if n == 0:
            return -1
        base = 0
        length = n
        while length > 1:
            half = length // 2
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            base += half * (arr[base + half] < key)
            length -= half
        if count_ops:
            ops[ACCESSES] += 1
            ops[COMPARISONS] += 1
        base += arr[base] < key
        return check_candidate(arr, key, base, ops)

    return branchless_binary_search


binary_search = _make_binary_search(count_ops=False)
binary_search_counted = _make_binary_search(count_ops=True)
branchless_binary_search = _make_branchless_binary_search(count_ops=False)
branchless_binary_search_counted = _make_branchless_binary_search(count_ops=True)
