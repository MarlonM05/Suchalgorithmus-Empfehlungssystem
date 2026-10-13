"""Fibonacci search [ferguson1960].

Like binary search, but the interval is split according to consecutive
Fibonacci numbers instead of halves. Probe positions are computed with
addition and subtraction only, which mattered on early hardware without fast
division; today it is mainly of historical interest. O(log n) comparisons.
Requires a sorted array.

Formulation used here (lower-bound variant): the candidate positions form an
interval of ``F(k) - 1`` elements starting at ``lo``. Probing at
``lo + F(k-1) - 1`` leaves either ``F(k-1) - 1`` elements on the left or
``F(k-2) - 1`` elements on the right. The array is treated as if padded with
``+infinity`` up to ``F(k) - 1`` elements; these virtual positions are never
read.
"""

from numba import njit

from ._common import ACCESSES, COMPARISONS, make_check_candidate


def _make_fibonacci_search(count_ops: bool):
    check_candidate = make_check_candidate(count_ops)

    @njit
    def fibonacci_search(arr, key, ops):
        """Narrow the interval using Fibonacci numbers; return first match or -1."""
        n = arr.shape[0]

        # Smallest Fibonacci number F(k) with F(k) - 1 >= n.
        # fib_k = F(k), fib_k1 = F(k-1).
        fib_k = 1
        fib_k1 = 1
        while fib_k - 1 < n:
            fib_k, fib_k1 = fib_k + fib_k1, fib_k

        lo = 0
        while fib_k > 1:  # interval [lo, lo + F(k) - 1) still has elements
            probe = lo + fib_k1 - 1
            probe_is_smaller = False
            if probe < n:  # positions >= n are virtual +infinity
                if count_ops:
                    ops[ACCESSES] += 1
                    ops[COMPARISONS] += 1
                probe_is_smaller = arr[probe] < key
            if probe_is_smaller:
                # Continue right of the probe: k -> k - 2.
                lo = probe + 1
                fib_k, fib_k1 = fib_k - fib_k1, 2 * fib_k1 - fib_k
            else:
                # Continue left of the probe (the probe stays a candidate): k -> k - 1.
                fib_k, fib_k1 = fib_k1, fib_k - fib_k1
        return check_candidate(arr, key, lo, ops)

    return fibonacci_search


fibonacci_search = _make_fibonacci_search(count_ops=False)
fibonacci_search_counted = _make_fibonacci_search(count_ops=True)
