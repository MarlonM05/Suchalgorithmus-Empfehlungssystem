"""Interpolation search with guard [peterson1957], [perl1978], [gonnet1980], [vansandt2019].

Instead of probing the middle, the probe position is estimated from the key
value, assuming the values grow roughly linearly with the index:

    pos = lo + (key - arr[lo]) / (arr[hi - 1] - arr[lo]) * (hi - 1 - lo)

Expected O(log log n) comparisons on uniformly distributed keys, but O(n) in
the worst case (strongly skewed data) [perl1978], [gonnet1980].

Guard [vansandt2019, Alg. 2, lines 14-17]: if the estimated position lies
within ``GUARD_SIZE`` elements of either end of the current interval, further
interpolations make too little progress, and the search switches to a
sequential scan starting at that end. [vansandt2019, App. A.3] found a guard
size of 8 to be the best all-round choice for their SIP algorithm, so we use 8.
Unlike SIP we do not reuse the slope or use fixed-point arithmetic; this is the
classic interpolation search plus the guard (see docs/annahmen.md).
Requires a sorted array with numeric keys.
"""

from numba import njit

from ._common import ACCESSES, COMPARISONS, make_check_candidate

GUARD_SIZE = 8


def _make_interpolation_search(count_ops: bool):
    check_candidate = make_check_candidate(count_ops)

    @njit
    def interpolation_search(arr, key, ops):
        """Interpolate the probe position; switch to sequential scan near the ends."""
        lo = 0
        hi = arr.shape[0]  # invariant: the lower bound lies in [lo, hi]
        while lo < hi:
            if count_ops:
                ops[ACCESSES] += 2
                ops[COMPARISONS] += 2
            first = arr[lo]
            last = arr[hi - 1]
            if key <= first:
                return check_candidate(arr, key, lo, ops)
            if key > last:
                return check_candidate(arr, key, hi, ops)

            # Now first < key <= last, hence last > first: no division by zero.
            # Floats avoid int64 overflow for extreme values.
            fraction = (float(key) - float(first)) / (float(last) - float(first))
            pos = lo + int(fraction * (hi - 1 - lo))
            pos = min(max(pos, lo), hi - 1)

            if pos + GUARD_SIZE >= hi - 1:
                # Guard: close to the right end -> scan backwards from hi - 1.
                # arr[hi - 1] >= key and arr[lo] < key are already known.
                i = hi - 1
                while i - 1 > lo:
                    if count_ops:
                        ops[ACCESSES] += 1
                        ops[COMPARISONS] += 1
                    if arr[i - 1] < key:
                        break
                    i -= 1
                return check_candidate(arr, key, i, ops)
            if pos - GUARD_SIZE <= lo:
                # Guard: close to the left end -> scan forwards from lo + 1.
                i = lo + 1
                while i < hi:
                    if count_ops:
                        ops[ACCESSES] += 1
                        ops[COMPARISONS] += 1
                    if arr[i] >= key:
                        break
                    i += 1
                return check_candidate(arr, key, i, ops)

            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[pos] < key:
                lo = pos + 1
            else:
                hi = pos
        return check_candidate(arr, key, lo, ops)

    return interpolation_search


interpolation_search = _make_interpolation_search(count_ops=False)
interpolation_search_counted = _make_interpolation_search(count_ops=True)
