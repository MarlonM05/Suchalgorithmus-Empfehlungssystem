"""Jump search [shneiderman1978].

Jumps forward in blocks of size m and then scans the block that must contain
the key. The optimal block size is m = sqrt(n), giving O(sqrt(n)) comparisons.
It only ever moves forward, so it suits structures where going back is
expensive or only sequential access is possible (e.g. linked lists).
Requires a sorted array.
"""

import numpy as np
from numba import njit

from ._common import ACCESSES, COMPARISONS, make_check_candidate


def _make_jump_search(count_ops: bool):
    check_candidate = make_check_candidate(count_ops)

    @njit
    def jump_search(arr, key, ops):
        """Find the first block whose last element is >= key, then scan it."""
        n = arr.shape[0]
        step = max(1, int(np.sqrt(n)))

        # Phase 1: jump over whole blocks [block_start, block_end) while the
        # last element of the block is still smaller than the key.
        block_start = 0
        block_end = min(step, n)
        while block_end < n:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[block_end - 1] >= key:
                break
            block_start = block_end
            block_end = min(block_end + step, n)

        # Phase 2: sequential scan inside the block for the lower bound.
        i = block_start
        while i < block_end:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[i] >= key:
                break
            i += 1
        return check_candidate(arr, key, i, ops)

    return jump_search


jump_search = _make_jump_search(count_ops=False)
jump_search_counted = _make_jump_search(count_ops=True)
