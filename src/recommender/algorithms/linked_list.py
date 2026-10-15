"""Search algorithms for a singly linked list (sequential access only).

The list is passed as the tuple ``(values, next_index, head)`` of its node pool
(see ``recommender.data.linked_list``). The result is the **logical position**
of the first node containing the key (0 = head), or -1.

Counting on a list: every visited node counts as one access, whether its key
is compared or only its pointer is followed. Binary, exponential, Fibonacci and
interpolation search need random access and are therefore not offered for
lists: on a list, reaching the middle alone already costs n/2 pointer steps.
"""

import numpy as np
from numba import njit

from ._common import ACCESSES, COMPARISONS

END = -1


def _make_list_linear_search(count_ops: bool):
    @njit
    def list_linear_search(data, key, ops):
        """Follow the list from the head and compare every node."""
        values, next_index, head = data
        slot = head
        position = 0
        while slot != END:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if values[slot] == key:
                return position
            slot = next_index[slot]
            position += 1
        return -1

    return list_linear_search


def _make_list_jump_search(count_ops: bool):
    @njit
    def list_jump_search(data, key, ops):
        """Jump search on a sorted list [shneiderman1978].

        Only the last node of each block of size m = sqrt(n) is compared. The
        start of the current block is remembered, so going back to it costs
        nothing; this is why jump search fits structures without random access.
        The list is traversed at most once plus one block: O(n) node visits,
        but only O(sqrt(n)) key comparisons.
        """
        values, next_index, head = data
        n = values.shape[0]
        if n == 0:
            return -1
        step = max(1, int(np.sqrt(n)))

        # Phase 1: walk block by block; compare only the last node of a block.
        block_start_slot = head
        block_start = 0
        while True:
            block_end = min(block_start + step, n) - 1  # position of the last node
            if block_end == n - 1:
                break  # last block: no comparison needed, scan it
            slot = block_start_slot
            for _ in range(block_end - block_start):
                if count_ops:
                    ops[ACCESSES] += 1
                slot = next_index[slot]
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if values[slot] >= key:
                break
            block_start_slot = next_index[slot]
            block_start = block_end + 1

        # Phase 2: scan the block from its remembered start for the lower bound.
        slot = block_start_slot
        position = block_start
        while position <= block_end:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if values[slot] >= key:
                if count_ops:
                    ops[COMPARISONS] += 1
                if values[slot] == key:
                    return position
                return -1
            slot = next_index[slot]
            position += 1
        return -1

    return list_jump_search


list_linear_search = _make_list_linear_search(count_ops=False)
list_linear_search_counted = _make_list_linear_search(count_ops=True)
list_jump_search = _make_list_jump_search(count_ops=False)
list_jump_search_counted = _make_list_jump_search(count_ops=True)
