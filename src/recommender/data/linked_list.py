"""Singly linked list stored in a node pool.

Numba cannot efficiently handle Python object graphs, so the list is stored as
two arrays (a *node pool*): ``values[s]`` is the key of the node in slot ``s``
and ``next_index[s]`` the slot of its successor (``-1`` at the end).

The logical order of the list is independent of the slot order: node ``j`` of
the list is placed in a random slot. Following the list therefore jumps
through memory like pointer chasing in a heap-allocated list, and the CPU
cannot prefetch the next node. Only sequential access is possible: reaching
position ``j`` requires following ``j`` pointers.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

END = -1


@dataclass(frozen=True)
class LinkedList:
    """A singly linked list in a node pool.

    Attributes:
        values: key of the node in each slot.
        next_index: slot of the successor node, or -1 for the last node.
        head: slot of the first node, or -1 for an empty list.
    """

    values: np.ndarray
    next_index: np.ndarray
    head: int

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Representation passed to the Numba search functions."""
        return self.values, self.next_index, self.head


def build_linked_list(keys: np.ndarray, rng: np.random.Generator) -> LinkedList:
    """Store ``keys`` (in this logical order) as a linked list in random slots."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    n = keys.shape[0]
    slot_of = rng.permutation(n)  # slot_of[j] = slot of the j-th node
    values = np.empty(n, dtype=np.int64)
    next_index = np.full(n, END, dtype=np.int64)
    values[slot_of] = keys
    next_index[slot_of[:-1]] = slot_of[1:]
    head = int(slot_of[0]) if n > 0 else END
    return LinkedList(values, next_index, head)


@njit
def _collect_in_list_order(values, next_index, head):
    out = np.empty(values.shape[0], dtype=np.int64)
    slot = head
    for j in range(out.shape[0]):
        out[j] = values[slot]
        slot = next_index[slot]
    return out


def linked_list_to_array(linked_list: LinkedList) -> np.ndarray:
    """Return the keys in logical list order (follows the pointers)."""
    return _collect_in_list_order(*linked_list.as_tuple())
