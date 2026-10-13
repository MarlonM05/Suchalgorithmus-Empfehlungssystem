"""Hash lookup with an own open-addressing hash table [richter2015].

Design (chosen following the analysis in [richter2015], which found
multiplicative hashing with linear probing to be a robust, fast combination
for integer keys):

* Hash function: multiplicative (Fibonacci) hashing,
  ``h(k) = (k * A mod 2^64) >> (64 - b)`` with ``A = floor(2^64 / phi)``.
* Collision resolution: open addressing with linear probing.
* Capacity: smallest power of two ``2^b >= 2 n`` (load factor <= 0.5).

The table stores each distinct key once together with the index of its
**first** occurrence in the original array, so duplicates behave like in the
other algorithms. Building costs O(n) expected time, a lookup O(1) expected.
Range queries are not supported. The table works on unsorted data.

The table is a tuple ``(slot_keys, slot_index, shift)``: two arrays of equal
length (``slot_index[s] == -1`` marks an empty slot) and the precomputed shift
``64 - b`` of the hash function, so a lookup does not have to recompute it.
"""

import numpy as np
from numba import njit, uint64

from ._common import ACCESSES, COMPARISONS

FIBONACCI_MULTIPLIER = uint64(11400714819323198485)  # floor(2^64 / golden ratio)
EMPTY = -1


@njit
def _hash_slot(key, shift):
    """Multiplicative hashing: the top bits of key * A (mod 2^64) form the slot."""
    return np.int64((uint64(key) * FIBONACCI_MULTIPLIER) >> uint64(shift))


@njit
def build_hash_table(arr):
    """Build the hash table for ``arr``; returns ``(slot_keys, slot_index, shift)``."""
    n = arr.shape[0]
    bits = 1
    while (1 << bits) < 2 * n:
        bits += 1
    capacity = 1 << bits
    mask = capacity - 1
    shift = 64 - bits

    slot_keys = np.zeros(capacity, dtype=np.int64)
    slot_index = np.full(capacity, EMPTY, dtype=np.int64)
    for i in range(n):
        key = arr[i]
        slot = _hash_slot(key, shift)
        while slot_index[slot] != EMPTY and slot_keys[slot] != key:
            slot = (slot + 1) & mask
        if slot_index[slot] == EMPTY:  # keep only the first occurrence
            slot_keys[slot] = key
            slot_index[slot] = i
    return slot_keys, slot_index, shift


def _make_hash_lookup(count_ops: bool):
    @njit
    def hash_lookup(table, key, ops):
        """Probe slots from h(key) on until the key or an empty slot is found."""
        slot_keys, slot_index, shift = table
        mask = slot_keys.shape[0] - 1
        slot = _hash_slot(key, shift)
        while True:
            if count_ops:
                ops[ACCESSES] += 1
            if slot_index[slot] == EMPTY:
                return -1
            if count_ops:
                ops[COMPARISONS] += 1
            if slot_keys[slot] == key:
                return slot_index[slot]
            slot = (slot + 1) & mask

    return hash_lookup


hash_lookup = _make_hash_lookup(count_ops=False)
hash_lookup_counted = _make_hash_lookup(count_ops=True)
