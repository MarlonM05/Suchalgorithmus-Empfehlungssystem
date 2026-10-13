"""Conventions shared by all search algorithms.

Signature
---------
Every search function has the signature ``search(data, key, ops) -> int`` and
returns the index of the **first** occurrence of ``key`` or ``-1``. For the
array-based algorithms ``data`` is a one-dimensional ``int64`` array, for the
hash lookup it is the prepared hash table.

Operation counting
------------------
``ops`` is a small ``int64`` array (see :func:`new_counter`). Two kinds of
operations are counted as a hardware-independent cost measure:

* ``ops[COMPARISONS]``: comparisons between the search key and a stored key
* ``ops[ACCESSES]``: reads of a stored element (array cell or hash slot)

Comparisons of loop indices are not counted, because they are the same for all
algorithms and say nothing about the data access cost.

Two compiled variants per algorithm
-----------------------------------
Each algorithm module contains a factory ``_make_<name>(count_ops)`` that
returns a Numba-compiled function. All counting code sits inside
``if count_ops:`` blocks. Numba treats closure variables as compile-time
constants, so in the variant built with ``count_ops=False`` these blocks are
removed by the compiler. Runtime measurements therefore execute the same source
code as the counted variant, just without the counters. A single source of
truth avoids two implementations drifting apart.

Note: the compiled functions are closures and are not cached on disk
(``cache=True`` cannot distinguish closures with different constants), so the
first call of each variant includes the compilation time. The benchmark
harness discards this warm-up call.
"""

import numpy as np
from numba import njit

COMPARISONS = 0
ACCESSES = 1


def new_counter() -> np.ndarray:
    """Return a fresh operation counter ``[comparisons, accesses]``."""
    return np.zeros(2, dtype=np.int64)


def make_check_candidate(count_ops: bool):
    """Build the final check shared by all algorithms on sorted arrays.

    Those algorithms first determine the *lower bound*, i.e. the smallest index
    ``i`` with ``arr[i] >= key``. Because the array is sorted, this is the first
    occurrence of ``key`` if the key is present at all. The check below
    confirms that ``arr[i]`` actually equals ``key``.
    """

    @njit
    def check_candidate(arr, key, i, ops):
        if i < arr.shape[0]:
            if count_ops:
                ops[ACCESSES] += 1
                ops[COMPARISONS] += 1
            if arr[i] == key:
                return i
        return -1

    return check_candidate
