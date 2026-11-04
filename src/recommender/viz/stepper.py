"""Step-by-step traces of the search algorithms for the learning mode.

Algorithm visualisations mainly help when the learner actively takes part
[hundhausen2002]. The app therefore shows one step at a time and the user
clicks on. Each function below re-implements one algorithm in plain Python,
following exactly the logic of the compiled version in
``recommender.algorithms`` (lower bound, then a final equality check), and
records every comparison as a :class:`Step` with a German explanation. The
tests check that every trace ends with the same result as the compiled
algorithm.

Meant for small arrays (n <= 64); hash lookup is not traced because its
interesting part, the hash table, is not an ordered array.
"""

import math
from dataclasses import dataclass

from ..algorithms.interpolation import GUARD_SIZE

MAX_LEARNING_SIZE = 64


@dataclass(frozen=True)
class Step:
    """State of the search after one comparison.

    Attributes:
        lo, hi: remaining candidate range ``[lo, hi)``.
        probe: index compared in this step (``None`` if none).
        text: what happened, in German.
        result: final index or -1 once the search is finished, else ``None``.
        marks: further indices to highlight (e.g. the end of a jump block).
    """

    lo: int
    hi: int
    probe: int | None
    text: str
    result: int | None = None
    marks: tuple[int, ...] = ()

    @property
    def done(self) -> bool:
        return self.result is not None


def _check(arr, key, i: int, lo: int, hi: int) -> Step:
    """Final step: is the lower bound ``i`` really the key?"""
    if i < len(arr) and arr[i] == key:
        return Step(i, i + 1, i, f"a[{i}] = {arr[i]} ist der gesuchte Schlüssel {key}: gefunden an Index {i}.", result=i)
    if i < len(arr):
        return Step(lo, hi, i, f"a[{i}] = {arr[i]} ≠ {key}: der Schlüssel kommt nicht vor.", result=-1)
    return Step(lo, hi, None, f"Alle Elemente sind kleiner als {key}: der Schlüssel kommt nicht vor.", result=-1)


def linear_steps(arr, key) -> list[Step]:
    steps = []
    n = len(arr)
    for i in range(n):
        if arr[i] == key:
            steps.append(Step(i, n, i, f"a[{i}] = {arr[i]} = {key}: gefunden an Index {i}.", result=i))
            return steps
        steps.append(Step(i + 1, n, i, f"a[{i}] = {arr[i]} ≠ {key}: weiter mit dem nächsten Element."))
    steps.append(Step(n, n, None, f"Alle {n} Elemente geprüft: der Schlüssel {key} kommt nicht vor.", result=-1))
    return steps


def _lower_bound_steps(arr, key, lo: int, hi: int, steps: list[Step]) -> int:
    """Classic binary search for the first index with a[i] >= key in [lo, hi)."""
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < key:
            steps.append(Step(mid + 1, hi, mid, f"Mitte a[{mid}] = {arr[mid]} < {key}: rechte Hälfte weiter durchsuchen."))
            lo = mid + 1
        else:
            steps.append(Step(lo, mid, mid, f"Mitte a[{mid}] = {arr[mid]} ≥ {key}: linke Hälfte weiter, a[{mid}] bleibt Kandidat."))
            hi = mid
    return lo


def binary_steps(arr, key) -> list[Step]:
    steps: list[Step] = []
    i = _lower_bound_steps(arr, key, 0, len(arr), steps)
    steps.append(_check(arr, key, i, i, i))
    return steps


def jump_steps(arr, key) -> list[Step]:
    n = len(arr)
    step = max(1, math.isqrt(n))
    block_ends = tuple(range(step - 1, n, step))  # highlighted in the learning mode
    steps = []
    block_start, block_end = 0, min(step, n)
    while block_end < n:
        if arr[block_end - 1] >= key:
            steps.append(Step(block_start, block_end, block_end - 1, f"Blockende a[{block_end - 1}] = {arr[block_end - 1]} ≥ {key}: der Schlüssel liegt in diesem Block ({block_start}–{block_end - 1}).", marks=block_ends))
            break
        steps.append(Step(block_end, n, block_end - 1, f"Blockende a[{block_end - 1}] = {arr[block_end - 1]} < {key}: Sprung um {step} Elemente.", marks=block_ends))
        block_start, block_end = block_end, min(block_end + step, n)
    i = block_start
    while i < block_end:
        if arr[i] >= key:
            steps.append(Step(i, block_end, i, f"Im Block: a[{i}] = {arr[i]} ≥ {key}: erster Kandidat gefunden.", marks=block_ends))
            break
        steps.append(Step(i + 1, block_end, i, f"Im Block: a[{i}] = {arr[i]} < {key}: weiter.", marks=block_ends))
        i += 1
    steps.append(_check(arr, key, i, i, block_end))
    return steps


def exponential_steps(arr, key) -> list[Step]:
    n = len(arr)
    steps = []
    bound = 1
    while bound <= n:
        if arr[bound - 1] >= key:
            steps.append(Step(bound // 2, min(bound - 1, n), bound - 1, f"a[{bound - 1}] = {arr[bound - 1]} ≥ {key}: Grenze gefunden, binäre Suche in [{bound // 2}, {bound - 1})."))
            break
        steps.append(Step(bound, n, bound - 1, f"a[{bound - 1}] = {arr[bound - 1]} < {key}: Grenze verdoppeln auf {2 * bound}."))
        bound *= 2
    i = _lower_bound_steps(arr, key, bound // 2, min(bound - 1, n), steps)
    steps.append(_check(arr, key, i, i, i))
    return steps


def fibonacci_steps(arr, key) -> list[Step]:
    n = len(arr)
    fib_k, fib_k1 = 1, 1
    while fib_k - 1 < n:
        fib_k, fib_k1 = fib_k + fib_k1, fib_k
    steps = []
    lo = 0
    while fib_k > 1:
        probe = lo + fib_k1 - 1
        hi = min(lo + fib_k - 1, n)
        if probe < n and arr[probe] < key:
            steps.append(Step(probe + 1, hi, probe, f"a[{probe}] = {arr[probe]} < {key}: rechts weiter (Intervall schrumpft auf F(k−2) − 1 = {fib_k - fib_k1 - 1} Plätze)."))
            lo = probe + 1
            fib_k, fib_k1 = fib_k - fib_k1, 2 * fib_k1 - fib_k
        else:
            shown = f"a[{probe}] = {arr[probe]} ≥ {key}" if probe < n else f"Position {probe} liegt hinter dem Array (gilt als +∞)"
            steps.append(Step(lo, min(probe, n), probe if probe < n else None, f"{shown}: links weiter (Intervall schrumpft auf F(k−1) − 1 = {fib_k1 - 1} Plätze)."))
            fib_k, fib_k1 = fib_k1, fib_k - fib_k1
    steps.append(_check(arr, key, lo, lo, lo))
    return steps


def interpolation_steps(arr, key) -> list[Step]:
    lo, hi = 0, len(arr)
    steps = []
    while lo < hi:
        first, last = arr[lo], arr[hi - 1]
        if key <= first:
            steps.append(Step(lo, hi, lo, f"{key} ≤ a[{lo}] = {first}: der Kandidat ist der Anfang des Bereichs."))
            steps.append(_check(arr, key, lo, lo, hi))
            return steps
        if key > last:
            steps.append(Step(hi, hi, hi - 1, f"{key} > a[{hi - 1}] = {last}: alle Elemente im Bereich sind kleiner."))
            steps.append(_check(arr, key, hi, hi, hi))
            return steps
        fraction = (float(key) - float(first)) / (float(last) - float(first))
        pos = min(max(lo + int(fraction * (hi - 1 - lo)), lo), hi - 1)
        estimate = f"Schätzung aus dem Wert: Position {pos}"
        if pos + GUARD_SIZE >= hi - 1:
            steps.append(Step(lo, hi, pos, f"{estimate}, weniger als {GUARD_SIZE} vor dem Ende: Guard schaltet auf sequenzielle Suche von hinten."))
            i = hi - 1
            while i - 1 > lo:
                if arr[i - 1] < key:
                    steps.append(Step(i, hi, i - 1, f"a[{i - 1}] = {arr[i - 1]} < {key}: Anfang gefunden."))
                    break
                steps.append(Step(lo, i, i - 1, f"a[{i - 1}] = {arr[i - 1]} ≥ {key}: weiter nach links."))
                i -= 1
            steps.append(_check(arr, key, i, i, hi))
            return steps
        if pos - GUARD_SIZE <= lo:
            steps.append(Step(lo, hi, pos, f"{estimate}, weniger als {GUARD_SIZE} nach dem Anfang: Guard schaltet auf sequenzielle Suche von vorn."))
            i = lo + 1
            while i < hi:
                if arr[i] >= key:
                    steps.append(Step(i, hi, i, f"a[{i}] = {arr[i]} ≥ {key}: erster Kandidat gefunden."))
                    break
                steps.append(Step(i + 1, hi, i, f"a[{i}] = {arr[i]} < {key}: weiter nach rechts."))
                i += 1
            steps.append(_check(arr, key, i, i, hi))
            return steps
        if arr[pos] < key:
            steps.append(Step(pos + 1, hi, pos, f"{estimate}: a[{pos}] = {arr[pos]} < {key}, rechts weiter."))
            lo = pos + 1
        else:
            steps.append(Step(lo, pos, pos, f"{estimate}: a[{pos}] = {arr[pos]} ≥ {key}, links weiter."))
            hi = pos
    steps.append(_check(arr, key, lo, lo, lo))
    return steps


# key -> (German name, trace function, needs sorted data)
TRACERS = {
    "linear": ("Lineare Suche", linear_steps, False),
    "binary": ("Binäre Suche", binary_steps, True),
    "jump": ("Sprungsuche", jump_steps, True),
    "exponential": ("Exponentielle Suche", exponential_steps, True),
    "fibonacci": ("Fibonacci-Suche", fibonacci_steps, True),
    "interpolation": ("Interpolationssuche (mit Guard)", interpolation_steps, True),
}


def trace(algorithm: str, arr, key) -> list[Step]:
    """All steps of ``algorithm`` searching ``key`` in ``arr``."""
    values = [int(v) for v in arr]
    return TRACERS[algorithm][1](values, int(key))
