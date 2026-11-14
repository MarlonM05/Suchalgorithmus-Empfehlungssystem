"""Strategies: what the recommender can choose (the algorithm space [rice1976]).

A strategy is a search algorithm together with the preprocessing it needs.
This matters because on unsorted data a fast search algorithm is only
available after sorting or after building a hash table, and these costs must
be paid before the first query.

Strategies that require sorted data must make sure the data really is sorted.
If the user does not guarantee it, the strategy first runs the full O(n)
check ``verify_sorted`` (see docs/annahmen.md A16); its cost is part of the
strategy's cost.
"""

from dataclasses import dataclass

from ..data.queries import POSITIONS


@dataclass(frozen=True)
class Strategy:
    """One option of the recommender.

    Attributes:
        key: identifier.
        name_de: German name shown in the user interface.
        structure: ``"array"`` or ``"linked_list"``.
        search: key of the search algorithm in the algorithm registry.
        preprocessing: ``None``, ``"sort"``, ``"hash_build"`` (array) or
            ``"list_hash_build"`` (copy the list into a hash table).
        requires_sorted: the data must already be sorted (checked first
            unless the user guarantees it).
    """

    key: str
    name_de: str
    structure: str
    search: str
    preprocessing: str | None = None
    requires_sorted: bool = False


_ARRAY_SORTED_SEARCHES = {
    "binary": "Binäre Suche",
    "binary_branchless": "Binäre Suche (verzweigungsfrei)",
    "jump": "Sprungsuche",
    "exponential": "Exponentielle Suche",
    "fibonacci": "Fibonacci-Suche",
    "interpolation": "Interpolationssuche",
}

STRATEGIES: dict[str, Strategy] = {
    s.key: s
    for s in [
        Strategy("linear", "Lineare Suche", "array", "linear"),
        *[Strategy(key, name, "array", key, requires_sorted=True) for key, name in _ARRAY_SORTED_SEARCHES.items()],
        Strategy("hash", "Hashtabelle aufbauen + Hash-Lookup", "array", "hash", preprocessing="hash_build"),
        Strategy("sort_binary", "Sortieren + binäre Suche (verzweigungsfrei)", "array", "binary_branchless", preprocessing="sort"),
        Strategy("list_linear", "Lineare Suche (Liste)", "linked_list", "list_linear"),
        Strategy("list_jump", "Sprungsuche (Liste)", "linked_list", "list_jump", requires_sorted=True),
        Strategy("list_hash", "Liste in Hashtabelle kopieren + Hash-Lookup", "linked_list", "hash", preprocessing="list_hash_build"),
    ]
}


def strategies_for(structure: str) -> list[Strategy]:
    return [s for s in STRATEGIES.values() if s.structure == structure]


@dataclass(frozen=True)
class Workload:
    """What the user tells the tool about the expected queries.

    Attributes:
        queries: expected number of queries q.
        hit_ratio: expected share of queries whose key is present.
        position: where hits are expected, ``"uniform"`` or ``"front"``.
        front_mean: for ``"front"``: mean hit position relative to n.
        sorted_known: the user guarantees that the data is sorted, so the
            O(n) check can be skipped.
    """

    queries: int
    hit_ratio: float = 1.0
    position: str = "uniform"
    front_mean: float = 0.01
    sorted_known: bool = False

    def __post_init__(self) -> None:
        if self.queries < 1:
            raise ValueError("at least one query is expected")
        if not 0.0 <= self.hit_ratio <= 1.0:
            raise ValueError("hit_ratio must lie in [0, 1]")
        if self.position not in POSITIONS:
            raise ValueError(f"unknown position {self.position!r}")
        if not 0.0 < self.front_mean <= 1.0:
            # linear_scans would be 0 and its logarithm undefined
            raise ValueError("front_mean must lie in (0, 1]")

    @property
    def scan_fraction(self) -> float:
        """Expected share of the data a linear search reads per query.

        Hits at uniform positions: 1/2; hits near the front: ``front_mean``
        (the mean position of the generator, see data/queries.py); misses read
        everything.
        """
        hit_share = 0.5 if self.position == "uniform" else self.front_mean
        return self.hit_ratio * hit_share + (1.0 - self.hit_ratio) * 1.0

    @property
    def linear_scans(self) -> float:
        """Expected number of complete passes of linear search over all q queries.

        Preprocessing (O(n)) and the sortedness check (one pass) pay off
        roughly when they cost less than the passes of linear search they
        replace, so this single number drives the amortization rules.
        """
        return self.queries * self.scan_fraction
