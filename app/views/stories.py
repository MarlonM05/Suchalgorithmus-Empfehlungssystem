"""The stories of the presentation mode, as plain data.

Every story is an everyday situation that the audience understands at once
(a phone book, a log file, ...) and that leads to a different recommendation.
Together they cover the main findings of the paper:

=================  ======================================  =========================
Story              Situation                               Recommendation
=================  ======================================  =========================
Telefonbuch        sorted, many queries                    binary search (checked)
Logdatei           unsorted, the wanted entries are new     linear search
Lagerbestand       unsorted, very many queries             hash table
Einkaufsliste      tiny (n = 10)                           linear search
Messwerte          sorted but skewed                       binary search, not interpolation
Playlist           linked list, few queries                linear search through the list
=================  ======================================  =========================

The texts state the expected recommendation, so ``expected`` is checked by
the tests: if the rule book or the generators change, the tests show which
story text no longer fits.
"""

from dataclasses import dataclass

import numpy as np

from recommender.data import DatasetConfig, LinkedList, build_linked_list, generate_keys, rescale_readable
from recommender.rules import Workload


@dataclass(frozen=True)
class Story:
    """One story of the presentation mode.

    Attributes:
        key: identifier (also used in the URL-free session state).
        icon: emoji shown on the card and in the title.
        title: short name.
        hook: one sentence on the card.
        scenario: the everyday situation, two or three sentences.
        dataset: how the data is generated.
        workload: what the user would tell the tool about the queries.
        expected: strategy the rule book recommends (checked by the tests).
        race: algorithms of the race (``recommender.viz.stepper``), empty if
            the recommended method cannot be shown as a race (hash table).
        lesson: the take-away message in one or two sentences.
    """

    key: str
    icon: str
    title: str
    hook: str
    scenario: str
    dataset: DatasetConfig
    workload: Workload
    expected: str
    race: tuple[str, ...]
    lesson: str

    def keys(self) -> np.ndarray:
        """The keys in logical order with readable values that keep the shape of the distribution (A39)."""
        return rescale_readable(generate_keys(self.dataset))

    def data(self) -> np.ndarray | LinkedList:
        """The data in its storage structure (same node placement as generate_dataset)."""
        keys = self.keys()
        if self.dataset.structure == "linked_list":
            return build_linked_list(keys, np.random.default_rng(self.dataset.seed + 1))
        return keys

    def search_key(self, keys: np.ndarray) -> int:
        """A key at the position where this workload typically finds its hits.

        Uniform hits: the middle on average; hits near the front: the mean
        front position (1 % of n). This makes the measured time of the linear
        search representative for the story.
        """
        position = int(self.workload.scan_fraction * (keys.size - 1))
        return int(keys[position])


STORIES: dict[str, Story] = {
    s.key: s
    for s in [
        Story(
            key="telefonbuch",
            icon="📞",
            title="Telefonbuch",
            hook="Sortierte Einträge, viele Nachschläge am Tag.",
            scenario=(
                "Ein Telefonbuch mit 100.000 Einträgen, alphabetisch sortiert. "
                "Am Tag wird etwa 1.000-mal nachgeschlagen, jeder Name ist gleich wahrscheinlich."
            ),
            dataset=DatasetConfig(n=100_000, distribution="uniform", order="sorted", seed=11),
            workload=Workload(queries=1_000),
            expected="binary_branchless",
            race=("linear", "binary"),
            lesson=(
                "Sortierte Daten sind ein Geschenk: Nach einer einmaligen Prüfung, ob wirklich sortiert ist, "
                "findet die binäre Suche jeden Eintrag in etwa 17 Schritten statt in durchschnittlich 50.000."
            ),
        ),
        Story(
            key="logdatei",
            icon="📜",
            title="Logdatei",
            hook="Ungeordnet, aber gesucht wird fast nur das Neueste.",
            scenario=(
                "Eine Logdatei mit 100.000 Fehlermeldungen, ungeordnet gespeichert, die neuesten stehen vorne. "
                "Etwa 100-mal wird nach einer Meldung gesucht, fast immer nach einer aktuellen."
            ),
            dataset=DatasetConfig(n=100_000, distribution="uniform", order="unsorted", seed=12),
            workload=Workload(queries=100, position="front"),
            expected="linear",
            race=("linear",),
            lesson=(
                "Nicht nur die Daten, auch das Suchverhalten zählt: Liegen die gesuchten Einträge vorne, "
                "ist die einfache lineare Suche kaum zu schlagen. Sortieren oder Hashen lohnt sich erst bei sehr vielen Anfragen."
            ),
        ),
        Story(
            key="lager",
            icon="📦",
            title="Lagerbestand",
            hook="Ungeordnete Artikelnummern, tausende Abfragen.",
            scenario=(
                "Ein Lager mit 100.000 Artikelnummern in zufälliger Reihenfolge. "
                "An der Kasse wird 10.000-mal am Tag geprüft, ob ein Artikel vorrätig ist."
            ),
            dataset=DatasetConfig(n=100_000, distribution="uniform", order="unsorted", seed=13),
            workload=Workload(queries=10_000),
            expected="hash",
            race=(),
            lesson=(
                "Bei vielen Anfragen auf ungeordneten Daten lohnt sich Vorarbeit: Einmal eine Hashtabelle bauen "
                "kostet so viel wie einige Dutzend lineare Suchen, danach braucht jede Anfrage nur noch einen Blick."
            ),
        ),
        Story(
            key="einkaufsliste",
            icon="🛒",
            title="Einkaufsliste",
            hook="Zehn Einträge, ein paar Blicke darauf.",
            scenario=(
                "Eine Einkaufsliste mit 10 Einträgen, alphabetisch geordnet. "
                "Im Laden wird fünfmal nachgesehen, ob etwas draufsteht."
            ),
            dataset=DatasetConfig(n=10, distribution="uniform", order="sorted", seed=14),
            workload=Workload(queries=5),
            expected="linear",
            race=("linear", "binary"),
            lesson=(
                "Bei zehn Einträgen ist Nachdenken teurer als Nachschauen: Die lineare Suche ist hier sogar "
                "schneller als die binäre, und jede Vorbereitung wäre verschwendet."
            ),
        ),
        Story(
            key="messwerte",
            icon="📈",
            title="Messwerte",
            hook="Sortiert, aber sehr ungleich verteilt.",
            scenario=(
                "100.000 Messwerte eines Sensors, sortiert gespeichert. Die meisten Werte sind klein, "
                "einige wenige riesig (lognormalverteilt). Es wird 1.000-mal nach einem Wert gesucht."
            ),
            dataset=DatasetConfig(n=100_000, distribution="lognormal", order="sorted", seed=15),
            workload=Workload(queries=1_000),
            expected="binary_branchless",
            race=("binary", "interpolation"),
            lesson=(
                "Ob kluges Raten hilft, hängt von der Verteilung ab: Bei schief verteilten Werten verschätzt sich "
                "die Interpolationssuche immer wieder, die binäre Suche bleibt verlässlich schnell."
            ),
        ),
        Story(
            key="playlist",
            icon="🎵",
            title="Playlist",
            hook="Eine verkettete Liste, nur der Reihe nach lesbar.",
            scenario=(
                "Eine Playlist mit 10.000 Liedern als verkettete Liste: Jedes Lied kennt nur das nächste. "
                "Ab und zu, etwa dreimal, wird ein bestimmtes Lied gesucht."
            ),
            dataset=DatasetConfig(n=10_000, distribution="uniform", order="unsorted", structure="linked_list", seed=16),
            workload=Workload(queries=3),
            expected="list_linear",
            race=("linear",),
            lesson=(
                "Die Speicherstruktur bestimmt, was überhaupt möglich ist: In einer verketteten Liste gibt es keine "
                "Sprünge, nur Schritt für Schritt. Erst bei vielen Suchen lohnt es sich, sie einmal in eine Hashtabelle zu kopieren."
            ),
        ),
    ]
}
