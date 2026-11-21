# Quellen im Code

Übersicht, welche Quelle an welcher Stelle der Implementierung verwendet wird.
Die Schlüssel entsprechen denen der Ausarbeitung. So lässt sich jede
Entscheidung im Code einer Quelle zuordnen und umgekehrt.

| Schlüssel | Wofür im Projekt | Stellen |
|---|---|---|
| [bentley1976] | Exponentielle Suche (unbeschränkte Suche) | `algorithms/exponential.py`, `experiments/exp_access_pattern.py` |
| [bischl2016] | Bewertung der Auswahl (Oracle, Single Best) | `rules/evaluation.py` |
| [charikar2000] | Anzahl verschiedener Werte ist aus einer Stichprobe nicht zuverlässig schätzbar → Duplikatanteil nur bei sortierten Daten | `features/extraction.py`, App-Seite „Analyse“ – **neu, noch nicht im Literaturverzeichnis; Angaben vor dem Zitieren prüfen** |
| [chen2016] | Blockmessung, robuste Statistik (Median, Minimum, Bootstrap) | `benchmark/timing.py`, `benchmark/statistics.py` |
| [ferguson1960] | Fibonacci-Suche | `algorithms/fibonacci.py` |
| [fleming1986] | Geometrisches Mittel für Speedups und Slowdowns | `benchmark/statistics.py`, `rules/evaluation.py` |
| [georges2007] | Aufwärmläufe verwerfen | `benchmark/timing.py` |
| [gonnet1980] | Worst Case der Interpolationssuche | `algorithms/interpolation.py` |
| [hoefler2015] | Umgebung protokollieren, Konfidenzintervalle | `benchmark/environment.py`, `benchmark/statistics.py` |
| [hundhausen2002] | Lernmodus: aktive Beteiligung statt bloßer Animation | `viz/stepper.py`, App-Seite „Lernmodus“ |
| [kalibera2013] | Wiederholungen und Aufwärmläufe | `benchmark/timing.py` |
| [kerschke2019] | Algorithmenauswahl: günstige Merkmale, Oracle/SBS, geschlossene Lücke | `features/extraction.py`, `rules/evaluation.py`, `rules/instances.py`, `rules/rules.yaml`, `experiments/exp_recommender.py` |
| [khuong2017] | Verzweigungsfreie binäre Suche | `algorithms/binary.py`, `rules/rules.yaml` |
| [li2004] | Kennzahlen der Daten zur Laufzeit schätzen (Stichprobe) | `features/extraction.py` |
| [mytkowicz2009] | Messverzerrung: zufällige Reihenfolge, feste Seeds, wechselnde Eingaben | `benchmark/runner.py`, `benchmark/timing.py`, `benchmark/experiment.py`, `data/generators.py` |
| [perl1978] | Analyse der Interpolationssuche | `algorithms/interpolation.py` |
| [peterson1957] | Interpolationssuche | `algorithms/interpolation.py`, `features/extraction.py` (KS-Merkmal), `experiments/exp_distribution.py` |
| [quinlan1986] | Entscheidungsbaum mit Informationsgewinn (ID3) | `rules/learned.py` |
| [rice1976] | Algorithmenauswahlproblem: Problem-, Merkmals-, Algorithmenraum, Leistungsmaß | `rules/strategies.py`, `rules/instances.py`, `docs/architektur.md` |
| [richter2015] | Hashtabelle (multiplikatives Hashing, lineares Sondieren); Entscheidungsgraph als Vorbild | `algorithms/hashing.py`, `rules/rules.yaml` |
| [shneiderman1978] | Sprungsuche, auch auf Listen | `algorithms/jump.py`, `algorithms/linked_list.py`, `rules/rules.yaml`, `experiments/exp_structure.py` |
| [vansandt2019] | Lineare Suche auf kleinen Arrays, Guard der Interpolationssuche (Guard-Größe 8), Lognormal µ = 0, σ = 2, Stichproben-Meta-Algorithmus | `algorithms/interpolation.py`, `algorithms/linear.py`, `data/generators.py`, `benchmark/grid.py`, `rules/rules.yaml`, `experiments/exp_recommender.py` |
| [vilasboas2021] | Geringe Baumtiefe für Interpretierbarkeit | `rules/learned.py` |

Aus der Literaturliste, aber an keiner einzelnen Codestelle verwendet:
[smithmiles2009] und [kotthoff2014] (Überblick über das
Algorithmenauswahlproblem). Sie gehören in die Einordnung im Theorieteil.

Alle Pfade beziehen sich auf `src/recommender/`, sofern nicht anders angegeben.
