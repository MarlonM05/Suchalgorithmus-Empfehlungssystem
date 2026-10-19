# Suchalgorithmus-Empfehlungssystem

Hochschulprojekt (DHBW Ravensburg, Modul Theoretische Informatik): ein Tool,
das anhand der Eigenschaften einer Datenmenge ein passendes Suchverfahren
empfiehlt, es automatisch ausführt und den Entscheidungsweg grafisch darstellt.

> Stand: Meilenstein M4 (Suchverfahren, Daten, Merkmale, Messumgebung, Experimente, Empfehlungssystem). Die vollständige README folgt in M6.

## Installation

Voraussetzung: Python 3.11 oder neuer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest
```

## Implementierte Suchverfahren

| Schlüssel | Verfahren | Voraussetzung | Komplexität | Quelle |
|---|---|---|---|---|
| `linear` | Lineare Suche | keine | O(n) | [vansandt2019] |
| `binary` | Binäre Suche | sortiert | O(log n) | [khuong2017] |
| `binary_branchless` | Binäre Suche (verzweigungsfrei) | sortiert | O(log n) | [khuong2017] |
| `jump` | Sprungsuche | sortiert | O(√n) | [shneiderman1978] |
| `exponential` | Exponentielle Suche | sortiert | O(log i) | [bentley1976] |
| `fibonacci` | Fibonacci-Suche | sortiert | O(log n) | [ferguson1960] |
| `interpolation` | Interpolationssuche mit Guard | sortiert, numerisch | O(log log n) erwartet | [peterson1957], [vansandt2019] |
| `hash` | Hash-Lookup | Hashtabelle (Aufbau O(n)) | O(1) erwartet | [richter2015] |

Auf verketteten Listen gibt es zusätzlich `list_linear` und `list_jump`
(Rückgabe: Position in der Liste).

Alle Verfahren liefern den Index des ersten Vorkommens oder `-1` und liegen in
einer Mess- und einer Zählvariante vor (siehe
`src/recommender/algorithms/_common.py`). Annahmen stehen in
[docs/annahmen.md](docs/annahmen.md).

## Daten und Merkmale

- `recommender.data`: synthetische Datenmengen (`DatasetConfig`: Größe,
  Verteilung, Sortiergrad, Duplikatanteil, Array oder verkettete Liste) und
  Suchanfragen (`QueryConfig`: Anzahl, Trefferquote, Zielposition).
- `recommender.features`: Merkmale per Stichprobe (`extract_features`) inklusive
  gemessener Analysekosten sowie die exakte Sortierprüfung `verify_sorted`.

## Experimente

Die Skripte in `experiments/` erzeugen die Messdaten (CSV mit
Umgebungsprotokoll) und Diagramme (PDF) für die Ausarbeitung.

```bash
python experiments/run_all.py --mode quick    # Funktionsprüfung, ca. 30 Sekunden
python experiments/run_all.py --mode full     # vollständige Messreihe für die Arbeit
python experiments/run_all.py --plot-only     # nur Diagramme aus vorhandenen CSV neu zeichnen
```

Ergebnisse landen in `results/quick/` bzw. `results/full/`. Vor der
vollständigen Messreihe: Netzteil anschließen, **Stromsparmodus ausschalten**
(Systemeinstellungen → Batterie → Stromsparmodus „Nie“), andere Programme
schließen, den Rechner während der Messung nicht benutzen. Im Stromsparmodus
laufen alle Messungen etwa doppelt so langsam. Die Experimente warnen davor,
und jede Ergebnisdatei protokolliert Stromquelle und Stromsparmodus.

```bash
pmset -g | grep lowpowermode    # muss "lowpowermode 0" zeigen
```

| Skript | Merkmal | Diagramme |
|---|---|---|
| `exp_size.py` | Datengröße | Laufzeit und Vergleiche über n, Speedup-Tabelle |
| `exp_distribution.py` | Verteilung | Laufzeit je Verteilung (mit KS-Wert) |
| `exp_duplicates.py` | Duplikate | Laufzeit je Duplikatanteil |
| `exp_access_pattern.py` | Zugriffsmuster | Laufzeit je Muster |
| `exp_structure.py` | Speicherstruktur | Array gegenüber verketteter Liste |
| `exp_amortization.py` | Sortierung + Anfragen | Gesamtkosten T(q), Break-even-Punkte |
| `exp_analysis_cost.py` | Analysekosten | Kosten gegenüber Suche, Genauigkeit nach Stichprobengröße |
| `exp_grid.py` | alle Merkmale kombiniert | Messraster für das Empfehlungssystem (nur CSV) |
| `exp_recommender.py` | – | Bewertung von Regelwerk und Entscheidungsbaum, Tabellen (CSV, LaTeX) |

## Empfehlungssystem

`recommender.rules` wählt eine **Strategie** (Suchverfahren samt
Vorverarbeitung) für Daten und erwartete Anfragen:

- **Regelwerk**: Entscheidungsgraph in
  [`src/recommender/rules/rules.yaml`](src/recommender/rules/rules.yaml). Jeder
  Knoten hat eine Frage, Schwelle, Begründung und Quellen, jedes Blatt eine
  Empfehlung mit Begründung in einem Satz. Die Schwellen stammen aus den
  Messungen in `results/full/`.
- **Entscheidungsbaum**: scikit-learn (Entropie, Tiefe 4), trainiert auf dem
  Messraster und gespeichert als `results/<modus>/learned_tree.json`.
- **Bewertung** (`exp_recommender.py`): Trefferquote, Slowdown, geschlossene
  Lücke zwischen Single Best Solver und Oracle, jeweils mit und ohne
  Analysekosten, auf einer getrennten Testhälfte.
