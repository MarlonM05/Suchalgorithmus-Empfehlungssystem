# Suchalgorithmus-Empfehlungssystem

Hochschulprojekt (DHBW Ravensburg, Modul Theoretische Informatik): ein Tool,
das anhand der Eigenschaften einer Datenmenge ein passendes Suchverfahren
empfiehlt, es automatisch ausführt und den Entscheidungsweg grafisch darstellt.

> Stand: Meilenstein M2 (Suchverfahren, Datengeneratoren, Merkmalsextraktion). Die vollständige README folgt in M6.

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
