# Suchalgorithmus-Empfehlungssystem

Hochschulprojekt (DHBW Ravensburg, Modul Theoretische Informatik): ein Tool,
das anhand der Eigenschaften einer Datenmenge ein passendes Suchverfahren
empfiehlt, es automatisch ausführt und den Entscheidungsweg grafisch darstellt.

> Stand: Meilenstein M5 (alles bis einschließlich Oberfläche). Die vollständige README folgt in M6.

## Installation

Voraussetzung: Python 3.11 oder neuer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Oberfläche starten

```bash
streamlit run app/streamlit_app.py
```

Die App öffnet sich im Browser und führt in sechs Schritten durch das Tool:

1. **Daten**: Datenmenge erzeugen oder CSV hochladen
2. **Zugriffsmuster**: Suchschlüssel, Anzahl Anfragen, Trefferquote, Zielposition, Sortiergarantie
3. **Analyse**: geschätzte Merkmale samt Analysekosten, vollständige Sortierprüfung
4. **Empfehlung**: Strategie mit Begründung, Steckbrief (Komplexität, Vorverarbeitung,
   Voraussetzung, Quellen), Entscheidungsweg als Graph (Regelwerk oder gelernter Baum)
   und ein animiertes **Wettrennen** der linearen Suche gegen das empfohlene Verfahren
   auf einem Ausschnitt der eigenen Daten
5. **Ausführung**: automatische Suche mit Laufzeit und Operationen, Vergleich aller Verfahren
6. **Lernmodus**: Suchverfahren Schritt für Schritt auf kleinen Arrays oder als
   **Wettrennen** mehrerer Verfahren nebeneinander (gleiches Array, gleicher Schlüssel,
   ein Vergleich pro Takt, Schrittzähler je Verfahren)

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
| `interpolation` | Interpolationssuche mit Guard | sortiert, numerisch | O(log log n) erwartet, O(n) im Worst Case | [peterson1957], [perl1978], [gonnet1980], [vansandt2019] |
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
python experiments/run_all.py --mode quick    # Funktionsprüfung (Grid zuerst, Laufzeit variiert)
python experiments/run_all.py --mode full     # vollständige Messreihe für die Arbeit
python experiments/run_all.py --mode full --results-dir results/full_y  # abweichender Ausgabeordner
python experiments/run_all.py --plot-only     # nur Diagramme aus vorhandenen CSV neu zeichnen
```

Die Grid-Dateien sind gemeinsame Rohdaten für die Empfehlungsauswertung und
Amortisationsanalyse. `grid_features.csv` beschreibt Datensätze,
`grid_components.csv` enthält Zeitkomponenten, `grid_operations.csv` die
gezählten Vergleiche/Zugriffe und `grid_sampling.csv` die Probeläufe. Die
Amortisationsanalyse nutzt nur exakt passende Grid-Szenarien und schreibt
`amortization_summary.csv` sowie `amortization_validation.csv` mit ausgewählten
vollständigen Läufen. In `amortization.pdf` steht q für die Zahl der Anfragen,
die y-Achse für die modellierte Gesamtzeit; markierte Schnittpunkte zeigen,
ab wann Sortieren plus binäre Suche bzw. Hashing die lineare Suche überholen.
Das Grid muss daher vor `exp_amortization.py` vorliegen;
`run_all.py` führt es in der richtigen Reihenfolge aus.

Ergebnisse landen standardmäßig in `results/quick/` bzw. `results/full/`;
mit `--results-dir` lässt sich für einen Lauf ein abweichender Ausgabeordner
festlegen. Die Ordner im Repository:

| Ordner | Rechner | Inhalt |
|---|---|---|
| `results/full/` | Messlaptop (Apple M5, macOS) | Messreihe M3 und Messraster M4 (540 Datenmengen); **Grundlage aller Zahlen in Bericht und Präsentation** (Annahme A18) |
| `results/full_y/`, `results/full_y2/` | Intel Core Ultra 5 245KF (Windows) | zwei vollständige Läufe mit der erweiterten Messmethodik (Raster mit 818 Datenmengen, vier Zugriffsmuster, Replikate, Amortisation aus dem Raster); dienen dem Vergleich von Verhältnissen zwischen Rechnern, nicht für absolute Zeiten (A39, A41) |

Das alte `results/full/amortization.csv` stammt noch aus der eigenständigen
Amortisationsmessung (Stand vor `exp_grid`-Umbau) und bleibt als Messartefakt
erhalten; `exp_amortization.py` liest seit dem Umbau das Raster und kommt auch
mit dem älteren Rasterformat in `results/full/` zurecht (A39). Vor der
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
| `exp_size_datasets.py` | Datengröße (Robustheit) | wie `exp_size.py`, aber 5 Datensätze pro Größe; zeigt, ob Ergebnisse vom Datensatz abhängen (nicht in `run_all.py`, Rechnername über `--label`) |
| `exp_grid.py` | alle Merkmale kombiniert | Messraster für das Empfehlungssystem (nur CSV) |
| `exp_recommender.py` | – | Bewertung von Regelwerk und Entscheidungsbaum, Tabellen (CSV, LaTeX) |

## Dokumentation für Bericht und Präsentation

[docs/review_und_lernmaterial.md](docs/review_und_lernmaterial.md) enthält
eine vollständige Durchsicht des Repositorys (Aufbau, Befunde, Änderungen),
den Abgleich der Präsentationszahlen mit den Messdaten, die Zuordnung der
Quellen zu den Aussagen und Lernmaterial zu allen Konzepten des Projekts.

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
