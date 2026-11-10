# Durchsicht, Befunde und Lernmaterial zum Suchalgorithmus-Empfehlungssystem

Stand: 3. November 2026, Branch `claude/vibrant-sagan-ap0l1w` (enthält `main`
und Yanniks Branch `messmethodik_test`).

Diese Datei ist so geschrieben, dass sie sich komplett kopieren und an eine
KI weitergeben lässt, um daraus ein Lerntool zu bauen oder Teile in den
Projektbericht und die Präsentation zu übernehmen. **Alle Aussagen stammen
ausschließlich aus dem Repository (Code, `docs/annahmen.md`, `README.md`,
Messdaten in `results/`) und aus den Quellen im Drive-Ordner
`DHBW/3.Semester/Theoretische Informatik/Quellen` sowie der dort liegenden
Präsentation `Empfehlungssystem.pptx`.** Wo eine Zahl steht, ist angegeben,
aus welcher Datei sie kommt. Quellenschlüssel in eckigen Klammern (z. B.
`[khuong2017]`) sind dieselben wie im Code und in `rules.yaml`.

Inhalt:

1. Was gemacht wurde (Vorgehen)
2. Aufbau des Repositorys
3. Die Suchverfahren (Lernstoff)
4. Daten, Merkmale, Zugriffsmuster
5. Messmethodik
6. Das Empfehlungssystem
7. Ergebnisse in Zahlen (Messlaptop Apple M5)
8. Yanniks Änderungen (Branch `messmethodik_test`)
9. Befunde: Fehler und Inkonsistenzen
10. Änderungen in dieser Durchsicht
11. Abgleich der Präsentation mit den Messdaten
12. Quellen: Zuordnung zu den Aussagen des Projekts
13. Lernmaterial: Glossar, Fragen und Antworten, Rechenbeispiele
14. Bausteine für Bericht und Präsentation
15. Offene Punkte für das Team

---

## 1. Was gemacht wurde (Vorgehen)

1. Alle Quelldateien gelesen: `src/recommender/**` (Algorithmen, Daten,
   Merkmale, Benchmark, Regeln, Visualisierung, Executor), `app/**`
   (Streamlit-Oberfläche), `experiments/**`, `tests/**`, `docs/annahmen.md`,
   `README.md`, `pyproject.toml`, `requirements.txt`, CI-Workflow.
2. Testsuite ausgeführt: vor dem Merge 502 Tests bestanden, nach dem Merge
   von Yanniks Branch 512, nach den eigenen Änderungen siehe Abschnitt 10.
3. Yanniks Push angesehen (Abschnitt 8): Branch `messmethodik_test`, ein
   Commit auf `main` aufgesetzt. Gemerged, Tests bestanden, Kompatibilität mit
   den alten Apple-M5-Ergebnissen geprüft und hergestellt.
4. Messdaten in `results/full/` (Apple M5) gegen jede Zahl der Präsentation
   geprüft (Abschnitt 11).
5. Die Quellen im Drive-Ordner gelesen (gezielt die Passagen, auf die sich
   der Code beruft) und den Aussagen des Projekts zugeordnet (Abschnitt 12).
6. Oberfläche erweitert: animiertes Wettrennen mehrerer Suchverfahren auf
   demselben Array (Lernmodus und Empfehlungsseite), Steckbrief der
   empfohlenen Strategie, Kostenmodell mit eingesetztem q (Abschnitt 10).
7. Kleine Fehler behoben, Dokumentation ergänzt (A39 in `docs/annahmen.md`,
   README).
8. Die App im Browser gestartet und beide neuen Ansichten mit Screenshots
   geprüft; alle Tests erneut ausgeführt; committet und gepusht.

---

## 2. Aufbau des Repositorys

```
Suchalgorithmus-Empfehlungssystem/
├── app/                      Streamlit-Oberfläche (6 Seiten)
│   ├── streamlit_app.py      Navigation, Sidebar
│   └── views/
│       ├── common.py         Session-State, Formatierung, Cache (Regelwerk, Baum, Kompilierung)
│       ├── data_page.py      1 · Daten: erzeugen oder CSV laden
│       ├── workload_page.py  2 · Zugriffsmuster: Suchschlüssel, q, Trefferquote, Position, Sortiergarantie
│       ├── analysis_page.py  3 · Analyse: Stichprobenmerkmale + Analysekosten, vollständige Sortierprüfung
│       ├── recommendation_page.py  4 · Empfehlung: Strategie, Steckbrief, Entscheidungsweg, Wettrennen
│       ├── execution_page.py 5 · Ausführung: automatische Suche, Vergleich aller Strategien
│       ├── learning_page.py  6 · Lernmodus: Schritt für Schritt oder Wettrennen
│       └── race_view.py      gemeinsamer Wettrennen-Block (neu)
├── src/recommender/          das eigentliche Paket (pip install -e .)
│   ├── algorithms/           8 Array-Verfahren + 2 Listen-Verfahren, Numba-kompiliert, Registry
│   ├── data/                 Generatoren (Verteilung, Sortiergrad, Duplikate), verkettete Liste, Anfragen
│   ├── features/             Stichprobenmerkmale, KS-Abstand, vollständige Sortierprüfung
│   ├── benchmark/            Zeitmessung (Blöcke), Statistik (Median, Bootstrap, geometrisches Mittel),
│   │                         Umgebungsprotokoll, Experimentgerüst, Messraster
│   ├── rules/                Strategien, Regelwerk (rules.yaml), gelernter Baum, Instanzen, Bewertung
│   ├── viz/                  Graphviz-Entscheidungsgraphen, Schrittfolgen (stepper), Wettrennen (race, neu), Diagramme
│   └── executor.py           führt eine Strategie auf echten Daten aus (Prüfung, Vorverarbeitung, Suche, Zeit)
├── experiments/              Messskripte (ein Merkmal pro Skript), Raster, Bewertung, run_all.py
├── results/full/             Messlaptop Apple M5 (Grundlage von Bericht und Präsentation)
├── results/full_y, full_y2/  Intel Core Ultra 5 245KF, erweiterte Methodik (Yannik)
├── tests/                    pytest, 523 Tests (nach dieser Durchsicht)
└── docs/annahmen.md          A1 bis A41: jede getroffene Annahme mit Begründung
```

**Datenfluss des Tools** (so läuft eine Empfehlung ab):

```
Datenmenge (Array oder Liste) + Angaben des Nutzers (q, Trefferquote, Position, Sortiergarantie)
        │
        ▼
features.extract_features  →  n, Struktur, Sortiergrad (Stichprobe), Duplikatanteil, KS-Abstand, Analysekosten
        │
        ▼
rules.decision.selection_features  →  flaches Dict aller Merkmale inkl. abgeleiteter Größen
        │                               (erwartete Durchläufe der linearen Suche, q/n, log10 n, ...)
        ▼
RuleBook.decide (rules.yaml)   oder   LearnedSelector.decide (learned_tree.json)
        │
        ▼
Recommendation(strategy, reason, path, leaf, sources)  →  Graph (viz.graph), Steckbrief, Wettrennen
        │
        ▼
executor.execute  →  Sortierprüfung (falls nötig), Vorverarbeitung, Suche, Zeiten, Operationen
```

**Entwurfsprinzipien, die sich durch den Code ziehen**

- *Eine Implementierung, zwei Varianten*: Jeder Algorithmus wird einmal
  geschrieben und mit `count_ops=True/False` zweimal kompiliert. Numba
  behandelt die Closure-Variable als Konstante und entfernt den Zählcode
  in der Messvariante (`algorithms/_common.py`, A4).
- *Erstes Vorkommen*: Alle Verfahren auf sortierten Daten suchen die
  untere Schranke (kleinster Index mit `arr[i] >= key`) und prüfen dann auf
  Gleichheit (`make_check_candidate`). Referenz in den Tests ist
  `np.searchsorted(side="left")` (A1).
- *Regeln sind Daten*: Das Regelwerk liegt als YAML vor und wird beim Laden
  validiert (unbekannte Ziele, Operatoren, Merkmale, Strategien, Zyklen).
- *Stichprobe zuletzt*: Merkmale, die eine Stichprobe brauchen, werden im
  Regelwerk erst gefragt, wenn kostenlose Merkmale nicht reichen, weil die
  Stichprobe selbst Zeit kostet (`[kerschke2019]`: Merkmale müssen billig
  sein).
- *Lineares Kostenmodell*: T(q) = Prüfung + Vorverarbeitung + q · Zeit pro
  Anfrage (A22). q wird nicht gemessen, sondern eingerechnet.

---

## 3. Die Suchverfahren (Lernstoff)

Alle Verfahren arbeiten auf `int64`-Arrays (A2) und liefern den Index des
ersten Vorkommens oder −1 (A1). Gezählt werden Schlüsselvergleiche und
Zugriffe auf gespeicherte Elemente, nicht Vergleiche von Schleifenindizes
(A3). Die Tabelle fasst zusammen, was der Code tut; die Spalte „Rolle im
System“ stammt aus `rules.yaml` und den Messergebnissen.

| Verfahren (Schlüssel) | Idee | Voraussetzung | Komplexität | Rolle im System | Quelle |
|---|---|---|---|---|---|
| Lineare Suche (`linear`) | Element für Element von links nach rechts | keine | O(n) | Standard bei n ≤ 16, bei unsortierten Daten mit wenigen Anfragen, auf Listen | `[vansandt2019]` (schnellste auf sehr kleinen Arrays) |
| Binäre Suche (`binary`) | Mitte prüfen, Hälfte verwerfen; Lower-Bound-Variante | sortiert, Direktzugriff | O(log n) | Referenz der Speedups | `[khuong2017]` |
| Verzweigungsfreie binäre Suche (`binary_branchless`) | wie binär, aber `base += half * (arr[base+half] < key)` statt if/else, immer ⌈log₂ n⌉ Runden | sortiert | O(log n) | **die Empfehlung für sortierte Daten ohne Vorverarbeitung** (in allen M5-Messungen schnellstes Verfahren ohne Vorverarbeitung) | `[khuong2017]` |
| Sprungsuche (`jump`) | Blöcke der Größe ⌊√n⌋ überspringen, dann im Block sequenziell | sortiert | O(√n) | nur auf sortierten **Listen** empfohlen (spart Vergleiche, besucht aber fast alle Knoten) | `[shneiderman1978]` |
| Exponentielle Suche (`exponential`) | Grenze verdoppeln (1, 2, 4, …), dann binär im letzten Intervall | sortiert | O(log i), i = Position | wird nie empfohlen: selbst bei vorne gehäuften Treffern langsamer als die verzweigungsfreie binäre Suche (38 gegenüber 17 ns, `access_pattern.csv`) | `[bentley1976]` |
| Fibonacci-Suche (`fibonacci`) | Teilung nach Fibonacci-Zahlen, nur Addition/Subtraktion | sortiert | O(log n) | historisch; nie empfohlen (73 ns gegenüber 25 ns bei n = 10⁶) | `[ferguson1960]` |
| Interpolationssuche mit Guard (`interpolation`) | Position aus dem Wert schätzen: `pos = lo + (key−a[lo])/(a[hi−1]−a[lo])·(hi−1−lo)`; Guard: liegt die Schätzung ≤ 8 Elemente vom Rand, sequenziell weiter | sortiert, numerisch | O(log log n) erwartet bei Gleichverteilung, O(n) im Worst Case | nie empfohlen: auf dem Messlaptop in keiner Messung schneller als die verzweigungsfreie binäre Suche; bei lognormalen Daten so langsam wie lineare Suche | `[peterson1957]`, `[perl1978]`, `[gonnet1980]`, `[vansandt2019]` |
| Hash-Lookup (`hash`) | eigene Tabelle: multiplikatives Hashing `h(k) = (k·A mod 2⁶⁴) >> (64−b)`, A = ⌊2⁶⁴/φ⌋, lineares Sondieren, Kapazität = kleinste Zweierpotenz ≥ 2n (Füllgrad ≤ 0,5) | Aufbau O(n) | O(1) erwartet | **die Empfehlung bei vielen Anfragen** (Aufbau kostet 7 bis 34 Durchläufe der linearen Suche) | `[richter2015]` |
| Lineare Suche auf Liste (`list_linear`) | Zeiger folgen, jeden Knoten vergleichen | keine | O(n) | Standard auf Listen mit wenigen Anfragen | |
| Sprungsuche auf Liste (`list_jump`) | nur den letzten Knoten jedes Blocks vergleichen, Blockanfang merken | sortierte Liste | O(√n) Vergleiche, O(n) Knotenbesuche | empfohlen bei garantiert sortierter Liste und wenigen Anfragen | `[shneiderman1978]` |

**Warum es drei „Familien“ gibt (Gliederung der Präsentation, Folie 9):**

1. *Sequenziell und sprungbasiert*: lineare Suche, Sprungsuche,
   exponentielle Suche (die Präsentation ordnet die exponentielle Suche hier
   ein, weil sie mit Sprüngen beginnt).
2. *Bisektionsbasiert*: klassische und verzweigungsfreie binäre Suche,
   Fibonacci-Suche (halbieren bzw. nach Fibonacci-Zahlen teilen).
3. *Positionsberechnend*: Interpolationssuche und Hash-Lookup (die Position
   wird aus dem Schlüsselwert berechnet, nicht erfragt).

**Was man aus den Quellen dazu wissen sollte**

- `[shneiderman1978]` leitet die optimale Sprungweite her: Die mittleren
  Kosten sind C(N) = ½·N/n + ½·(n − 1) für Sprungweite n; Ableitung null
  setzen ergibt n = √N. Jump Search ist dort ausdrücklich für Fälle gedacht,
  in denen „binary searching is not feasible“, etwa verkettete sequenzielle
  Dateien. Zitat: „when binary searching can be done it is hard to do better,
  but some situations exist when jump searching is preferred. The most common
  situation is in sorted files which are linked to maintain a sequential
  ordering.“ Genau so setzt das Projekt sie ein (nur auf Listen).
- `[bentley1976]` („unbounded searching“) beschreibt Algorithmus B₁: erst
  Verdoppeln bis `F(2^m − 1) = Y`, dann gebundene binäre Suche; Kosten
  2⌊lg n⌋ + 1. Das ist die exponentielle Suche des Projekts. Anwendung laut
  Quelle: „searching an extremely large ordered table if one wanted to pay a
  search cost proportional to the item's distance from the front of the
  table“.
- `[ferguson1960]` motiviert die Fibonacci-Suche mit Maschinen ohne
  Shift/Division: „successive increments are found by subtraction“. Mittlere
  Suchzeit O(log n), Worst Case log_φ n.
- `[peterson1957]` ist die Erstbeschreibung der Interpolationsidee
  (Telefonbuch-Beispiel: „If the record with identification No. 1000 is
  desired, it would seem more reasonable to look first at a point about
  one-tenth of the way through the file“) und zugleich die Quelle der
  Hashtabellen-Analyse mit offener Adressierung („open system“): mittlere
  Suchlänge 1,541 bei 50 % und 5,526 bei 90 % Belegung (Tabelle in der
  Quelle; Folie 23 zitiert genau diese Werte).
- `[perl1978]` beweist, dass die Interpolationssuche auf gleichverteilten
  Schlüsseln im Mittel log log N Zugriffe braucht (Supermartingal-Argument),
  und nennt im Experiment mit 400 000 Schlüsseln 4,28 Zugriffe im Mittel
  gegenüber log log 400 000 = 4,21. Wichtig für die Arbeit: Die Quelle sagt
  selbst, dass bei interner Suche binäre Suche und Interpolationssuche „take
  approximately the same time“ und empfiehlt, am Ende auf sequenzielle Suche
  umzuschalten (die Idee des Guards).
- `[gonnet1980]` präzisiert: Erwartungswert lg lg n + O(1), Varianz
  O(lg lg n), gute Näherung C_n ≈ lg lg(n + 3), und warnt: Bei nicht
  gleichverteilten Daten „the algorithm departs significantly from the
  lg lg n behavior“. Das erklärt die 113 740 ns bei lognormalen Daten.
- `[vansandt2019]` (SIP/TIP) liefert den Guard (Alg. 2, Zeilen 14 bis 17)
  und die Guard-Größe 8 („best all-around performer across the range of our
  experiments“, Anhang A.3). Das Projekt übernimmt **nur** den Guard, nicht
  Slope Reuse, Festkomma-Arithmetik oder TIP (A5). Die Lognormal-Parameter
  µ = 0, σ = 2 stammen aus derselben Quelle (A10). Außerdem stammt daher der
  Stichproben-Meta-Algorithmus (jedes Verfahren beantwortet Probeanfragen,
  das schnellste wird gewählt), der in M4 als Vergleichsverfahren dient.
- `[khuong2017]` erklärt, warum verzweigungsfrei schneller ist: Conditional
  Moves vermeiden Branch Mispredictions, die die Pipeline leeren. Zentrale
  Aussage der Quelle: „for small values of n, sorted order, combined with a
  good implementation of binary search is best“; für große n ist dort das
  Eytzinger-Layout am schnellsten, das im Projekt nicht betrachtet wird
  (mögliche Erweiterung für „Ausblick“).
- `[richter2015]` untersucht Hashing für Ganzzahlschlüssel in fünf
  Dimensionen (Datenverteilung, Füllgrad, Größe, Lese-/Schreibverhältnis,
  erfolgreiche/erfolglose Suchen). Befunde, die das Projekt nutzt: „Mult is
  the fastest hash function when integrated with all hashing schemes“ und
  „we always find LP, QP, and RH as the fastest methods“ (lineares,
  quadratisches, Robin-Hood-Sondieren). Die Quelle schließt mit einem
  **Entscheidungsgraphen** („decision graph, Figure 8“) für Praktiker; genau
  das ist das Vorbild für `rules.yaml` (Kopfkommentar).

---

## 4. Daten, Merkmale, Zugriffsmuster

**Datengenerator** (`data/generators.py`, A10 bis A13): ein Pool
verschiedener Schlüssel aus der Verteilung (gleich, normal, exponentiell,
lognormal µ = 0 / σ = 2, geclustert = 10 schmale Normalverteilungen mit
festen Zentren), Auffüllen mit Kopien bis der Duplikatanteil
p = (n − verschiedene)/n exakt stimmt, dann sortiert / fast sortiert
(round(k·n) Vertauschungen, Standard k = 1 %) / unsortiert (Permutation).
Stetige Werte werden mit großen Faktoren (bis 2⁵²) skaliert und abgerundet,
damit das Runden keine ungewollten Duplikate erzeugt.

**Verkettete Liste** (`data/linked_list.py`, A13): Knotenpool aus zwei Arrays
`values` und `next_index`; die logische Reihenfolge liegt in zufälligen
Slots, damit das Durchlaufen wie Zeigerverfolgung im Heap wirkt (kein
Prefetching). Nur sequenzieller Zugriff.

**Anfragen** (`data/queries.py`, A15): q Schlüssel; Trefferquote; Position
`uniform` oder `front` (Exponentialverteilung mit Mittelwert 1 % von n:
63 % der Treffer im ersten Prozent, 99 % in den ersten 4,6 %). Fehlschläge
liegen in Lücken des Wertebereichs.

**Merkmale** (`features/extraction.py`, A16): auf einer Stichprobe von 1000
Nachbarpaaren bzw. Werten (Ziehen mit Zurücklegen, kompiliert):

| Merkmal | Was es misst | Kosten | Hinweis |
|---|---|---|---|
| `n` | Größe | O(1) | |
| `disorder` | Anteil falsch geordneter Nachbarpaare | Stichprobe | 0 sortiert, ≈ 0,5 zufällig, ≈ 2k bei „fast sortiert“ (A12) |
| `looks_sorted` | `disorder == 0` | Stichprobe | **kein Beweis**; deshalb `verify_sorted` (O(n), bricht beim ersten Fehler ab) |
| `duplicate_share` | gleiche Nachbarn, skaliert mit (n−1)/n | Stichprobe | nur bei sortiert wirkenden Daten; sonst `None` (`[charikar2000]`: Anzahl verschiedener Werte aus Stichprobe nicht zuverlässig schätzbar) |
| `ks_uniform` | Kolmogorov-Smirnov-Abstand zur Gleichverteilung auf [min, max] der Stichprobe | Stichprobe | 95-%-Grenzwert bei 1000 Werten 1,36/√1000 ≈ 0,043 |
| `analysis_time_ns` | gemessene Kosten der Analyse | | wird in der Bewertung mitgerechnet (A28) |

Abgeleitete Größen (`rules/decision.py`): `linear_scans` = q · gelesener
Anteil (½ bei gleichverteilten Treffern, 1 bei Fehlschlägen, 0,01 bei vorne
gehäuften Treffern, A33), `queries_per_n` = q/n, dazu log₁₀-Varianten für
den Baum.

**Gemessene Analysekosten** (`analysis_cost.csv`, Apple M5, Median):

| n | Stichprobe (1000) | vollständige Sortierprüfung | eine lineare Suche | eine binäre Suche |
|---|---|---|---|---|
| 10³ | 5,5 µs (= 46 lineare Suchen) | 0,33 µs | 120 ns | 8 ns |
| 10⁴ | 48,5 µs | 2,3 µs | 1,1 µs | 20 ns |
| 10⁶ | 43,6 µs | 230 µs (= 2,0 lineare Suchen) | 115 µs | 32 ns |
| 10⁷ | 62,1 µs | 2 290 µs | 1 231 µs | 66 ns |

Genauigkeit des geschätzten Sortiergrads (`analysis_sample.csv`, n = 10⁶,
wahrer Wert ≈ 0,02): RMSE 0,0144 bei Stichprobe 100, 0,0045 bei 1000,
0,0014 bei 10 000; Zeit 4,4 µs, 37,9 µs, 472 µs. Die Theorie √(p(1−p)/s)
passt (Diagramm `analysis_sample.pdf`).

---

## 5. Messmethodik

Jede Entscheidung der Messumgebung hat eine Quelle (`benchmark/timing.py`,
`statistics.py`, `environment.py`, A17 bis A24):

| Entscheidung | Begründung | Quelle |
|---|---|---|
| Blockmessung: ein Block von Anfragen ≥ 1 ms, Zeit durch Blocklänge teilen; Blocklänge durch Verdoppeln kalibriert | Einzelne Suche liegt unter der Timer-Auflösung | `[chen2016]` (Algorithmus 1: Anzahl Wiederholungen so wählen, dass der Timer-Fehler amortisiert wird) |
| Aufwärmläufe verwerfen (Kompilierung + 2 Läufe) | erster Aufruf kompiliert (Numba) | `[kalibera2013]`, `[georges2007]` |
| 30 Wiederholungen, jede mit neuem zufälligen Anfrageblock aus einem Pool von 10 000 | identische Eingaben lassen die CPU Sprungvorhersage und Cache „lernen“ (in M2 beobachtet: Analyse 95 → 33 µs, A17) | `[mytkowicz2009]` (Messbias; Setup Randomization) |
| zufällige Reihenfolge der Konfigurationen und Algorithmen mit protokolliertem Seed | Messbias durch Reihenfolge vermeiden | `[mytkowicz2009]` |
| Median mit 95-%-Bootstrap-Konfidenzintervall, Minimum als robuste Schätzung | Laufzeiten sind nicht normalverteilt, schief oder multimodal; Störungen verlängern nur | `[chen2016]` (Minimum als Schätzer), `[hoefler2015]` (Median und Konfidenzintervall berichten) |
| Speedups mit dem **geometrischen** Mittel zusammenfassen | arithmetisches Mittel normierter Zahlen hängt von der Basis ab | `[fleming1986]` (Regel 1: kein arithmetisches Mittel normierter Zahlen; Regel 2: geometrisches Mittel) |
| Umgebung (CPU, RAM, OS, Python, numpy, numba, pandas, Git-Commit, Stromquelle, Stromsparmodus) in jeder Ergebnisdatei | Reproduzierbarkeit | `[hoefler2015]` |
| Ein Faktor zur Zeit, Basis n = 10⁶, gleichverteilt, sortiert, Array, nur Treffer | Wechselwirkungen erst im Messraster (A23) | |

**Lehrstück aus dem Projekt (A35):** Das erste Messraster lief versehentlich
im Stromsparmodus; alle Zeiten waren gleichmäßig etwa doppelt so hoch
(lineare Suche n = 10⁶: 230 statt 113 µs). Seitdem protokolliert jede
Ergebnisdatei `power_source` und `low_power_mode`, und jedes Experiment
warnt vor dem Start. Die Wiederholung ohne Stromsparmodus stimmt mit M3
überein (116,6 ns gegenüber 117 ns bei n = 1000). Das ist ein konkretes
Beispiel für Messbias nach `[mytkowicz2009]` und gehört in den Bericht.

**Was `[chen2016]` genau sagt** (für die Begründung des Minimums und der
Blöcke): Störungen (OS-Jitter, Cache-Flushes, Kontextwechsel) „only delay
the completion of the original instructions, rather than speed them up“;
daraus folgt, dass das Minimum der beste Schätzer der ungestörten Laufzeit
ist und dass mehr Wiederholungen den Timer-Fehler amortisieren, aber
systematische Verzögerungen nicht zwingend beseitigen.

**Was `[fleming1986]` genau sagt**: Tabelle III/IV der Quelle zeigt drei
Maschinen, die je nach Normierungsbasis mal 25 % langsamer, mal schneller
erscheinen; nur das geometrische Mittel ist unabhängig von der Basis
(Beweis über die multiplikative Eigenschaft). Deshalb `size_speedup.csv`
mit geometrischem Mittel.

---

## 6. Das Empfehlungssystem

### 6.1 Rahmen: das Algorithmenauswahlproblem

`[rice1976]` definiert das Modell mit Problemraum, Merkmalsraum (Extraktion
f(x)), Algorithmenraum, Auswahlabbildung S und Leistungsmaß. Im Projekt:

- Problemraum: Datenmenge + Zugriffsmuster (eine „Instanz“).
- Merkmale: Abschnitt 4.
- Algorithmenraum = **Strategien** (`rules/strategies.py`): Suchverfahren
  samt Vorverarbeitung, z. B. `hash` = Hashtabelle aufbauen + Lookup,
  `sort_binary` = Sortieren + verzweigungsfreie binäre Suche,
  `list_hash` = Liste in Hashtabelle kopieren. 11 Strategien.
- Leistung: Gesamtzeit T(q) für q Anfragen (nur Zeit, kein Speicher, keine
  Bereichsanfragen, A26).

`[kerschke2019]` liefert die Bewertungsbegriffe: **Virtual Best Solver
(VBS)** = Oracle, das je Instanz das beste Verfahren wählt (untere
Schranke); **Single Best Solver (SBS)** = das eine Verfahren, das über alle
Instanzen am besten ist (obere Schranke für sinnvolle Selektoren); die
„Lücke“ zwischen SBS und VBS ist das, was ein Selektor schließen kann. Die
Quelle betont außerdem, dass Merkmale „informative and cheaply computable“
sein müssen und dass Merkmalskosten in die Bewertung gehören (Beispiel in
der Quelle: ein Selektor schloss „despite its non-negligible feature
computation costs“ über 75 % der Lücke). Genau deshalb berichtet das Projekt
alle Maße mit und ohne Analysekosten.

### 6.2 Das Regelwerk (`rules.yaml`, Version 2)

18 Entscheidungsknoten, 10 Blätter, Vorbild `[richter2015]`. Reihenfolge:
erst kostenlose Merkmale (Struktur, n, Nutzerangaben), die Stichprobe
zuletzt. Der Weg für Arrays:

```
Liste? ─nein─▶ n ≤ 16? ─ja─▶ ≥ 200 Anfragen? ─ja─▶ Hash │ nein ▶ linear
                 │
                nein
                 ▼
              n ≤ 100? ─ja─▶ ≥ 40 Anfragen? ─ja─▶ Hash │ nein ▶ Sortierung garantiert? ja ▶ binär (vzf.) │ nein ▶ linear
                 │
                nein
                 ▼
   erwartete Durchläufe der linearen Suche ≥ 20?
        ja ▶ Sortierung garantiert? ja ▶ q ≥ 0,2·n? ja ▶ Hash │ nein ▶ binär (vzf.)
                                   nein ▶ Stichprobe sortiert? ja ▶ q ≥ 0,2·n? (wie oben) │ nein ▶ Hash
        nein ▶ Durchläufe ≥ 1?
                ja ▶ Sortierung garantiert? ja ▶ binär │ nein ▶ Stichprobe sortiert? ja ▶ binär (nach Prüfung) │ nein ▶ linear
                nein ▶ Sortierung garantiert? ja ▶ binär │ nein ▶ linear
```

Für Listen: n ≤ 16 → ≥ 100 Anfragen? → Liste in Hashtabelle sonst linear;
sonst ≥ 2 Durchläufe → Hashtabelle; sonst Sortierung garantiert → Sprungsuche,
sonst linear.

**Herkunft der Schwellen** (Kopf von `rules.yaml`, A34, alle Apple M5):

| Schwelle | Messung |
|---|---|
| n ≤ 16 | lineare schneller als binäre bei n = 10 (Faktor 1,42), langsamer bei n = 30 (`size_speedup.csv`) |
| ≥ 20 Durchläufe für Hashtabelle | Aufbau kostet 7 bis 34 komplette Durchläufe der linearen Suche (n = 10³ … 10⁷, `amortization.csv`); Break-even 14 bis 68 Anfragen |
| ≥ 1 Durchlauf für Sortierprüfung | Prüfung kostet 2,0 bis 2,9 lineare Suchen mit gleichverteilten Treffern (`analysis_cost.csv`) |
| q ≥ 0,2·n | ab dort schlägt die Hashtabelle die verzweigungsfreie binäre Suche auch auf sortierten Daten |
| 200 / 40 / 100 Anfragen bei kleinem n | fester Grundaufwand jeder Vorverarbeitung (Hash-Aufbau ≈ 960 ns schon bei n = 10); Break-even aus `grid_components.csv`, **nur auf der Kalibrierungshälfte** geprüft (A30, A34) |
| Liste: ≥ 2 Durchläufe | Kopieren in Hashtabelle ≈ 1,6 Listendurchläufe |

Verteilung und Duplikate stehen bewusst **nicht** im Regelwerk: Sie ändern
die Wahl nicht, weil die Interpolationssuche auf dem Messlaptop nie gewinnt
(`distribution.csv`) und Duplikate die Zeiten kaum verändern
(`duplicates.csv`: binär 33,1 / 33,1 / 32,0 ns bei 0 / 10 / 50 %).

### 6.3 Der gelernte Baum (`rules/learned.py`, A31)

scikit-learn `DecisionTreeClassifier(criterion="entropy", max_depth=4,
min_samples_leaf=20)`; Entropie-Kriterium nach ID3 `[quinlan1986]`
(Informationsgewinn gain(A) = I(p, n) − E(A), Attribut mit maximalem Gewinn
wird Wurzel, rekursiv weiter). Begrenzte Tiefe für Interpretierbarkeit und
Entscheidung in konstanter Zeit `[vilasboas2021]` („a decision tree with
limited depth can recommend an algorithm in constant time“; Hauptvorteil
„interpretability“). Der Baum wird als reine Arrays in JSON gespeichert und
in der App ohne scikit-learn ausgewertet. Trainiert auf der
Kalibrierungshälfte (`results/full/learned_tree.txt` zeigt ihn: Wurzel
„Verkettete Liste ≤ 0,5“, dann log₁₀ erwartete Durchläufe, Duplikatanteil,
q/n, log₁₀ n).

### 6.4 Bewertung (`rules/evaluation.py`, `exp_recommender.py`)

Instanzen = jede Datenmenge des Rasters × 3 Zugriffsmuster × q ∈ {1, 3, 10,
30, 100, 300, 10³, 10⁴, 10⁵, 10⁶} × Sortiergarantie (ja/nein, nur wenn
wirklich sortiert). Kosten jeder Strategie aus dem linearen Kostenmodell.
Aufteilung der Datenmengen je (n, Struktur) zur Hälfte in Kalibrierung und
Test; Baum trainiert, SBS bestimmt und Regeln nachgeschärft **nur** auf der
Kalibrierungshälfte; alle Maße auf der Testhälfte (A30). Maße nach
`[kerschke2019]`, `[bischl2016]`:

- Trefferquote (Oracle-Strategie getroffen) und Anteil ≤ 1,1 × Optimum (A29:
  nahezu gleich schnelle Strategien werden sonst willkürlich getrennt).
- Slowdown = realisierte Kosten / Oracle-Kosten; geometrisches Mittel
  `[fleming1986]`, Median, 95-%-Quantil, Maximum.
- Lücke geschlossen = (SBS − Selektor)/(SBS − VBS), auf Summen und auf
  geometrischen Mitteln.
- Realisierte Kosten enthalten: Analysekosten, falls ein Stichprobenmerkmal
  gefragt wurde; Kosten einer fehlgeschlagenen Sortierprüfung plus
  Neuentscheidung; Ersatz einer Strategie für die falsche Struktur (nur beim
  Baum möglich).

---

## 7. Ergebnisse in Zahlen (Messlaptop Apple M5, `results/full/`)

### 7.1 Laufzeit nach Datengröße (`size.csv`, Median ns pro Anfrage, sortiert, gleichverteilt, nur Treffer)

| n | linear | binär | binär vzf. | Sprung | exponentiell | Fibonacci | Interpolation | Hash |
|---|---|---|---|---|---|---|---|---|
| 10 | **5,2** | 7,4 | 6,2 | 14,9 | 8,4 | 13,7 | 10,8 | 1,3 |
| 100 | 17,3 | 8,6 | **6,4** | 22,6 | 11,6 | 24,1 | 17,9 | 3,6 |
| 1 000 | 117,0 | 7,9 | **7,0** | 34,7 | 13,2 | 34,6 | 23,6 | 3,3 |
| 10⁴ | 1 096,8 | 18,5 | **10,0** | 66,6 | 18,0 | 42,9 | 31,9 | 2,3 |
| 10⁶ | 115 365,9 | 33,2 | **25,0** | 519,6 | 105,4 | 75,4 | 47,1 | 4,0 |

Vergleiche pro Anfrage bei n = 10⁶: linear 449 821, binär 20,9, binär vzf.
22, Interpolation 14,4, Sprung 942,8, Hash 1,3. Faktor Vergleiche
linear/binär ≈ 21 000, Faktor Zeit ≈ 3 500: sequenzielles Lesen ist für die
CPU billig und vorhersagbar (Folie 16).

Geometrisches Mittel der Speedups gegenüber der klassischen binären Suche
über alle 13 Größen (`size_speedup.csv`): binär vzf. 1,44; Hash 6,15;
exponentiell 0,59; Interpolation 0,59; Fibonacci 0,42; Sprung 0,18;
linear 0,013.

### 7.2 Verteilung (`distribution.csv`, n = 10⁶, ns)

| Verteilung | binär vzf. | Interpolation | Vergleiche Interpolation |
|---|---|---|---|
| gleich | 24,8 | 43,2 | 13,6 |
| normal | 25,7 | 601,7 | 158,5 |
| exponentiell | 24,8 | 1 258,5 | 367,5 |
| geclustert | 24,8 | 1 536,7 | 558,8 |
| lognormal | 25,8 | **113 740,9** | 123 302,4 |

Hash: 4,1 bis 4,3 ns, unabhängig von der Verteilung. Fazit: weniger
Vergleiche heißt nicht weniger Zeit (13,6 Vergleiche, aber 43 ns, weil jede
Schätzung Gleitkommarechnung kostet).

### 7.3 Zugriffsmuster (`access_pattern.csv`, n = 10⁶, ns)

| Muster | linear | binär vzf. | exponentiell | Hash |
|---|---|---|---|---|
| Treffer gleichverteilt | 109 490 | 25,9 | 104,2 | 4,2 |
| Treffer vorne gehäuft | 2 177 | 17,1 | 38,1 | 0,9 |
| nur Fehlschläge | 220 914 | 25,7 | 104,4 | 9,1 |

### 7.4 Speicherstruktur (`structure.csv`, n = 10⁶)

| | Zeit | Vergleiche | Zugriffe |
|---|---|---|---|
| lineare Suche, Array | 114 µs | 449 821 | 449 821 |
| Sprungsuche, Array | 0,54 µs | 943 | 943 |
| lineare Suche, Liste | **4,26 ms** | 449 821 | 449 821 |
| Sprungsuche, Liste | **4,34 ms** | 943 | 450 816 |

Die Liste ist 37-mal langsamer als das Array bei gleicher Zahl besuchter
Elemente (Zeigerverfolgung ohne Prefetching). Die Sprungsuche spart auf der
Liste 99,8 % der Vergleiche, aber keine Zeit, weil sie fast jeden Knoten
besuchen muss.

### 7.5 Amortisation (`amortization.csv`, `amortization_break_even.csv`, unsortierte Daten)

| n | Sortieren | Hash-Aufbau | linear/Anfrage | binär/Anfrage | Hash/Anfrage | Break-even Sortieren + binär | Break-even Hash |
|---|---|---|---|---|---|---|---|
| 10³ | 7,0 µs | 1,5 µs | 117 ns | 8,2 ns | 3,3 ns | 64 | 14 |
| 10⁴ | 210 µs | 17 µs | 1,1 µs | 19 ns | 2,3 ns | 192 | 15 |
| 10⁵ | 3,7 ms | 233 µs | 11,1 µs | 26 ns | 3,2 ns | 336 | 21 |
| 10⁶ | 43,8 ms | 5,2 ms | 113 µs | 32 ns | 4,2 ns | 387 | 46 |
| 10⁷ | 507 ms | 84 ms | 1,24 ms | 62 ns | 3,4 ns | 407 | 68 |

Hash-Aufbau in „Durchläufen der linearen Suche“ (ein Durchlauf = 2 × mittlere
Suchzeit mit gleichverteilten Treffern): 6,6 / 7,6 / 10,5 / 22,9 / 33,8 →
„7 bis 34“.

### 7.6 Bewertung des Empfehlungssystems (`recommender_summary.csv`, Testhälfte, 11 550 Instanzen)

| Selektor | Treffer | ≤ 1,1× | Slowdown geom. ohne / mit Analysekosten | Lücke geschlossen (geom.) ohne / mit | Stichprobe genutzt |
|---|---|---|---|---|---|
| Oracle (VBS) | 100 % | 100 % | 1,000 | 100 % | 0 % |
| SBS (Hashtabelle / Liste in Hashtabelle) | 52,1 % | 53,4 % | 3,699 | 0 % | 0 % |
| Immer lineare Suche | 35,7 % | 37,4 % | 9,321 | −70,7 % | 0 % |
| Immer binäre Suche (nur Arrays, 5 730) | 17,0 % | 19,8 % | 5,666 | −1,1 % | 0 % |
| **Regelwerk** | **87,4 %** | 90,0 % / 78,7 % | **1,078 / 1,236** | **94,2 % / 83,8 %** | 19,1 % |
| Entscheidungsbaum | 87,3 % | 90,3 % / 66,0 % | 1,108 / 2,492 | 92,2 % / 30,2 % | 31,3 % |

Nur Arrays mit Analysekosten: Regelwerk 80,2 % Lücke geschlossen, Baum
−3,6 % (schlechter als der SBS). Nur Listen: Regelwerk 90,5 %, Baum 93,0 %
(dort braucht keiner die Stichprobe).

Meta-Algorithmus `[vansandt2019]` auf sortierten Arrays mit Garantie
(`recommender_sampling.csv`, 1 590 Instanzen): Treffer 58,2 %, Slowdown
geom. 24,8; Regelwerk dort 80,3 % und 1,09. Bei wenigen Anfragen kosten die
Probeanfragen mehr als die Suche selbst.

Häufigste Fehler des Regelwerks (`recommender_choices.csv`): bei
Oracle = linear wählt es 393-mal Hash (kleine Datenmengen, Grenzfälle der
Break-even-Punkte), bei Oracle = list_linear 388-mal list_hash.

---

## 8. Yanniks Änderungen (Branch `messmethodik_test`)

Ein Commit (`2d052e2`, 3. November 2026, „Messdaten full_y/full_y2 und
test_experiments hinzugefuegt“), aufgesetzt auf `main`. Vorher hatte Yannik
`results/full_y2` direkt auf `main` gepusht und wieder gelöscht; der Branch
enthält beide Ordner.

### 8.1 Was er geändert hat

| Bereich | Änderung | Bewertung |
|---|---|---|
| `benchmark/experiment.py` | Option `--results-dir` für einen abweichenden Ausgabeordner (globaler Zustand `RESULTS_DIR`) | sinnvoll für Messungen auf mehreren Rechnern; einfache Umsetzung |
| `benchmark/timing.py` | `time_search` nimmt optionale `query_blocks`: **dieselben Anfrageblöcke je Wiederholung für alle Algorithmen** einer Konfiguration (gepaarter Vergleich) | methodisch gut: Unterschiede zwischen Algorithmen hängen nicht mehr an unterschiedlich gezogenen Blöcken; frische Blöcke zwischen Wiederholungen bleiben erhalten (A17) |
| `benchmark/grid.py` | viertes Zugriffsmuster `misses_only`; Tabelle `grid_operations.csv` (gezählte Vergleiche/Zugriffe je Datensatz, Muster, Algorithmus); exakter Duplikatanteil (`distinct_count`, `actual_duplicate_share`); `seed_replicate`; `query_id`/`query_seed` an jeder Komponente | gut: macht das Raster vollständig nachvollziehbar und liefert hardwareunabhängige Zählungen für den Bericht |
| `experiments/exp_grid.py` | Größen 10, 16, 30, 32, 100, 10³ … 10⁶ (9 statt 6); vier Fälle bei n = 1000 mit 3 Seeds (Replikate) → 818 Datensätze | gut: die neuen Größen liegen genau an den Schwellen des Regelwerks (16, 30/32), die Replikate geben erstmals eine Datensatzvarianz im Raster |
| `experiments/exp_amortization.py` | komplett neu: liest die Komponenten aus dem Raster (nur exakt passende Datensätze), rechnet Analysekosten in T(q) ein, verwendet die verzweigungsfreie binäre Suche, misst vollständige Läufe zur Validierung des additiven Modells, 2×2-Panel-Diagramm; n = 10⁷ entfällt | methodisch ein Fortschritt (eine Messbasis statt zwei, Modell wird geprüft). **Aber**: brach die Kompatibilität mit `results/full/` (altes Rasterformat ohne `seed_replicate`/`query_id`) und mit `--plot-only` auf alten Daten; die Zahlen der Präsentation stammen aus der alten Pipeline. Behoben, siehe 8.3 |
| `experiments/exp_size.py` | Größen 16 und 32 ergänzt | konsistent mit dem Raster; `results/full/size.csv` hat diese Punkte nicht (nur Intel) |
| `experiments/run_all.py` | Raster läuft jetzt **vor** der Amortisation | nötig wegen der Abhängigkeit |
| `experiments/exp_recommender.py` | `sys.stdout.reconfigure(encoding="utf-8")` | Windows-Konsolenfix, harmlos |
| `pyproject.toml` | `pythonpath = ["."]` für pytest | nötig, damit `tests/test_experiments.py` die Skripte importieren kann |
| `tests/test_experiments.py` (neu), `test_benchmark.py`, `test_data.py` | 10 neue Tests | gut, decken die neuen Pfade ab |
| `docs/annahmen.md` | A22 und A25 umgeschrieben, A41 neu | passt zum Code; A22 beschreibt jetzt die neue Pipeline, die alte ist nur noch in der Git-Historie beschrieben → deshalb A39 ergänzt |
| `results/full_y`, `results/full_y2` | zwei vollständige Läufe (Intel Core Ultra 5 245KF, Windows 11, Python 3.12, numba 0.68.1); je ~190 000 Zeilen CSV | wertvoll als Vergleichshardware; `git_commit: unknown` und `power_source: unknown` (Git nicht im Pfad, Stromprüfung nur unter macOS) |

### 8.2 Was seine Messungen zeigen (für den Bericht)

- **Die Schwellen des Regelwerks sind hardwareabhängig.** Auf dem Intel
  trifft das Regelwerk nur 74,0 % (full_y) bzw. 73,8 % (full_y2) der
  Oracle-Strategien (Messlaptop 87,4 %), der Baum dort 84,0 % / 82,7 %
  (trainiert auf der Intel-Kalibrierungshälfte). Mit Analysekosten schließt
  das Regelwerk trotzdem mehr der Lücke (73,3 % / 76,1 %) als der Baum
  (66,4 % / 69,7 %). Die Kernaussage „Regelwerk schlägt Baum, sobald
  Analysekosten zählen“ gilt also auch auf anderer Hardware; die absoluten
  Trefferquoten nicht.
- **Die Interpolationssuche gewinnt auf dem Intel bei Gleichverteilung**:
  n = 10⁶, uniform: 69,5 ns (full_y) / 78,0 ns (full_y2) gegenüber 128 /
  129 ns für die verzweigungsfreie binäre Suche. Auf dem Messlaptop verliert
  sie (43,2 gegenüber 24,8 ns). Die Aussage „Interpolationssuche schlägt die
  verzweigungsfreie binäre Suche in keiner Messung“ (Folie 24, Fazit) muss
  deshalb mit „auf dem Messlaptop“ eingeschränkt werden. Bei lognormal
  bleibt sie überall so langsam wie die lineare Suche (167 µs / 192 µs).
- **Exponentielle Suche** bleibt auch auf dem Intel bei vorne gehäuften
  Treffern langsamer als die verzweigungsfreie binäre Suche (87,9 gegenüber
  74,5 ns; 77,2 gegenüber 72,4 ns), aber der Abstand ist viel kleiner als
  auf dem Messlaptop (38 gegenüber 17 ns).
- **Die beiden Intel-Läufe streuen untereinander stark**: Break-even
  Sortieren + binäre Suche bei n = 10³ 90 gegenüber 176 Anfragen, Hash 23,9
  gegenüber 14,7; Replikate innerhalb eines Laufs streuen dagegen wenig
  (89 bis 99). Das spricht für wechselnde Last oder Taktung des Rechners
  zwischen den Läufen; `power_source`/`low_power_mode` sind dort
  „unknown“, weil die Prüfung nur unter macOS läuft.
- **Validierung des additiven Kostenmodells** (`amortization_validation.csv`,
  Verhältnis gemessene Gesamtzeit / Modell, Median): bei n = 10⁵ 0,92 bis
  1,42; bei n = 10³ und q = 1 jedoch 1,3 bis 2,25. Das Modell unterschätzt
  kleine Fälle, weil Blockmessungen den Aufruf-Overhead je Aufruf nicht
  enthalten (A21). Das ist ein ehrlicher Befund für „Grenzen der Arbeit“.
- Auf dem Intel ist die binäre Suche bei n = 10 mit 25 ns dreimal
  langsamer als die lineare (8 ns); auf dem Messlaptop 7,4 zu 5,2. Die
  Schwelle n ≤ 16 bleibt also in der Richtung richtig.

### 8.3 Was ich daran geändert habe

- `exp_amortization.py` akzeptiert jetzt auch das alte Rasterformat:
  fehlt `seed_replicate`, wird 0 angenommen; fehlen `query_id`/`query_seed`,
  entfällt der Filter (das alte Raster hat genau einen Anfragepool je
  Datensatz und Muster). Damit läuft die Auswertung auch auf
  `results/full/`. Prüfung: Aus dem M5-Raster ergeben sich Break-even-Punkte
  von 57 / 167 / 353 / 387 (Sortieren + binär) und 12 / 15 / 22 / 44 (Hash)
  für n = 10³ … 10⁶, gegenüber 64 / 192 / 336 / 387 und 14 / 15 / 21 / 46
  aus der alten Messung. Beide Pipelines stimmen also überein (A39).
- Sein Test `test_amortization_explains_when_grid_schema_is_stale` prüft
  jetzt ein wirklich kaputtes Schema (ohne `seed`); ein neuer Test deckt
  das alte Format ab.
- Alles andere übernommen. Die Ordner `full_y`/`full_y2` bleiben im
  Repository (README und A39 ordnen sie ein).

---

## 9. Befunde: Fehler und Inkonsistenzen

Sortiert nach Gewicht. Status: **behoben** (in dieser Durchsicht) oder
**offen** (Entscheidung des Teams).

| Nr. | Befund | Ort | Gewicht | Status |
|---|---|---|---|---|
| B1 | Yanniks neues `exp_amortization.py` konnte die Apple-M5-Daten in `results/full/` weder neu auswerten (`ValueError: Grid features lack ['seed_replicate']`) noch mit `--plot-only` zeichnen (`amortization_summary.csv` fehlt). Die Präsentationszahlen wären damit nicht mehr reproduzierbar gewesen. | `experiments/exp_amortization.py` | hoch | behoben (Abschnitt 8.3) |
| B2 | Die Aussage „Interpolationssuche schlägt die verzweigungsfreie binäre Suche in keiner Messung“ gilt nur für den Messlaptop; auf Yanniks Intel gewinnt sie bei Gleichverteilung (Abschnitt 8.2). Folie 24 und Fazit-Punkt 2 brauchen die Einschränkung. | Präsentation, `rules.yaml`-Kommentar | hoch (inhaltlich) | dokumentiert (A39); Folie anpassen: offen |
| B3 | `docs/annahmen.md` sprang von A38 auf A40 (A39 fehlte). | `docs/annahmen.md` | mittel | behoben: A39 beschreibt jetzt die zwei Amortisationspipelines und die Intel-Läufe |
| B4 | Docstring von `exp_recommender.py` sagte „tree trained on all instances (used by the app)“, der Code trainiert den exportierten Baum auf der Kalibrierungshälfte (A30, A38). | `experiments/exp_recommender.py` | mittel (Doku) | behoben |
| B5 | `[charikar2000]` wird dreimal zitiert (`features/extraction.py`, `analysis_page.py`, A16), liegt aber **nicht** im Drive-Ordner `Quellen`. Entweder Quelle nachliefern oder Zitat durch eine vorhandene ersetzen. | Quellenordner | mittel | offen |
| B6 | „Zufälliger Fehlschlag“ auf Seite 2 prüfte nur die ersten 2 Mio. Werte; bei n = 3 Mio. oder 10 Mio. konnte der „Fehlschlag“ ein Treffer sein. | `app/views/workload_page.py` | mittel | behoben (`np.isin` über alle Werte, Test ergänzt) |
| B7 | `Workload` validierte `front_mean` und `hit_ratio` nicht; `front_mean = 0` hätte `log10(0)` in `selection_features` ausgelöst. | `rules/strategies.py` | niedrig | behoben |
| B8 | README-Tabelle nannte für die Interpolationssuche nur zwei Quellen und „O(log log n) erwartet“, die Registry vier Quellen und den O(n)-Worst-Case. | `README.md` | niedrig | behoben |
| B9 | `recommendation_page._value` duplizierte `viz/graph._format_value`. | App | niedrig | behoben (`format_value` öffentlich, einmal definiert) |
| B10 | Ergebnisdatei `results/full/size.csv` trägt `git_commit: 687d13e (uncommitted changes)`; die M3-Messreihe lief also mit nicht committetem Code. Seit PR #9 zählt nur noch Code (`src`, `experiments`, …) als „dirty“. Für den Bericht: als Grenze nennen oder die Messreihe mit dem committeten Stand wiederholen. | `results/full/*.csv` | niedrig | offen |
| B11 | Yanniks Ergebnisdateien tragen `git_commit: unknown` und `power_source: unknown` (Windows: kein Git im Pfad, Stromprüfung nur unter macOS). | `results/full_y*/` | niedrig | offen (Hinweis in A39); `environment.py` könnte unter Windows `powercfg` abfragen |
| B12 | `experiments/exp_size_datasets.py` steht nicht in `run_all.py` (gewollt, dokumentiert) und importiert `from exp_size import …`, läuft also nur als Skript aus seinem Ordner. | `experiments/` | niedrig | offen, kein Handlungsbedarf |
| B13 | Streamlit 1.66 warnt: `use_container_width` wird nach 2025-12-31 entfernt (`width="stretch"`); `requirements.txt` erlaubt ab 1.40, wo das neue Argument fehlt. | App | niedrig | offen (beim nächsten Streamlit-Update umstellen) |
| B14 | `benchmark/experiment.py` leitet `PROJECT_ROOT` über `parents[3]` der Datei ab; das funktioniert nur mit der Editable-Installation (`-e .`), die `requirements.txt` vorsieht. | | niedrig | offen (Hinweis genügt) |
| B15 | `test_comparison_mode` klickt `app.button[0]` und verlässt sich auf die Reihenfolge der Buttons. | `tests/test_app.py` | niedrig | offen |
| B16 | Folie 11 enthält den Hinweis „Zuordnung der beiden Zeiten prüfen“: Laut `structure.csv` ist die **Sprungsuche** auf der Liste mit 4,34 ms die langsamere, die lineare Suche 4,26 ms (Abschnitt 7.4). Die Folie ordnet richtig zu. | Präsentation | | geklärt |

Keine Korrektheitsfehler in den Algorithmen gefunden: Alle 10 Verfahren
werden gegen `np.searchsorted` bzw. das erste Vorkommen getestet (beide
Varianten, Randfälle n = 0 und 1, Duplikate, Extremwerte des int64), die
Schrittfolgen des Lernmodus enden nachweislich mit demselben Ergebnis wie
die kompilierten Verfahren, und das Regelwerk wird beim Laden auf
Zyklen und unbekannte Ziele geprüft.

---

## 10. Änderungen in dieser Durchsicht

**Neu**

- `src/recommender/viz/race.py`: baut aus den Schrittfolgen
  (`viz/stepper.py`) ein eigenständiges HTML-Dokument mit animiertem
  Wettrennen: je Verfahren eine Bahn mit demselben Array, pro Takt ein
  Schritt, Start/Pause, Einzelschritt vor/zurück, Neustart, Tempo,
  Schrittzähler, Abzeichen „gefunden nach k Schritten“ bzw. „nicht
  vorhanden nach k Schritten“. Keine externen Ressourcen. Exakte Vergleiche
  und Zugriffe kommen aus der kompilierten Zählvariante.
- `app/views/race_view.py`: Streamlit-Block aus Rennen plus Rangtabelle
  (Platz, Schritte, Vergleiche, Zugriffe, Ergebnis, Komplexität);
  Zuordnung Strategie → darstellbares Verfahren.
- `tests/test_race.py` (7 Tests), 4 neue App-Tests.
- `docs/review_und_lernmaterial.md` (diese Datei).

**Geändert**

- `app/views/learning_page.py`: Ansicht „Schritt für Schritt“ (wie bisher)
  oder „Wettrennen“ (Mehrfachauswahl der Verfahren, Standard lineare gegen
  binäre Suche). Gleiche Array- und Schlüsselsteuerung für beide.
- `app/views/recommendation_page.py`: Steckbrief der empfohlenen Strategie
  (Verfahren, Komplexität, Vorverarbeitung, Voraussetzung, Struktur,
  Quellen), „Warum diese Strategie?“ als nummerierte Fragenfolge mit
  Antworten und Merkmalswerten, Kostenmodell mit eingesetztem q, und das
  Wettrennen „lineare Suche gegen das empfohlene Verfahren“ auf 24 zufällig
  gezogenen, sortierten, auf 0–999 skalierten Werten der aktuellen Daten.
  Hash-Lookup wird erklärt statt gezeichnet.
- `app/views/workload_page.py`: B6.
- `src/recommender/rules/strategies.py`: B7.
- `src/recommender/viz/graph.py`: B9.
- `experiments/exp_amortization.py`, `tests/test_experiments.py`: B1.
- `experiments/exp_recommender.py`: B4.
- `README.md`: Seiten 4 und 6, Quellen der Interpolationssuche, Tabelle der
  Ergebnisordner, Verweis auf diese Datei.
- `docs/annahmen.md`: A38 (Wettrennen), A39 (neu).

**Geprüft**: 523 Tests bestanden (pytest), App im Browser gestartet,
Lernmodus-Wettrennen und Empfehlungsseite mit Screenshots kontrolliert
(Start, nach 6 Takten, Ende; Rangtabelle; Steckbrief).

---

## 11. Abgleich der Präsentation mit den Messdaten

Jede Zahl der Folien (`Empfehlungssystem.pptx`, Stand 03.11.26) gegen
`results/full/` geprüft. ✓ = stimmt, ✗ = anpassen.

| Folie | Aussage | Prüfung |
|---|---|---|
| 7 | n = 10: sequentiell 5,2 ns, binär 7,4 ns | ✓ `size.csv` |
| 10 | n = 10⁶: sequentiell ≈ 3 500-mal langsamer als binär | ✓ 115 366 / 33,2 = 3 475 |
| 11 | Sprungsuche: 940 statt 450 000 Vergleiche; 4,34 gegen 4,26 ms | ✓ 942,8 / 449 821; Liste: Sprung 4,344 ms, linear 4,264 ms. Hinweis „Zuordnung prüfen“ erledigt: Sprungsuche ist die 4,34 |
| 12 | exponentiell 38 ns, verzweigungsfrei binär 17 ns bei vorne gehäuften Treffern | ✓ 38,1 / 17,1 `access_pattern.csv` |
| 13 | Methodik: Numba, zwei Varianten, Blöcke ≥ 1 ms, Aufwärmläufe, 30 Wiederholungen, Vorrat 10 000, ein Faktor zur Zeit, geometrisches Mittel | ✓ `timing.py`, `experiment.py`, A19 bis A24 |
| 14 | Grenzen: eine Maschine, Stromsparmodus-Vorfall, ein Datensatz, warmer Cache | ✓ A18, A35, A40. Ergänzen: seit PR #11 gibt es `exp_size_datasets.py` mit 5 Datensätzen (A40), seit Yannik eine zweite Maschine (A39) |
| 15 | Tabelle Median ns (10 / 100 / 1 000 / 10⁴ / 10⁶) | ✓ alle 20 Werte stimmen mit `size.csv` |
| 16 | 450 000 gegen 21 Vergleiche, Faktor 21 000 / 3 500 | ✓ |
| 17 | Lower-Bound-Variante, [l, r), O(log n), O(1) Speicher, ⌈log₂ n⌉ Runden verzweigungsfrei | ✓ `binary.py` (verzweigungsfrei: `while length > 1`, Runden = ⌈log₂ n⌉) |
| 19 | Hash-Aufbau 7 bis 34 Durchläufe; Break-even 14 bis 68 (Hash), 64 bis 407 (Sortieren + binär) | ✓ `amortization_break_even.csv`, Abschnitt 7.5 |
| 20 | Interpolationsformel p = l + ⌊(k − aₗ)/(aᵣ − aₗ)·(r − l)⌋, Peterson 1957 | ✓ `interpolation.py` (mit hi−1 statt r, halboffenes Intervall) |
| 21 | 1000 Werte aᵢ = 10·i, gesucht 4230 → p = 423; Ausreißer 10⁶ → p = 4, 242 Schätzungen ohne Guard; ≈ 4,3 Zugriffe bei n = 10⁶ | Rechenbeispiel der Folie, nicht aus den Messdaten; log log₂ 10⁶ = log₂ 19,9 = 4,3 ✓ (`[gonnet1980]`: lg lg n + O(1)) |
| 22 | Guard 8 nach Van Sandt 2019, getestet 0, 8, 16 | Guard 8 ✓ (A5). **„getestet wurden 0, 8 und 16“ findet sich weder im Code noch in `annahmen.md`** (A5 sagt ausdrücklich „von uns nicht neu kalibriert“). Entweder Aussage streichen oder die Messung nachholen ✗ |
| 23 | Hashing: h(k,i) = (h₀(k)+i) mod m, A = ⌊2⁶⁴/φ⌋, m kleinste Zweierpotenz ≥ 2n, α ≤ 0,5; 2,5 Sondierungen bei α = 0,5 erfolglos; Peterson 1,541 / 5,526 | ✓ `hashing.py`; Peterson-Werte stehen so in `[peterson1957]`. Die Kurve „Sondierungen bei erfolgloser Suche“ (1 … 13) ist die Formel ½(1 + 1/(1−α)²) (Knuth-Formel für lineares Sondieren; nicht im Quellenordner belegt, im Projekt nicht gemessen) – als „Theorie“ kennzeichnen ✗ |
| 24 | Interpolation 13,6 statt 22 Vergleiche; 43,2 gegen 24,8 ns; lognormal 113 740,9 ns; Hash 4,1 bis 4,3 ns | ✓ `distribution.csv`. **Einschränkung „auf dem Messlaptop“ ergänzen** (B2) ✗ |
| 27 | Stichprobe ≈ 5,5 µs bei n = 10³ = 46 sequentielle Suchen; Sortierprüfung ≈ 230 µs bei n = 10⁶ | ✓ `analysis_cost.csv` (5 541 / 120 = 46,2; 229 604) |
| 28 | Beispieldatensatz 10⁵, lognormal, sortiert, q = 100: 7 Fragen → verzweigungsfreie binäre Suche nach Sortierprüfung; Baum nach 4 Fragen dasselbe | ✓ in der App nachvollzogen (Screenshot: 7 Fragen, Durchläufe = 50, q/n = 0,001); 18 Knoten, 10 Blätter ✓ |
| 29 | Tabelle Testhälfte 11 550: Regelwerk 87,4 %, 1,078/1,236, 94,2 %/83,8 %; Baum 87,3 %, 1,108/2,492, 92,2 %/30,2 %; Baum Stichprobe 31,3 %, Regelwerk 19,1 %; Arrays Baum −3,6 % | ✓ `recommender_summary.csv` |
| 30 | Meta-Algorithmus 1 590 Fälle: 58,2 %, Slowdown 24,8; Regelwerk 80,3 %, 1,09 | ✓ `recommender_sampling.csv` |
| 33 | Fazit 2 „Interpolationssuche und exponentielle Suche gewinnen in der Praxis nie“ | auf dem Messlaptop ✓; auf dem Intel gewinnt die Interpolationssuche bei Gleichverteilung (B2) ✗ |
| 34 | Grenzen: ein Rechner, Raster bis 10⁶, q über Kostenmodell, nur Zeit | ✓ A18, A25, A22, A26; ergänzen: Modell bei kleinem n/q bis Faktor 2 zu optimistisch (A39) |
| 36 | Quellenliste | `Kotthoff 2012`, `Smith-Miles 2008`: im Ordner liegen `Kotthoff (2014).pdf` und `smith-miles2009.pdf`; Jahreszahlen abgleichen ✗ |

---

## 12. Quellen: Zuordnung zu den Aussagen des Projekts

Alle Dateien im Drive-Ordner `Quellen` und wofür das Projekt sie nutzt
(Zahl = Anzahl Zitate im Code/YAML/Markdown):

| Datei | Schlüssel | Genutzt für |
|---|---|---|
| vansandt2019.pdf | `[vansandt2019]` (28) | Guard der Interpolationssuche (Alg. 2, Z. 14–17; Größe 8, Anhang A.3); Lognormal µ = 0, σ = 2; Meta-Algorithmus mit Probeanfragen; lineare Suche schnellste auf kleinen Arrays; SIP/TIP als Abgrenzung (A5) |
| richter2015.pdf | `[richter2015]` (15) | Hashtabelle: multiplikatives Hashing + lineares Sondieren, Füllgrad; Entscheidungsgraph als Vorbild des Regelwerks |
| kerschke2019.pdf | `[kerschke2019]` (11) | Algorithmenauswahl: VBS/SBS, geschlossene Lücke, billige Merkmale, Merkmalskosten; Reihenfolge der Fragen im Regelwerk |
| Shneiderman1978Jump.pdf | `[shneiderman1978]` (9) | Sprungsuche, √n, Einsatz auf verketteten Dateien |
| khuong2017.pdf | `[khuong2017]` (9) | verzweigungsfreie binäre Suche, Conditional Moves, Branch Misprediction |
| peterson1957.pdf | `[peterson1957]` (7) | Interpolationsidee; offene Adressierung; Suchlängen 1,541 / 5,526 |
| mytkowicz2009.pdf | `[mytkowicz2009]` (5) | Messbias, Setup-Randomisierung, feste Seeds, wechselnde Eingaben |
| fleming1986.pdf | `[fleming1986]` (5) | geometrisches Mittel für Speedups und Slowdowns |
| chen2016.pdf | `[chen2016]` (5) | Blockmessung, Minimum als Schätzer, nicht-normale Verteilungen |
| ferguson1960.pdf | `[ferguson1960]` (4) | Fibonacci-Suche |
| bentley1976.pdf | `[bentley1976]` (4) | exponentielle Suche (unbounded search) |
| quinlan1986.pdf | `[quinlan1986]` (3) | ID3, Entropie-Kriterium des Baums |
| p550-perl.pdf | `[perl1978]` (3) | log log N der Interpolationssuche; Umschalten auf sequenzielle Suche |
| kalibera2013.pdf | `[kalibera2013]` (3) | Aufwärmläufe, Wiederholungen |
| gonnet1980.pdf | `[gonnet1980]` (3) | lg lg n + O(1); Abweichung bei nicht gleichverteilten Daten |
| georges2007.pdf | `[georges2007]` (3) | Konfidenzintervalle statt „best of 30“, Aufwärmen |
| vilasboas2021.pdf | `[vilasboas2021]` (2) | Entscheidungsbäume begrenzter Tiefe für die Algorithmenauswahl, Interpretierbarkeit, konstante Entscheidungszeit |
| rice1976.pdf | `[rice1976]` (2) | Modell des Algorithmenauswahlproblems |
| hundhausen2002.pdf | `[hundhausen2002]` (2) | Lernmodus: Visualisierung hilft, wenn Lernende aktiv sind (Klick pro Schritt) |
| hoefler2015.pdf | `[hoefler2015]` (2) | Umgebung protokollieren, Median/CI berichten, keine Verhältnisse ohne Absolutwerte |
| li2004.pdf | `[li2004]` (1) | Laufzeitstatistiken aus Stichproben (dynamisch abgestimmte Sortierbibliothek) |
| bischl2016.pdf | `[bischl2016]` (1) | Bewertungsmaße der Algorithmenauswahl (ASlib) |
| Kotthoff (2014).pdf, smith-miles2009.pdf, Xu2008.pdf | in der Präsentation, nicht im Code | Übersichten zur Algorithmenauswahl (Folie 36) |
| kraska2018.pdf, marcus2020.pdf | nicht zitiert | gelernte Indexstrukturen (möglicher Ausblick) |
| pagh2004.pdf | in der Präsentation (Folie 36) | Cuckoo-Hashing (in A7 als nicht verglichen genannt) |
| bayer1972.pdf, comer1979.pdf | nicht zitiert | B-Bäume (Abgrenzung: Indexstrukturen außerhalb des Umfangs) |
| mehlhorn1993.pdf, mehlhorn2008_kap2.pdf, ottmann2017.pdf, 10.1515_9783111327471.pdf | nicht zitiert | Lehrbuchgrundlagen (Datenstrukturen/Algorithmen); für Kapitel „Grundlagen“ des Berichts |
| harris2020.pdf, hunter2007.pdf | nicht zitiert | NumPy bzw. Matplotlib (Werkzeugzitate) |
| **fehlt:** charikar2000 | `[charikar2000]` (3) | Anzahl verschiedener Werte aus Stichprobe nicht zuverlässig schätzbar (A16) → B5 |

---

## 13. Lernmaterial

### 13.1 Glossar

- **Untere Schranke (lower bound)**: kleinster Index i mit a[i] ≥ key. Auf
  sortierten Daten ist das die Position des ersten Vorkommens, falls der
  Schlüssel vorkommt. Alle Verfahren des Projekts auf sortierten Daten
  suchen sie und prüfen danach einmal auf Gleichheit.
- **Halboffenes Intervall [lo, hi)**: Kandidatenbereich; lo gehört dazu, hi
  nicht. Invariante der binären Suche: die untere Schranke liegt in [lo, hi].
- **Verzweigungsfrei (branchless)**: kein datenabhängiger bedingter Sprung in
  der Schleife; die Entscheidung wird gerechnet (`half * (a[...] < key)`).
  Vermeidet Pipeline-Leerung bei falscher Sprungvorhersage `[khuong2017]`.
- **Guard**: Schaltet die Interpolationssuche auf sequenzielle Suche um,
  wenn die Schätzung nahe am Intervallrand liegt (≤ 8 Elemente). Verhindert
  lange Folgen kleiner Fortschritte, nicht den Worst Case O(n) (A5).
- **Offene Adressierung / lineares Sondieren**: Kollision → nächster Slot
  (slot + 1) mod m. **Füllgrad α** = belegte / Slots; hier ≤ 0,5.
- **Multiplikatives (Fibonacci-)Hashing**: h(k) = obere b Bits von
  k · ⌊2⁶⁴/φ⌋ mod 2⁶⁴.
- **Knotenpool**: Liste als zwei Arrays (Werte, Nachfolger-Slot); logische
  Reihenfolge in zufälligen Slots → Zeigerverfolgung ohne Prefetching.
- **Stichprobenmerkmal**: Sortiergrad, Duplikatanteil, KS-Abstand; kostet
  Analysezeit, die mitgerechnet wird. **Kostenloses Merkmal**: n, Struktur,
  Nutzerangaben.
- **Erwartete Durchläufe der linearen Suche**: q · (gelesener Anteil pro
  Suche). Steuergröße der Amortisationsregeln (A33).
- **Strategie**: Suchverfahren + Vorverarbeitung (+ Sortierprüfung, wenn
  nötig). **Oracle / VBS**: beste Strategie je Instanz. **SBS**: beste feste
  Strategie. **Slowdown**: Kosten / Oracle-Kosten. **Lücke geschlossen**:
  (SBS − Selektor)/(SBS − VBS).
- **Kalibrierungs-/Testhälfte**: Datensätze je (n, Struktur) halbiert;
  Lernen und Nachschärfen nur auf der einen, Bewerten nur auf der anderen
  Hälfte (A30).
- **Blockmessung**: viele Anfragen in einer kompilierten Schleife, Zeit
  durch Anzahl; Block ≥ 1 ms.
- **Bootstrap-Konfidenzintervall**: Median vieler Resamples (2000) der 30
  Messwerte, 2,5-%- und 97,5-%-Quantil.
- **Geometrisches Mittel**: n-te Wurzel des Produkts; einziges Mittel, das
  unabhängig von der Normierungsbasis ist `[fleming1986]`.

### 13.2 Fragen und Antworten (Prüfungs- und Lerntool-Stoff)

1. *Warum ist die lineare Suche bei n = 10 schneller als die binäre, obwohl
   O(n) > O(log n)?* O-Notation versteckt konstante Faktoren. Bei 10
   Elementen passt alles in eine Cache-Zeile, die Schleife ist vorhersagbar
   und vektorisierbar; die binäre Suche zahlt pro Schritt einen
   datenabhängigen Sprung. Messung: 5,2 gegenüber 7,4 ns (`size.csv`).
   Deshalb Regel „n ≤ 16 → linear“.
2. *Warum braucht die binäre Suche bei n = 10⁶ 21 000-mal weniger
   Vergleiche, aber nur 3 500-mal weniger Zeit?* Sequentielles Lesen ist
   billig (Prefetching, Vektorisierung), die binäre Suche springt im Speicher
   und erleidet Cache-Misses und Fehlvorhersagen.
3. *Warum liefert die Sprungsuche auf Listen weniger Vergleiche, aber keine
   Zeitersparnis?* Sie muss jeden Knoten besuchen, um weiterzukommen
   (Zeiger folgen); gespart werden nur Vergleiche (943 statt 449 821), die
   Knotenbesuche (450 816) bleiben. Zeit 4,34 gegenüber 4,26 ms.
4. *Wann lohnt sich eine Hashtabelle?* Wenn q · Zeit(linear) − q · Zeit(hash)
   > Aufbaukosten, also q > t_build/(t_linear − t_hash). Gemessen: 14 bis 68
   Anfragen (n = 10³ bis 10⁷), das entspricht 7 bis 34 kompletten
   Durchläufen. Bei n ≤ 16 wegen des festen Grundaufwands erst ab ≈ 200
   Anfragen.
5. *Warum steht die Stichprobe im Regelwerk zuletzt?* Sie kostet bei
   n = 10³ etwa 5,5 µs, so viel wie 46 lineare Suchen. Merkmale müssen
   billig sein `[kerschke2019]`; bei n ≤ 100 wird sie gar nicht gezogen.
6. *Warum reicht die Stichprobe nicht, um „sortiert“ zu entscheiden?* Ein
   einziges vertauschtes Paar macht die binäre Suche falsch; die Stichprobe
   kann Sortierung nicht beweisen. Daher vollständige O(n)-Prüfung (bricht
   beim ersten Fehler ab) vor jedem Verfahren für sortierte Daten, sofern
   der Nutzer die Sortierung nicht garantiert.
7. *Was passiert, wenn die Stichprobe die Unordnung übersieht?* Die
   vollständige Prüfung schlägt fehl, ihre (Abbruch-)Kosten werden bezahlt,
   und es wird mit dem Wissen „unsortiert“ neu entschieden (A27,
   `evaluation.realised`, Seite 5 der App zeigt die Warnung).
8. *Warum ist die Interpolationssuche bei lognormalen Daten so langsam?*
   Ihre Schätzung setzt lineares Wachstum der Werte mit dem Index voraus;
   bei schiefen Daten liegt die Schätzung weit daneben, die Suche macht
   viele kleine Schritte (123 302 Vergleiche bei n = 10⁶). `[gonnet1980]`:
   bei nicht gleichverteilten Daten „departs significantly from the lg lg n
   behavior“.
9. *Warum gewinnt die Interpolationssuche selbst bei Gleichverteilung nicht
   (auf dem Messlaptop)?* 13,6 statt 22 Vergleiche, aber jede Schätzung
   kostet Gleitkomma-Division und -Multiplikation: 43 ns gegenüber 25 ns. Auf
   dem Intel-Rechner ist das Verhältnis umgekehrt (70 gegenüber 128 ns):
   hardwareabhängig.
10. *Warum zählt das Projekt Vergleiche und Zugriffe zusätzlich zur Zeit?*
    Zeiten gelten nur für die Messplattform (A18); gezählte Operationen sind
    hardwareunabhängig und erklären, wo Zeit entsteht.
11. *Was ist der Single Best Solver im Projekt?* Je Struktur die immer
    anwendbare Strategie mit dem besten geometrischen Slowdown auf der
    Kalibrierungshälfte: Hashtabelle (Arrays) und Liste in Hashtabelle. Sie
    trifft 52 % der Instanzen; das Regelwerk schließt 94 % der Lücke zum
    Oracle (ohne Analysekosten).
12. *Warum fällt der Entscheidungsbaum mit Analysekosten so stark ab?* Er
    fragt in 31 % der Fälle ein Stichprobenmerkmal (Duplikatanteil als
    zweite Ebene), das Regelwerk nur in 19 %, und der Baum wurde nicht
    kostensensitiv trainiert: ein teurer Fehler zählt im Training wie ein
    billiger (A31). Auf Arrays ist er mit Analysekosten schlechter als der
    SBS (−3,6 %).
13. *Warum verliert der Meta-Algorithmus von Van Sandt?* Er probiert jedes
    Verfahren mit 10 Anfragen; bei q = 1 oder 3 kosten die Proben mehr als
    die eigentliche Suche: Slowdown 24,8 gegenüber 1,09 für das Regelwerk.
14. *Was bedeutet „Lücke geschlossen 83,8 %“?* Von dem Abstand zwischen SBS
    (Slowdown 3,699) und Oracle (1,0) hat das Regelwerk mit Analysekosten
    (1,236) auf der log-Skala 83,8 % beseitigt.
15. *Warum geometrisches Mittel?* Bei normierten Zahlen (Speedup, Slowdown)
    hängt das arithmetische Mittel von der Bezugsbasis ab; Fleming und
    Wallace zeigen Beispiele, in denen dieselben Daten je nach Basis ein
    gegenteiliges Ergebnis liefern.
16. *Warum frische Anfragen in jeder Wiederholung?* Mit denselben Eingaben
    stellt sich die CPU (Sprungvorhersage, Cache) ein; die Analyse sank so
    von 95 auf 33 µs (A17). Das wäre Messbias `[mytkowicz2009]`.
17. *Warum stabiles argsort in der App statt np.sort?* Nur so lässt sich
    der Index im sortierten Array auf die Originalposition zurückrechnen,
    und Stabilität erhält das erste Vorkommen (A37).
18. *Wie ist der Duplikatanteil definiert und warum ist er bei unsortierten
    Daten nicht schätzbar?* p = (n − verschiedene)/n. Auf sortierten Daten
    sind Duplikate Nachbarn → aus Nachbarpaaren schätzbar; auf unsortierten
    nicht, und die Anzahl verschiedener Werte lässt sich aus einer
    Stichprobe nicht zuverlässig schätzen (`[charikar2000]`).
19. *Was zeigt das Wettrennen in der App?* Mehrere Verfahren auf demselben
    Array, ein Schritt pro Takt: die lineare Suche braucht bis zu n Schritte,
    die binäre etwa log₂ n, die Interpolationssuche bei gleichmäßigen Werten
    noch weniger. Die Tabelle darunter nennt die exakten Vergleiche und
    Zugriffe der kompilierten Zählvariante.
20. *Welche Grenzen hat die Arbeit?* Ein Messlaptop (Zeiten nicht
    übertragbar, Intel-Läufe zeigen andere Rangfolgen); Raster bis 10⁶; q
    über ein additives Kostenmodell, das bei kleinem n/q bis Faktor 2 zu
    optimistisch ist; nur Zeit als Leistungsmaß (kein Speicher, keine
    Bereichsanfragen); nur int64; Guard-Größe nicht neu kalibriert.

### 13.3 Rechenbeispiele

- **Binäre Suche, n = 24, Schlüssel am Index 17** (Standardbeispiel des
  Wettrennens, Seed 4): Prüfpositionen 12 → 18 → 15 → 17 → 16 (untere
  Schranke gefunden), danach Gleichheitsprüfung an Index 17: 6 Schritte;
  lineare Suche 18 Schritte;
  Interpolationssuche 9 Schritte. ⌈log₂ 24⌉ = 5 Halbierungen + 1 Prüfung.
- **Break-even Hashtabelle, n = 10³** (`amortization.csv`): t_build =
  1 541 ns, t_linear = 117 ns, t_hash = 3,3 ns → q* = 1541/(117 − 3,3) =
  13,55 → ab 14 Anfragen.
- **Durchläufe**: Hash-Aufbau bei n = 10⁷ = 84 ms; eine lineare Suche mit
  gleichverteilten Treffern liest im Mittel die Hälfte (1,24 ms), ein
  kompletter Durchlauf also 2,49 ms → 84/2,49 = 33,8 Durchläufe.
- **Erwartete Durchläufe im Regelwerk**: q = 100, Trefferquote 100 %,
  Treffer gleichverteilt → 100 · ½ = 50 Durchläufe ≥ 20 → Frage nach
  Sortierung (Folie 28, Screenshot der App: „Erwartete Durchläufe = 50“).
  Mit Trefferquote 50 %: 100 · (0,5·½ + 0,5·1) = 75.
- **KS-Grenzwert**: 1,36/√1000 = 0,043; gemessene KS-Abstände der
  Verteilungen stehen als Beschriftung in `distribution.pdf`.
- **Interpolationsschätzung** (Folie 21): a_i = 10·i, i = 0 … 999, Schlüssel
  4230: p = 0 + (4230 − 0)/(9990 − 0) · 999 = 423 ✓. Mit a_999 = 10⁶:
  p = 4230/10⁶ · 999 = 4,2 → 4.
- **Hashslot**: Kapazität m = kleinste Zweierpotenz ≥ 2n; n = 1000 → m =
  2048, b = 11 Bits, Shift 53; α = 1000/2048 = 0,49.

### 13.4 Typische Fehlvorstellungen (und die Korrektur aus dem Projekt)

- „O(log log n) ist immer schneller als O(log n).“ Nein: Konstanten,
  Rechenkosten pro Schritt und Datenverteilung entscheiden (Abschnitt 7.2).
- „Weniger Vergleiche = weniger Zeit.“ Nein: Sprungsuche auf Listen,
  Interpolationssuche.
- „Eine Hashtabelle ist immer die beste Wahl.“ Nur bei genug Anfragen
  (Aufbau 7 bis 34 Durchläufe), ohne Bereichsanfragen, ohne
  Speicherbetrachtung (A26).
- „Die Stichprobe sagt, ob die Daten sortiert sind.“ Sie sagt nur, ob sie
  sortiert *wirken*; der Beweis kostet O(n).
- „Ein gelernter Baum ist dem Regelwerk überlegen.“ Ohne Analysekosten
  gleichauf, mit Analysekosten deutlich schlechter, weil er die Kosten der
  Merkmale nicht kennt.
- „Mehr Wiederholungen beseitigen alle Störungen.“ `[chen2016]`: sie
  amortisieren den Timer-Fehler, nicht systematische Verzögerungen; deshalb
  Minimum/Median und Umgebungsprotokoll.

---

## 14. Bausteine für Bericht und Präsentation

### 14.1 Kapitelgerüst für den Projektbericht (mit Belegstellen)

1. **Einleitung**: Leitfrage, Algorithmenauswahlproblem `[rice1976]`,
   `[kerschke2019]`.
2. **Grundlagen**: Komplexitätsklassen (Folien 4–6), Datenmerkmale
   (Abschnitt 4), drei Familien der Suchverfahren (Abschnitt 3, Quellen
   `[shneiderman1978]`, `[bentley1976]`, `[ferguson1960]`, `[khuong2017]`,
   `[peterson1957]`, `[perl1978]`, `[gonnet1980]`, `[vansandt2019]`,
   `[richter2015]`).
3. **Implementierung**: Paketstruktur (Abschnitt 2), eine Implementierung
   mit zwei Varianten (A4), Semantik erstes Vorkommen (A1), Hashtabelle (A7),
   Guard (A5), Liste als Knotenpool (A13).
4. **Messmethodik**: Abschnitt 5, Tabelle der Entscheidungen mit Quellen;
   Stromsparmodus-Vorfall (A35) als Beispiel für Messbias; Yanniks
   Erweiterung (gepaarte Blöcke, Replikate, Validierung des Kostenmodells,
   A41/A39).
5. **Ergebnisse**: Abschnitt 7 (Tabellen 7.1 bis 7.5), Diagramme
   `size_runtime.pdf`, `size_comparisons.pdf`, `distribution.pdf`,
   `access_pattern.pdf`, `structure_runtime.pdf`, `structure_accesses.pdf`,
   `amortization.pdf`, `analysis_cost.pdf`, `analysis_sample.pdf`.
6. **Empfehlungssystem**: Abschnitt 6; Regelwerk mit Herkunft der
   Schwellen; Baum; Bewertung (Tabelle 7.6, `recommender_summary.tex`,
   `recommender_slowdown.pdf`, `learned_tree.pdf`); Meta-Algorithmus.
7. **Oberfläche**: sechs Seiten, Entscheidungsweg als Graph, Wettrennen,
   Lernmodus (`[hundhausen2002]`: aktives Lernen).
8. **Grenzen**: A18, A22/A39 (Modell bei kleinem n/q), A25, A26, A31
   (nicht kostensensitiv), B2 (Hardwareabhängigkeit), B5, B10.
9. **Fazit**: die vier Punkte von Folie 33, Punkt 2 eingeschränkt auf den
   Messlaptop.

### 14.2 Folienideen

- Folie „Wettrennen“: Screenshot oder Live-Demo der Lernmodus-Ansicht
  (lineare gegen binäre Suche, n = 24, Schlüssel am Ende: 23 gegenüber 6
  Schritte). Erklärt in 20 Sekunden, warum log₂ n gewinnt.
- Folie „Empfehlung erklärt sich“: Steckbrief + sieben Fragen mit Antworten
  (Seite 4) statt nur des Graphen.
- Folie „Zwei Rechner, zwei Rangfolgen“: Interpolation vs. binär
  verzweigungsfrei bei n = 10⁶, uniform: M5 43 / 25 ns, Intel 70 / 128 ns.
  Beleg für „Zeiten gelten nur für die Messplattform“.
- Folie „Kostenmodell geprüft“: Verhältnis gemessen/Modell aus
  `amortization_validation.csv` (0,9 bis 1,4 bei n = 10⁵, bis 2,25 bei
  n = 10³, q = 1).
- Folie 22: „getestet wurden 0, 8 und 16“ streichen oder belegen (Abschnitt
  11).
- Folie 36: Jahreszahlen Kotthoff/Smith-Miles mit den Dateien abgleichen.

---

## 15. Offene Punkte für das Team

1. Entscheiden, ob `results/full/` mit dem neuen Rasterformat auf dem
   Messlaptop neu gemessen wird (818 Datensätze, vier Muster), damit
   Bericht und Code vollständig zusammenpassen; sonst A39 als Erklärung
   verwenden. Beide Pipelines liefern dieselben Größenordnungen.
2. Quelle `charikar2000` in den Drive-Ordner legen oder das Zitat ersetzen
   (B5).
3. Folien 22, 24, 33, 36 anpassen (Abschnitt 11).
4. Falls gewünscht: `environment.py` um eine Windows-Stromabfrage erweitern
   und Git in Yanniks Pfad aufnehmen, damit seine Ergebnisdateien
   `git_commit` und `power_source` tragen (B11).
5. Einheitlicher Stand: Dieser Branch enthält `main`, Yanniks Branch und
   diese Durchsicht. Nach dem Review in `main` mergen, dann arbeiten alle
   auf derselben Basis.
