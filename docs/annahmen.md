# Annahmen und Grenzen

Diese Datei sammelt alle Annahmen, die während der Implementierung getroffen
wurden. Sie dient als Grundlage für das Kapitel „Grenzen der Arbeit“.

## M1 – Suchverfahren

### A1: Einheitliche Semantik „erstes Vorkommen“
Alle Verfahren liefern bei Duplikaten den Index des **ersten** Vorkommens. Die
Verfahren für sortierte Daten suchen dazu direkt die *untere Schranke*
(kleinster Index `i` mit `arr[i] >= key`) und prüfen anschließend auf
Gleichheit. Ein nachträgliches lineares Zurücklaufen zum ersten Duplikat wird
vermieden, weil es bei vielen Duplikaten O(k) Zusatzkosten verursachen und die
Messung verfälschen würde. Referenz für die Tests ist
`np.searchsorted(side="left")`.

### A2: Datentyp
Alle Verfahren arbeiten auf eindimensionalen `int64`-Arrays. Andere
Datentypen (Gleitkommazahlen, Zeichenketten) werden nicht betrachtet.

### A3: Definition der gezählten Operationen
- **Vergleich**: Vergleich des Suchschlüssels mit einem gespeicherten Schlüssel.
- **Zugriff**: Lesen eines gespeicherten Elements (Arrayzelle bzw. Slot der
  Hashtabelle).

Vergleiche von Schleifenindizes werden nicht gezählt. Bei der
Interpolationssuche zählen das Lesen von `arr[lo]` und `arr[hi-1]` in jedem
Schritt als zwei Zugriffe und zwei Vergleiche. Bei der Hashtabelle zählt das
Prüfen eines leeren Slots als Zugriff, aber nicht als Vergleich. Die
abschließende Gleichheitsprüfung der Kandidatenposition wird bei allen
Verfahren für sortierte Daten gleich gezählt. Die Zählweise ist eine
Konvention: Absolute Zahlen sind nur innerhalb dieser Arbeit vergleichbar.

### A4: Faire Kompilierung und Zählvarianten
Alle Verfahren sind mit Numba (`@njit`) kompiliert. Jedes Verfahren wird
einmal implementiert und daraus in zwei Varianten kompiliert: mit und ohne
Zählcode. In der Messvariante entfernt der Compiler den Zählcode vollständig
(geprüft am erzeugten LLVM-IR). Laufzeitmessungen verwenden ausschließlich die
Messvariante, Operationszahlen ausschließlich die Zählvariante. Die
Zählvariante ist spürbar langsamer (lineare Suche ca. 7×, weil die Schleife
nicht mehr vektorisiert wird) und eignet sich daher nicht für Zeitmessungen.

### A5: Interpolationssuche – Guard und Abgrenzung zu SIP
Implementiert ist die klassische Interpolationssuche [peterson1957] mit dem
Guard aus [vansandt2019, Alg. 2, Z. 14–17]: Liegt die geschätzte Position
höchstens `GUARD_SIZE = 8` Elemente von einem Intervallende entfernt, wird ab
diesem Ende sequenziell weitergesucht. Der Wert 8 stammt aus
[vansandt2019, Anhang A.3] („best all-around“ für SIP). Er wurde dort für C++
und andere Hardware ermittelt und von uns nicht neu kalibriert.

Nicht übernommen sind die weiteren SIP-Optimierungen (Wiederverwendung der
Steigung, Festkommaarithmetik) und TIP. Unsere Interpolationssuche ist daher
nicht mit SIP gleichzusetzen.

Grenze: Der Guard verkürzt typische Suchen, garantiert aber **kein**
O(log n). Bei stark schiefen Daten oder langen Duplikatketten nahe der
geschätzten Position kann der sequenzielle Teil weiterhin linear viele
Elemente besuchen.

### A6: Fibonacci-Suche als Variante mit unterer Schranke
Die Fibonacci-Suche [ferguson1960] ist so formuliert, dass sie die untere
Schranke findet (zweiwertiger statt dreiwertiger Vergleich). Das Array wird
gedanklich mit +∞ auf `F(k) − 1` Elemente aufgefüllt. Diese virtuellen
Positionen werden nie gelesen. Die asymptotische Komplexität O(log n) und die
Beschränkung auf Addition und Subtraktion bleiben erhalten.

### A7: Hashtabelle
Eigene Hashtabelle mit offener Adressierung, linearem Sondieren und
multiplikativem (Fibonacci-)Hashing, Füllgrad ≤ 0,5 (Kapazität =
kleinste Zweierpotenz ≥ 2n). Begründung: [richter2015] identifiziert diese
Kombination als robust und schnell für Ganzzahlschlüssel. Andere
Hash-Verfahren (Cuckoo, Robin Hood, Verkettung) werden nicht verglichen. Der
Speicherbedarf der Tabelle (2 · Kapazität · 8 Byte) wird in der Arbeit nicht
als Kriterium bewertet.

### A8: Sprungweite
Die Sprungsuche verwendet die theoretisch optimale Sprungweite ⌊√n⌋
[shneiderman1978]. Andere Sprungweiten werden nicht untersucht.

### A9: Erste Ausführung inklusive Kompilierung
Die kompilierten Funktionen sind Closures und werden von Numba nicht auf der
Festplatte zwischengespeichert. Der erste Aufruf jeder Variante enthält
deshalb die Kompilierzeit. Die Messumgebung (M3) verwirft diesen Aufruf als
Aufwärmlauf [kalibera2013], [georges2007].

## M2 – Datengeneratoren und Merkmalsextraktion

### A10: Erzeugung der Schlüssel
Stetige Zufallszahlen werden mit einem festen Faktor pro Verteilung skaliert
und auf `int64` abgerundet (gleichverteilt U(0,1) · 2⁵², normalverteilt
N(0,1) · 2⁴⁰, exponentiell Exp(1) · 2⁴⁰, lognormal LogN(0, 2) · 2³²,
geclustert · 2⁵²). Die großen Faktoren verhindern unbeabsichtigte Duplikate
durch das Runden. Die Lognormal-Parameter µ = 0, σ = 2 stammen aus
[vansandt2019].

Geclusterte Daten sind eine Mischung aus 10 schmalen Normalverteilungen
(Standardabweichung 0,001) mit festen, gleichverteilten Zentren in
[0,05; 0,95]. Die Zentren sind für alle Seeds gleich, damit sich nur die
Stichprobe ändert und nicht die Form der Verteilung. Anzahl und Breite der
Cluster sind frei gewählt.

### A11: Definition des Duplikatanteils
Duplikatanteil p = (n − Anzahl verschiedener Werte) / n. Der Generator zieht
n − round(p · n) verschiedene Werte und füllt mit zufälligen Kopien daraus
auf. Kopien verteilen sich zufällig über die Werte, es gibt also viele kurze
Duplikatketten statt weniger langer. Lange Ketten (z. B. ein dominanter Wert)
werden nicht untersucht.

### A12: Sortiergrade
- **sortiert**: aufsteigend. Absteigend sortierte Daten werden nicht betrachtet.
- **fast sortiert**: sortiert, danach round(k · n) Vertauschungen zweier
  zufälliger Positionen (Standard k = 1 %). Jede Vertauschung stört bis zu
  vier Nachbarpaare, der gemessene Anteil falsch geordneter Nachbarpaare liegt
  deshalb bei etwa 2k.
- **unsortiert**: zufällige Permutation (Anteil falsch geordneter Paare ≈ 0,5).

### A13: Verkettete Liste als Knotenpool
Die Liste ist als Knotenpool gespeichert (Arrays `values` und `next_index`).
Die Knoten liegen in zufälliger Reihenfolge im Speicher, damit das Durchlaufen
wie Zeigerverfolgung im Heap wirkt. Der Pool erlaubt es, für die
Merkmalsanalyse zufällige Knoten in O(1) auszuwählen. Bei einer echten
Liste ohne Pool wäre schon die Stichprobe O(n), die Analysekosten für Listen
sind daher eine **untere Schranke**.

### A14: Suchverfahren auf Listen
Auf Listen werden nur lineare Suche und Sprungsuche angeboten. Binäre,
exponentielle, Fibonacci- und Interpolationssuche benötigen Direktzugriff,
auf einer Liste kostet bereits das Erreichen der Mitte n/2 Zeigerschritte.
Bei der Sprungsuche auf Listen zählt jeder besuchte Knoten als Zugriff. Der
Rücksprung zum gemerkten Blockanfang ist kostenlos.

### A15: Zugriffsmuster
- Position „vorne gehäuft“: Die Trefferposition folgt einer
  Exponentialverteilung mit Mittelwert 1 % von n, abgeschnitten auf das Array.
  Damit liegen etwa 63 % der Treffer im ersten Prozent und 99 % in den ersten
  4,6 % der Positionen. Der Wert 1 % ist frei gewählt.
- Fehlschläge werden gleichverteilt aus dem Wertebereich der Daten gezogen
  (fallen also in Lücken zwischen gespeicherten Schlüsseln). Hat der
  Wertebereich keine Lücken, liegen sie außerhalb.
- Bei Daten mit Duplikaten ist die richtige Antwort das erste Vorkommen des
  gezogenen Werts.

### A16: Merkmale auf Stichproben
Standard-Stichprobengröße 1000 (Ziehen mit Zurücklegen). Begründung: Der
95-%-Grenzwert der KS-Statistik liegt dann bei 1,36/√1000 ≈ 0,043, der
Standardfehler des Sortiergrads bei höchstens 0,016. Bei höchstens 1000
Paaren bzw. Werten werden alle verwendet (exaktes Ergebnis).

- **Sortiergrad**: Anteil falsch geordneter Nachbarpaare auf zufälligen
  Paaren. Eine Stichprobe kann Sortierung nicht beweisen, deshalb gibt es die
  vollständige O(n)-Prüfung `verify_sorted` mit eigener Kostenangabe. Sie
  bricht beim ersten Fehler ab und ist bei unsortierten Daten daher sehr billig.
- **Duplikatanteil**: nur bei (der Stichprobe nach) sortierten Daten
  schätzbar, über gleiche Nachbarpaare. Bei unsortierten Daten ist die Anzahl
  verschiedener Werte aus einer Stichprobe nicht zuverlässig schätzbar
  [charikar2000], das Merkmal bleibt dann leer.
- **Gleichverteilung**: KS-Abstand der Stichprobe zur Gleichverteilung auf
  [Stichprobenminimum, Stichprobenmaximum]. Das Minimum und Maximum der
  Gesamtdaten würde einen O(n)-Durchlauf erfordern.

### A17: Messung der Analysekosten
Die Analyse ist wie die Suchverfahren mit Numba kompiliert. Die gemessene
Zeit enthält nicht die einmalige Kompilierung. Beobachtung: Wiederholt man
die Analyse mit **derselben** Stichprobe, sinkt die Zeit von ca. 95 µs auf
ca. 33 µs, weil die CPU Sprungvorhersage und Cache auf die wiederholten
Zugriffe einstellt. Realistisch sind die Kosten mit wechselnden Stichproben.
Die Messumgebung (M3) variiert deshalb Seeds und Eingaben zwischen den
Wiederholungen [mytkowicz2009].

## M3 – Messumgebung und Experimente

### A18: Messplattform
Alle Messungen laufen auf einem einzigen Rechner (Laptop, Apple M5, macOS),
die Umgebung steht in jeder Ergebnisdatei. macOS erlaubt es weder, Threads
fest an Leistungskerne zu binden, noch die dynamische Taktanpassung
abzuschalten. Absolute Laufzeiten gelten daher nur für diese Plattform. Die
gezählten Operationen (Vergleiche, Zugriffe) ergänzen sie als
hardwareunabhängiges Maß. Empfehlung für die vollständige Messreihe: Netzteil
angeschlossen, andere Programme geschlossen, Rechner während der Messung nicht
benutzen.

### A19: Blockmessung
Ein gemessener Block dauert mindestens 1 ms (etwa 25 000-fache
Timer-Auflösung) [chen2016]. Die Blocklänge wird durch Verdoppeln kalibriert
(höchstens 100 000 Anfragen). Ist schon eine einzelne Anfrage länger als 1 ms
(z. B. lineare Suche bei n = 10⁷), besteht der Block aus einer Anfrage. Die
Zeit pro Anfrage enthält den Schleifen-Overhead des Blocks (≈ 1 ns). Er ist
für alle Verfahren gleich.

### A20: Wiederholungen
Vollständige Messreihe: 30 gemessene Wiederholungen nach Kompilierung und 2
verworfenen Aufwärmläufen [kalibera2013], [georges2007]. Jede Wiederholung
zieht einen neuen zufälligen Block aus einem Pool von 10 000 Anfragen (A17).
Schnelltest: 5 Wiederholungen, nur zur Funktionsprüfung, nicht für die Arbeit.

### A21: Einzelmessung von Vorverarbeitung und Analyse
Sortieren, Hashtabellen-Aufbau, Merkmalsanalyse und Sortierprüfung werden pro
Aufruf gemessen, nicht in Blöcken. Ihre Zeiten enthalten deshalb den
Aufruf-Overhead von Python in den kompilierten Code (einige hundert ns). Das
entspricht den Kosten, die das Tool tatsächlich hat, überwiegt aber bei sehr
kleinem n.

### A22: Amortisationsrechnung
Die Gesamtkosten T(q) = Vorverarbeitung + q · Kosten pro Anfrage werden aus
getrennt gemessenen Komponenten berechnet (lineares Kostenmodell), nicht als
Gesamtlauf gemessen. Wechselwirkungen (z. B. Cache-Zustand nach dem Sortieren)
sind nicht berücksichtigt. Die Break-even-Punkte werden aus den Medianen
berechnet und haben kein eigenes Konfidenzintervall. Sortiert wird mit dem
Numba-kompilierten `np.sort`, gesucht mit der klassischen binären Suche.

### A23: Versuchsplan „ein Faktor zur Zeit“
Jedes Experiment variiert ein Merkmal und hält die anderen auf einer Basis
fest: gleichverteilt, sortiert, keine Duplikate, nur Treffer mit
gleichverteilter Position, Array, n = 10⁶ (Schnelltest 10⁵). Wechselwirkungen
zwischen Merkmalen (z. B. lognormal *und* viele Duplikate) zeigen diese
Experimente nicht. Sie werden erst durch das Messraster für das
Empfehlungssystem (M4) abgedeckt.

### A24: Zusammenfassung der Speedups
Das geometrische Mittel der Speedups gegenüber der binären Suche
[fleming1986] gewichtet jede gemessene Datengröße gleich. Die Wahl der
Größenstufen beeinflusst damit das Mittel.

## M4 – Empfehlungssystem

### A25: Messraster
Für Training und Bewertung wird jede Kombination gemessen aus Datengröße
(10, 10², …, 10⁶), fünf Verteilungen, drei Sortiergraden, drei
Duplikatanteilen (0 %, 10 %, 50 %) und zwei Speicherstrukturen, also 540
Datenmengen, jeweils mit drei Zugriffsmustern (Treffer gleichverteilt, Treffer
vorne gehäuft, 50 % Fehlschläge). n = 10⁷ fehlt aus Zeit- und
Speichergründen. Pro Messpunkt gibt es 10 statt 30 Wiederholungen, weil die
beste Strategie meist deutlich vorne liegt. Die Anzahl der Anfragen q (1 bis
10⁶) wird nicht gemessen, sondern über das lineare Kostenmodell (A22)
eingerechnet.

### A26: Strategieraum
Empfohlen werden Strategien, also Suchverfahren samt Vorverarbeitung:
lineare Suche, die Verfahren für sortierte Daten (mit vorheriger
Sortierprüfung, wenn die Sortierung nicht garantiert ist), „Hashtabelle
aufbauen + Lookup“ und „Sortieren + verzweigungsfreie binäre Suche“, für
Listen „lineare Suche“, „Sprungsuche“ und „in Hashtabelle kopieren“. Das
Leistungsmaß ist allein die Zeit. Speicherbedarf und Bereichsanfragen, bei
denen Sortieren der Hashtabelle überlegen wäre, sind nicht berücksichtigt.
Deshalb gewinnt die Hashtabelle bei vielen Anfragen fast immer.

### A27: Sortiergarantie und Sortierprüfung
Ob die Sortierung garantiert ist, gibt der Nutzer an (wie die Speicherstruktur).
Ohne Garantie kostet jede Strategie, die Sortierung voraussetzt, zusätzlich die
vollständige Prüfung (A16). Schlägt die Prüfung fehl, weil die Stichprobe eine
Unordnung übersehen hat, wird ihr Abbruchzeitpunkt bezahlt und mit dem Wissen
„unsortiert“ neu entschieden.

### A28: Analysekosten in der Bewertung
Alle Maße werden mit und ohne die Kosten der Merkmalsanalyse angegeben. Die
Analysekosten fallen nur an, wenn der Entscheidungsweg ein Stichprobenmerkmal
braucht. Das Regelwerk stellt diese Fragen deshalb zuletzt [kerschke2019].
Der konstante Aufruf-Overhead (A21) ist in allen gemessenen Kosten enthalten.

### A29: Bestes Verfahren (Oracle)
Das Oracle ist die Strategie mit den geringsten berechneten Gesamtkosten.
Nahezu gleich schnelle Strategien (z. B. klassische und verzweigungsfreie
binäre Suche) werden dadurch willkürlich getrennt. Deshalb wird neben der
exakten Trefferquote auch der Anteil der Empfehlungen innerhalb von 10 % des
Optimums berichtet.

### A30: Aufteilung in Kalibrierung und Test
Die Datenmengen werden je Kombination aus Größe und Struktur zur Hälfte in
Kalibrierung und Test aufgeteilt. Keine Datenmenge liegt in beiden Hälften,
auch nicht mit anderem q oder Zugriffsmuster. Der Baum wird auf der
Kalibrierungshälfte trainiert, der Single Best Solver dort bestimmt, und
Anpassungen am Regelwerk dürfen sich nur auf die Kalibrierungshälfte stützen.
Alle Maße beziehen sich auf die Testhälfte. Der Single Best Solver wird je
Speicherstruktur unter den immer anwendbaren Strategien gewählt (bestes
geometrisches Mittel des Slowdowns).

### A31: Entscheidungsbaum
scikit-learn, `criterion="entropy"` (ID3 [quinlan1986]), maximale Tiefe 4,
mindestens 20 Fälle pro Blatt, Eingaben wie beim Regelwerk (keine
Generatoreinstellungen). Gelernt wird eine reine Klassifikation auf die
Oracle-Strategie, nicht kostensensitiv: Ein teurer Fehler zählt im Training
so viel wie ein billiger. Empfiehlt der Baum eine Strategie für die falsche
Speicherstruktur, wird sie durch ihr Gegenstück ersetzt (z. B. Hashtabelle →
Liste in Hashtabelle), und der Fall wird gezählt.

### A32: Stichproben-Meta-Algorithmus [vansandt2019]
Jedes Verfahren beantwortet 10 Probeanfragen, das schnellste wird gewählt (5
Durchgänge je Datenmenge und Zugriffsmuster). Verglichen wird nur auf
sortierten Arrays mit Sortiergarantie, weil der Meta-Algorithmus keine
Vorverarbeitung vorsieht. Die Probezeit wird voll angerechnet, die
Probeanfragen zählen nicht als beantwortete Anfragen.

### A33: Erwartete Durchläufe der linearen Suche
Die Regeln nutzen q · (Anteil des Arrays, den eine lineare Suche im Mittel
liest): ½ bei gleichverteilten Treffern, 1 bei Fehlschlägen und der
Mittelwert der Trefferposition (1 % von n) bei vorne gehäuften Treffern. Der
letzte Wert ist an den Anfragegenerator gebunden (A15). In der Anwendung
muss der Nutzer ihn einschätzen.

### A34: Nachschärfung des Regelwerks für kleine Datenmengen (Version 2)
Version 1 des Regelwerks leitete alle Schwellen aus M3 ab. Dort wurde die
Amortisation erst ab n = 1 000 gemessen. Bei n ≤ 100 empfahl Version 1
deshalb zu früh eine Hashtabelle oder Sortierprüfung, weil jede
Vorverarbeitung einen festen Grundaufwand hat (z. B. Hash-Aufbau ≈ 960 ns
schon bei n = 10). Version 2 ergänzt eigene Fragen für n ≤ 16 und n ≤ 100
(Arrays) bzw. n ≤ 16 (Listen) mit Schwellen in Anfragen statt in Durchläufen
und verzichtet bei n ≤ 100 auf die Stichprobe.

Die Schwellen (200, 40 und 100 Anfragen) stammen aus den gemessenen
Break-even-Punkten und wurden **nur auf der Kalibrierungshälfte** verglichen
(Varianten mit 50/100/200, 20/40/100, 50/100 Anfragen). Die Testhälfte
blieb dabei unberührt. Ergebnisse auf der Testhälfte:

| Regelwerk | Trefferquote | Slowdown ohne / mit Analysekosten | Lücke geschlossen (geom.) ohne / mit |
|---|---|---|---|
| Version 1 (nur M3) | 84,1 % | 1,122 / 1,340 | 91,2 % / 77,6 % |
| Version 2 | 87,3 % | 1,078 / 1,236 | 94,2 % / 83,8 % |

Das Raster enthält bei kleinen Datenmengen nur n = 10 und n = 100 sowie
die Anfragezahlen 1, 3, 10, 30, 100, 300, …. Schwellen zwischen diesen
Stufen (z. B. 40 statt 100) unterscheidet es nicht, dort entscheidet der
gemessene Break-even-Punkt.

### A35: Gedrosselte CPU während des Messrasters
Im Messraster (19.10.2026) sind alle Zeiten gleichmäßig etwa doppelt so hoch
wie in der Messreihe von M3 (16.10.2026), z. B. lineare Suche bei n = 10⁶:
≈ 230 µs statt ≈ 113 µs. Der Faktor ist vom Anfang bis zum Ende des Laufs
konstant, ein schleichender Effekt wie Überhitzung ist daher
unwahrscheinlich. Ursache (am 19.10.2026 nachgeprüft): Der Laptop lief im Akkubetrieb mit
eingeschaltetem Stromsparmodus (`pmset -g`: `lowpowermode 1`). Eine
Kontrollmessung mit den Einstellungen von M3 ergab im selben Zustand
ebenfalls ≈ 245 ns statt 117 ns für die lineare Suche bei n = 1 000. Auch
eine Wiederholung des Rasters im selben Zustand war gleich langsam.
Seitdem protokolliert jede Ergebnisdatei Stromquelle und Stromsparmodus
(`power_source`, `low_power_mode`), und jedes Experiment warnt vor dem Start,
wenn der Rechner gedrosselt laufen könnte.

**Behoben:** Das Raster wurde am 19.10.2026 um 13:53 mit ausgeschaltetem
Stromsparmodus neu gemessen (Akkubetrieb). Die Zeiten stimmen jetzt mit M3
überein (lineare Suche bei n = 1 000: 116,6 ns statt 117 ns, bei n = 10⁶:
112,7 µs statt 113,1 µs). Akkubetrieb ohne Stromsparmodus macht auf diesem
Rechner also keinen messbaren Unterschied. Alle Ergebnisse von M4 in
`results/full/` beruhen auf dieser Messung. Die Schwellen des Regelwerks
(Version 2) wurden auf der Kalibrierungshälfte erneut geprüft und bleiben
unverändert. Keine Variante war mit Analysekosten besser. Weil alle Komponenten gleich betroffen
sind, bleiben Verhältnisse und Empfehlungen weitgehend gleich. Die absoluten
Zeiten aus Raster und M3 sollten aber nicht direkt verglichen werden.

## M5 – Oberfläche

### A36: Zeiten in der automatischen Ausführung
Die App misst die Zeit pro Anfrage als Mittel aus 1 000 Wiederholungen
**derselben** Suche. Weil sich die CPU auf wiederholte Eingaben einstellt
(A17), ist dieser Wert eher optimistisch. Sortierprüfung und
Vorverarbeitung werden einmal gemessen, inklusive Aufruf-Overhead. Die
angezeigte Gesamtzeit für q Anfragen folgt dem Kostenmodell aus A22. Die
Zahlen der App dienen der Veranschaulichung, die Ergebnisse der Arbeit
stammen aus den Messreihen in `results/full/`.

### A37: Sortieren in der App
Für „Sortieren + binäre Suche“ sortiert die App mit einem stabilen
`np.argsort`. Nur so lässt sich der gefundene Index auf die ursprüngliche
Position zurückrechnen, und wegen der Stabilität bleibt das erste Vorkommen
erhalten. In den Messreihen wurde stattdessen das Numba-kompilierte
`np.sort` gemessen. Die Sortierzeiten der App sind deshalb nicht direkt mit
den Messreihen vergleichbar.

### A38: Grenzen der Oberfläche
- Verkettete Listen sind auf n ≤ 10⁶ begrenzt, weil schon eine Suche auf
  längeren Listen Sekunden dauert.
- Hochgeladene CSV-Dateien müssen eine Spalte mit ganzen Zahlen enthalten
  (A2). Leere Zeilen werden übersprungen.
- Der Lernmodus arbeitet mit höchstens 64 Werten, die auf 0 bis 999
  skaliert werden (die Form der Verteilung bleibt erhalten, Duplikate
  können entstehen). Der Hash-Lookup ist dort nicht enthalten, weil seine
  Hashtabelle kein geordnetes Array ist.
- Der Entscheidungsbaum der App ist der auf der Kalibrierungshälfte
  trainierte Baum aus `results/full/learned_tree.json`.
- Der Entscheidungsweg wird standardmäßig kompakt gezeigt: nur der Weg und
  an jeder Frage die nicht gewählte Abzweigung. Der vollständige Graph ist
  zuschaltbar.
