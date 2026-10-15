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
