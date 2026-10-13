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
