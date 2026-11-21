# Screenshots für Ausarbeitung und Präsentation

Vorschlag, welche Ansichten der App in die Arbeit gehören, mit genauen
Einstellungen, damit die Bilder reproduzierbar sind.

**Vorbereitung:** `streamlit run app/streamlit_app.py`, Browserfenster etwa
1440 × 900, helles Theme (Menü ⋮ → Settings → Theme „Light“). Aufnahme unter
macOS mit `⌘ + ⇧ + 4`, dann Leertaste und Fenster anklicken.

| Nr. | Seite | Einstellungen | Zeigt | Kapitel |
|---|---|---|---|---|
| 1 | Daten | n = 100.000, lognormal, sortiert, 0 % Duplikate, Array, Seed 1, Werte umrechnen an | Erzeugung mit Histogramm einer schiefen Verteilung | Empfehlungssystem: Bedienung |
| 2 | Zugriffsmuster | q = 100, Trefferquote 100 %, gleichverteilt, Sortierung nicht garantiert | Nutzerangaben und „erwartete Durchläufe“ | Empfehlungssystem: Merkmale |
| 3 | Analyse | wie 1, Stichprobe 1.000, danach „Jetzt vollständig prüfen“ | geschätzte Merkmale, Analysekosten, Sortierprüfung | Experimentelle Analyse: Kosten der Merkmale |
| 4 | Empfehlung | wie 1–2, Regelwerk, kompakte Ansicht | Empfehlung mit Begründung und Entscheidungsweg | Empfehlungssystem: Regelwerk |
| 5 | Empfehlung | wie 4, „Vollständigen Graphen zeigen“ an | gesamtes Regelwerk mit hervorgehobenem Pfad | Empfehlungssystem: Regelwerk |
| 6 | Empfehlung | wie 4, „Gelernter Entscheidungsbaum“ | Weg durch den gelernten Baum | Empfehlungssystem: Vergleich der Varianten |
| 7 | Ausführung | wie 1–2, danach „Alle Verfahren vergleichen“ | automatische Ausführung und Vergleichstabelle | Empfehlungssystem: Ausführung |
| 8 | Ausführung | Daten: n = 100.000, gleichverteilt, **unsortiert**; q = 1.000 | Empfehlung Hashtabelle, Vergleich zeigt Amortisation | Empfehlungssystem: Amortisation |
| 9 | Lernmodus | Interpolationssuche, n = 32, lognormal, Seed 3, zufälliger Treffer, einige Schritte weiter | Guard und Probleme bei schiefen Daten | Präsentation |
| 10 | Lernmodus | Binäre Suche, n = 32, gleichverteilt, Seed 3, Mitte des Ablaufs | halbierter Suchbereich | Präsentation |

Für die Arbeit sind Vektorgrafiken schärfer als Screenshots. Den
Entscheidungsweg gibt es deshalb auch als DOT-Datei (Knopf „Graph als
DOT-Datei“ auf der Empfehlungsseite). Daraus wird mit Graphviz
(`brew install graphviz`) ein PDF:

```bash
dot -Tpdf entscheidungsweg.dot -o entscheidungsweg.pdf
```
