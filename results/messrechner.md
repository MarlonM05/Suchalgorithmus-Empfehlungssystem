# Messrechner und Messbedingungen

Welche Ergebnisdateien von welchem Rechner stammen. Absolute Zeiten im Bericht kommen nur vom Messlaptop (Annahme A18), die beiden Intel-Rechner dienen dem Vergleich von Verhältnissen (Bericht, Tabelle 4.2).

| Ordner / Dateien | Rechner | Wer | Code-Stand | Stromquelle und Energiemodus |
|---|---|---|---|---|
| `full/` (ohne `*andre-laptop*`) | Messlaptop, Apple M5, macOS | Marlon | steht in jeder Datei | in jeder Datei protokolliert |
| `full_y/`, `full_y2/` | Desktop-PC, Intel Core Ultra 5 245KF, Windows 11 | Yannik | Branch `messmethodik_test` (erweiterte Messmethodik), nicht protokolliert | nicht protokolliert (`unknown`) |
| `full/size_datasets_andre-laptop*` | Laptop, Intel Core Ultra 7 255H, Windows 11 | André | `c7bfb4f` (Branch `app-zusammenfuehren`) | vor der Messung von Hand geprüft: Netzteil, Windows-Energiemodus „Beste Leistung“ |

## Hinweise

- `full_y/` und `full_y2/` sind zwei vollständige Läufe am 03.11.2026 auf demselben Rechner. Sie wurden mit dem Code aus `messmethodik_test` erzeugt (u. a. Raster mit 818 Datenmengen); dieser Code ist nicht in `main`. Für den Bericht wird nur `size.csv` verwendet (Mittel beider Läufe).
- `size_datasets_andre-laptop*` entstand am 14.11.2026 mit `experiments/exp_size_datasets.py --mode full --label andre-laptop` (fünf Datensätze pro Größe, Messpunkte in zufälliger Reihenfolge). Unter Windows schreibt das Skript Stromquelle und Energiemodus als `unknown`, deshalb stehen die Bedingungen hier.
- Unter Windows liegt die Zeit pro Suche schon bei kleinen n bei rund 25–30 ns (Mac: 6–7 ns). Absolute Zeiten der Intel-Rechner sind deshalb nicht mit dem Mac vergleichbar, nur die Verhältnisse zwischen den Verfahren.
