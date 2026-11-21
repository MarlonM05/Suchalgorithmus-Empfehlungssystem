"""Build the static presentation website (GitHub Pages) from the stories of the app.

    python presentation/build_site.py

Writes ``site/``: an overview page and one page per story with the same
content as the presentation mode of the app (``app/views/start_page.py`` and
``story_page.py``), as plain HTML, CSS and JavaScript without external
libraries, so it also works offline. The GitHub Actions workflow
``.github/workflows/pages.yml`` publishes ``site/`` on every push to main.

All numbers are computed while building, with the same functions as the app,
and the times are measured on the computer that runs this script. Build it on
the measurement laptop (power adapter, low-power mode off), then the website
shows the same kind of numbers as the paper. The footer of every page states
where and when it was measured.
"""

import html
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from recommender.benchmark.environment import collect_environment, throttling_warning  # noqa: E402
from recommender.executor import warm_up  # noqa: E402
from recommender.rules import STRATEGIES, RuleBook, Workload, selection_features  # noqa: E402
from recommender.viz.formatting import german, percent  # noqa: E402
from recommender.viz.race import build_lanes, race_height, race_html  # noqa: E402
from recommender.benchmark.environment import read_results  # noqa: E402
from views.common import APP_URL  # noqa: E402
from views.start_page import PIPELINE  # noqa: E402
from views.stories import STORIES, Story  # noqa: E402
from views.story_page import COUNT_QUERIES, QUERY_STEPS, StoryResults, _curve_strategies, compute_story_results  # noqa: E402
from views.story_texts import agreement_text, answer_parts, data_text, profile_cards  # noqa: E402

SITE = ROOT / "site"
ASSETS = Path(__file__).parent / "assets"
REPOSITORY = "https://github.com/MarlonM05/Suchalgorithmus-Empfehlungssystem"
MAX_POINTS = 1_500
TITLE = "Suchalgorithmus-Empfehlungssystem"


def md(text: str) -> str:
    """Escape for HTML and turn **bold** into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def page(title: str, body: str, active: str, environment: dict) -> str:
    pills = "".join(
        f'<a class="pill{" active" if key == active else ""}" href="{key}.html">{story.icon} {html.escape(story.title)}</a>'
        for key, story in STORIES.items()
    )
    measured = f"Gemessen am {environment['date'][:10]} auf {html.escape(environment['cpu'])}, Code-Stand {html.escape(environment['git_commit'])}."
    return f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · {TITLE}</title>
<link rel="stylesheet" href="assets/style.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔎</text></svg>">
<script src="assets/site.js" defer></script>
</head>
<body>
<header class="topbar"><div class="wrap">
<a class="brand" href="index.html">🔎 {TITLE}</a>
<nav class="pills">{pills}<a class="pill{" active" if active == "aufbau" else ""}" href="aufbau.html">🧩 Aufbau</a><a class="pill app" href="{APP_URL}">▶ Zur App</a></nav>
</div></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">
DHBW Ravensburg · Theoretische Informatik · <a href="{REPOSITORY}">Code und Ausarbeitung auf GitHub</a><br>
{measured} Alle Zahlen wurden beim Erstellen der Seite mit dem Code des Projekts berechnet.
Das Werkzeug zum freien Ausprobieren (eigene Daten, CSV) startet lokal mit <code>streamlit run app/streamlit_app.py</code>.
</div></footer>
</body>
</html>
"""


# --- Overview -------------------------------------------------------------------------------


def evaluation() -> pd.DataFrame | None:
    path = ROOT / "results" / "full" / "recommender_summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return table[(table["Bereich"] == "alle") & (table["Analysekosten"] == "ja")].set_index("Selektor")


def overview(environment: dict) -> str:
    steps = []
    for index, (icon, title, text, rice) in enumerate(PIPELINE):
        if index:
            steps.append('<div class="arrow">➜</div>')
        steps.append(
            f'<div class="card"><div class="icon">{icon}</div><h3>{index + 1} · {html.escape(title)}</h3>'
            f'<div class="small">{html.escape(text)}</div><div class="rice">Rice: {html.escape(rice)}</div></div>'
        )
    results = ""
    table = evaluation()
    if table is not None and "Regelwerk" in table.index:
        rules = table.loc["Regelwerk"]
        sbs = table.loc[[name for name in table.index if name.startswith("Single Best")][0]]
        tiles = [
            ("Richtige Wahl", percent(rules["Trefferquote"], 0), f"Immer dieselbe Strategie: {percent(sbs['Trefferquote'], 0)}"),
            ("Abstand zum Optimum", f"{german(round(rules['Slowdown geom. Mittel'], 2))}×", f"Immer dieselbe Strategie: {german(round(sbs['Slowdown geom. Mittel'], 2))}×"),
            ("Lücke geschlossen", percent(rules["Lücke geschlossen (geom.)"], 0), "zwischen bester Einzelstrategie und Optimum"),
        ]
        cards = "".join(
            f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="verdict">{note}</div></div>'
            for label, value, note in tiles
        )
        results = (
            f'<section class="chapter"><h2>Was es bringt</h2><div class="grid three">{cards}</div>'
            f'<p class="small">Bewertet auf {german(int(rules["Instanzen"]))} Testfällen des Messrasters, die nicht zum Entwickeln der Regeln verwendet wurden, '
            "einschließlich der Kosten der Stichprobe.</p></section>"
        )
    cards = "".join(
        f'<a class="card story-card" href="{key}.html"><div class="icon">{story.icon}</div><h3>{html.escape(story.title)}</h3>'
        f'<div class="small">{html.escape(story.hook)}</div><span class="go">Geschichte ansehen →</span></a>'
        for key, story in STORIES.items()
    )
    body = f"""
<h1>Welches Suchverfahren passt zu meinen Daten?</h1>
<p class="lead">Linear, binär, Hashtabelle oder etwas ganz anderes? Das hängt von den Daten ab und davon,
wie oft gesucht wird. Unser Tool schaut sich beides an und empfiehlt die passende Strategie.</p>
<section class="chapter"><h2>So arbeitet das Tool</h2><div class="pipeline">{"".join(steps)}</div></section>
{results}
<section class="chapter"><h2>Sechs Geschichten</h2>
<p class="question">Jede Geschichte führt zu einer anderen Empfehlung. Klick auf eine Karte, um sie Schritt für Schritt zu erleben.</p>
<div class="grid three">{cards}</div></section>
<section class="chapter"><h2>Selbst ausprobieren</h2>
<p class="question">Die Geschichten oben sind vorbereitete Beispiele. In der App wählst du Daten und Anfragen selbst und siehst Schritt für Schritt,
wie das Tool analysiert, entscheidet und sucht.</p>
<p><a class="button" href="{APP_URL}">▶ Zur App</a> <a class="button secondary" href="aufbau.html">🧩 So ist es aufgebaut</a></p>
<p class="small">Die App läuft online auf einem geteilten Server und braucht nach längerer Pause etwa eine Minute zum Aufwachen.
Ihre Zeiten stammen von diesem Server, nicht vom Messlaptop.</p></section>
"""
    return page("Überblick", body, "", environment)


# --- Architecture page ----------------------------------------------------------------------

MODULES = [
    ("Daten und Anfragen", "src/recommender/data/", "Erzeugt Datenmengen mit gezielten Eigenschaften (Größe, Verteilung, Sortierung, Duplikate, Array oder verkettete Liste) und Suchanfragen mit Trefferquote und Zielposition."),
    ("Merkmalsanalyse", "src/recommender/features/", "Schätzt auf einer Stichprobe: Sind die Daten sortiert? Wie gleichmäßig sind sie verteilt? Gibt es Duplikate? Misst dabei die eigenen Kosten. Dazu eine exakte Sortierprüfung in O(n)."),
    ("Suchverfahren", "src/recommender/algorithms/", "Acht Verfahren für Arrays und zwei für Listen, alle mit Numba kompiliert. Jedes ist einmal geschrieben und zweimal übersetzt: mit Zählern für Vergleiche und ohne Zähler für faire Zeitmessungen."),
    ("Empfehlung", "src/recommender/rules/", "Zwei austauschbare Selektoren: das Regelwerk (Entscheidungsgraph in rules.yaml, jede Regel mit Begründung und Quelle) und ein gelernter Entscheidungsbaum. Beide liefern Strategie, Begründung und Entscheidungsweg."),
    ("Ausführung", "src/recommender/executor.py", "Führt eine Strategie samt Vorbereitung aus (Sortierprüfung, Sortieren, Hashtabelle) und misst Zeit, Vergleiche und Zugriffe."),
    ("Messumgebung", "src/recommender/benchmark/", "Blockmessung, verworfene Aufwärmläufe, wechselnde Eingaben, robuste Statistik (Median, Bootstrap-Konfidenzintervall) und ein Protokoll der Messumgebung in jeder Ergebnisdatei."),
    ("Experimente", "experiments/", "Messreihen zu jedem Merkmal, das Messraster aller Kombinationen und die Bewertung der Selektoren. Ergebnisse als CSV und PDF in results/full/."),
    ("Oberfläche", "app/", "Die Streamlit-App: Präsentation (Überblick, Geschichten), das Werkzeug in fünf Schritten und der Lernmodus."),
]


def architecture_page(environment: dict) -> str:
    rows = "".join(
        f"<tr><td><strong>{html.escape(name)}</strong><br><code>{html.escape(path)}</code></td><td>{html.escape(task)}</td></tr>" for name, path, task in MODULES
    )
    grid_text = ""
    grid = ROOT / "results" / "full" / "grid_features.csv"
    table = evaluation()
    if grid.exists() and table is not None and "Regelwerk" in table.index:
        datasets = len(read_results(grid)[0])
        grid_text = (
            f"<li><strong>Messraster:</strong> {german(datasets)} Datenmengen mit je drei Zugriffsmustern, aufgeteilt in eine Kalibrierungs- "
            f"und eine Testhälfte. Bewertet wird nur auf der Testhälfte ({german(int(table.loc['Regelwerk', 'Instanzen']))} Fälle).</li>"
        )
    body = f"""
<h1>So ist es aufgebaut</h1>
<p class="lead">Das Tool folgt dem Rahmen der Algorithmenauswahl nach Rice: Aus den Daten und Anfragen (Problemraum) werden Merkmale bestimmt,
eine Auswahlabbildung wählt daraus eine Strategie (Algorithmenraum), und deren Ausführung wird gemessen (Leistungsmaß).</p>
<section class="chapter"><h2>Architektur</h2>
<div class="card figure"><img src="assets/architektur.svg" alt="Architekturdiagramm: oben der Weg einer Empfehlung, darunter die Messungen, unten die Oberfläche"></div>
<p class="small">Oben der Weg einer Empfehlung, darunter die Messungen, aus denen die Regeln stammen, unten die Oberfläche.</p></section>
<section class="chapter"><h2>Die Bausteine</h2>
<table class="modules"><thead><tr><th>Baustein</th><th>Aufgabe</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="chapter"><h2>Wie das Regelwerk entsteht</h2>
<ol class="steps">
<li><strong>Messreihen:</strong> Jedes Merkmal wird einzeln variiert (Datengröße, Verteilung, Duplikate, Zugriffsmuster, Speicherstruktur), jeweils mit 30 Wiederholungen.</li>
<li><strong>Break-even-Punkte:</strong> Aus den Messungen folgt, ab wie vielen Anfragen sich Vorbereitung lohnt, z. B. eine Hashtabelle nach etwa 7 bis 34 kompletten Durchläufen der linearen Suche.</li>
<li><strong>Regeln:</strong> Diese Schwellen stehen mit Begründung und Quelle im Regelwerk. Kostenlose Merkmale wie Größe und Angaben zu den Anfragen werden zuerst abgefragt, die Stichprobe nur bei Bedarf.</li>
{grid_text}
<li><strong>Vergleich:</strong> Ein Entscheidungsbaum (scikit-learn, Tiefe 4) lernt dieselbe Aufgabe aus den Messdaten. Ohne Analysekosten sind beide gleich gut, mit Analysekosten gewinnt das Regelwerk, weil es die Stichprobe seltener braucht.</li>
</ol></section>
<section class="chapter"><h2>Selbst ausprobieren</h2>
<p><a class="button" href="{APP_URL}">▶ Zur App</a> <a class="button secondary" href="{REPOSITORY}">Code auf GitHub</a></p>
<p>Lokal mit echten Messwerten des eigenen Rechners:</p>
<pre><code>python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py</code></pre></section>
"""
    return page("Aufbau", body, "aufbau", environment)


# --- Story pages ----------------------------------------------------------------------------


def chapter(number: int, title: str, question: str, content: str) -> str:
    return f'<section class="chapter"><h2><span class="chapter-no">{number}</span>{html.escape(title)}</h2><p class="question">{html.escape(question)}</p>{content}</section>'


def what_if_data(story: Story, results: StoryResults) -> dict:
    """Cost curves of the shown strategies and the recommendation for every q of the slider."""
    measured = [r for r in results.measured if not r.check_failed]
    steps = sorted(set(QUERY_STEPS) | {story.workload.queries})
    selector = RuleBook.load()
    decisions, shown = {}, list(_curve_strategies(story, results))
    for q in steps:
        workload = Workload(queries=q, hit_ratio=story.workload.hit_ratio, position=story.workload.position, sorted_known=story.workload.sorted_known)
        recommended = selector.decide(selection_features(results.features, workload)).strategy
        chosen = next(r for r in measured if r.strategy == recommended)
        fastest = min(measured, key=lambda r: r.total_ns(q))
        if chosen not in shown:
            shown.append(chosen)  # every recommendation of the slider has its own line
        decisions[str(q)] = {
            "recommended": STRATEGIES[recommended].name_de,
            "total_ns": chosen.total_ns(q),
            "agreement": agreement_text(STRATEGIES[recommended].name_de, STRATEGIES[fastest.strategy].name_de, chosen.total_ns(q) / fastest.total_ns(q)),
        }
    curves = [{"name": STRATEGIES[r.strategy].name_de, "prep": r.check_ns + r.preprocessing_ns, "query": r.query_ns} for r in shown]
    return {"steps": steps, "default_q": story.workload.queries, "decisions": decisions, "curves": curves}


def story_page(story: Story, results: StoryResults, environment: dict) -> tuple[str, str]:
    """HTML of the story page and of its race (separate file, shown in an iframe)."""
    dataset, workload = story.dataset, story.workload
    position = "die gesuchten Einträge liegen vorne" if workload.position == "front" else "jeder Eintrag gleich wahrscheinlich"
    badges = "".join(
        f'<span class="badge{" orange" if orange else ""}">{html.escape(text)}</span>'
        for text, orange in [
            (f"n = {german(dataset.n)}", False),
            ("verkettete Liste" if dataset.structure == "linked_list" else "Array", False),
            ({"sorted": "sortiert", "unsorted": "unsortiert", "nearly_sorted": "fast sortiert"}[dataset.order], False),
            ({"uniform": "gleichverteilt", "lognormal": "lognormal (stark schief)"}.get(dataset.distribution, dataset.distribution), False),
            (f"{german(workload.queries)} Anfragen", True),
            (position, True),
        ]
    )
    header = f'<h1>{story.icon} {html.escape(story.title)}</h1><p class="lead">{html.escape(story.scenario)}</p><div class="badges">{badges}</div>'

    first = ", ".join(german(int(v)) for v in results.keys[:8])
    data = chapter(
        1, "Die Daten", "Wie sehen die Daten aus, die durchsucht werden sollen?",
        f'<div class="grid two"><div class="chart" id="chart-data"></div><div>{"".join(f"<p>{md(line)}</p>" for line in data_text(story))}'
        f'<p class="small">Die ersten Werte: {first} …</p></div></div>',
    )

    cards = "".join(
        f'<div class="card" title="{html.escape(card.help)}"><div class="label">{html.escape(card.label)}</div>'
        f'<div class="value">{html.escape(card.value)}</div><div class="verdict">{html.escape(card.verdict)}</div></div>'
        for card in profile_cards(results.features, story)
    )
    used = "gebraucht" if results.recommendation.uses_sampled_features else "nicht gebraucht: Größe und die Angaben zu den Anfragen reichten"
    profile = chapter(
        2, "Was das Tool erkennt", "Der Steckbrief der Daten: aus einer kleinen Stichprobe geschätzt, dazu die Angaben zu den Anfragen.",
        f'<div class="grid four">{cards}</div><p class="small">Die Stichprobe wurde für die Entscheidung {used}.</p>',
    )

    recommendation = results.recommendation
    name = STRATEGIES[recommendation.strategy].name_de
    steps = []
    for step in recommendation.path:
        yes, measured = answer_parts(step)
        badge = '<span class="badge yes">ja</span>' if yes else '<span class="badge no">nein</span>'
        steps.append(f'<li><span>{html.escape(step.question)}</span><span class="answer">{badge}<span class="measured">{html.escape(measured)}</span></span></li>')
    sources = ", ".join(f"[{s}]" for s in recommendation.sources)
    decision = chapter(
        3, "Die Entscheidung", "Welche Fragen stellt das Regelwerk, und wie werden sie beantwortet?",
        f'<div class="card recommendation"><div class="label">Empfehlung</div><div class="name">{html.escape(name)}</div>'
        f'<div>{html.escape(recommendation.reason)}</div>{f"<p class=small>Quellen: {html.escape(sources)}</p>" if sources else ""}</div>'
        f'<ol class="chain">{"".join(steps)}</ol><div class="chain-end">➜ {html.escape(name)}</div>',
    )

    race_document = ""
    if story.race:
        lanes = build_lanes(results.race_values, results.race_key, list(story.race))
        race_document = race_html(results.race_values, results.race_key, lanes, autoplay=False)
        race_block = (
            f"<p><strong>Im Kleinen:</strong> {results.race_values.size} Werte aus den Daten, gesucht wird {german(results.race_key)}. Starte das Rennen mit ▶.</p>"
            f'<iframe class="race-frame" src="rennen-{story.key}.html" height="{race_height(lanes, int(results.race_values.size))}" title="Wettrennen"></iframe>'
            '<p class="small">Das kleine Rennen hängt davon ab, welche Werte im Ausschnitt landen. Belastbar sind die gezählten Vergleiche auf allen Daten.</p>'
        )
    else:
        race_block = "<p>Die Hashtabelle springt direkt an die richtige Stelle. Ein Rennen über ein Array lässt sich für sie nicht zeigen, die gezählten Vergleiche schon.</p>"
    race = chapter(
        4, "Das Rennen", "Wer findet den Eintrag schneller? Erst anschaulich im Kleinen, dann gezählt auf allen Daten.",
        f'{race_block}<p><strong>Auf allen {german(dataset.n)} Werten:</strong> Vergleiche pro Suche, gemittelt über {COUNT_QUERIES} Suchen '
        f'mit dem Zugriffsmuster der Geschichte (logarithmische Länge).</p><div class="bars" id="chart-counts"></div>',
    )

    what_if = chapter(
        5, "Was wäre wenn?", "Wie ändert sich die beste Wahl, wenn mehr oder weniger gesucht wird? Schieb den Regler.",
        '<div id="what-if"><div class="slider-row"><input type="range" min="0" step="1" aria-label="Anzahl Anfragen"><output></output></div>'
        '<div class="chart"></div><div class="grid two" style="margin-top:1rem">'
        '<div class="card recommendation"><div class="label">Empfehlung des Regelwerks</div><div class="name" data-field="recommended"></div>'
        '<div class="verdict" data-field="agreement"></div></div>'
        '<div class="card"><div class="label">Geschätzte Gesamtzeit</div><div class="value" data-field="total"></div></div></div>'
        '<p class="small">Jede Linie ist Vorbereitung + q · Zeit pro Anfrage, gemessen auf den Daten dieser Geschichte. '
        "Wo sich Linien kreuzen, lohnt sich die Vorbereitung (Break-even).</p></div>",
    )

    lesson = chapter(6, "Die Lehre", "Was diese Geschichte zeigt.", f'<div class="card lesson">💡 {html.escape(story.lesson)}</div>')

    keys = list(STORIES)
    index = keys.index(story.key)
    previous = f'<a href="{keys[index - 1]}.html">← {html.escape(STORIES[keys[index - 1]].title)}</a>' if index else '<a href="index.html">← Überblick</a>'
    following = f'<a href="{keys[index + 1]}.html">{html.escape(STORIES[keys[index + 1]].title)} →</a>' if index + 1 < len(keys) else '<a href="index.html">Zurück zum Überblick →</a>'

    positions = np.unique(np.linspace(0, results.keys.size - 1, min(results.keys.size, MAX_POINTS)).astype(np.int64))
    payload = {
        "points": [[int(p), int(results.keys[p])] for p in positions],
        "counts": [{"name": row["Verfahren"], "value": float(row["Vergleiche"]), "recommended": bool(row["empfohlen"])} for _, row in results.counts.iterrows()],
        "what_if": what_if_data(story, results),
    }
    script = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    body = (
        header + data + profile + decision + race + what_if + lesson
        + f'<div class="next-story">{previous}{following}</div>'
        + f'<script type="application/json" id="story-data">{script}</script>'
    )
    return page(story.title, body, story.key, environment), race_document


def build(site: Path = SITE, progress: bool = True) -> None:
    """Write the complete website into ``site`` (replaces its previous content)."""
    warm_up()  # compile first, so no compilation time ends up in the measurements
    environment = collect_environment()
    if site.exists():
        shutil.rmtree(site)
    (site / "assets").mkdir(parents=True)
    for asset in ASSETS.iterdir():
        shutil.copy(asset, site / "assets" / asset.name)
    (site / ".nojekyll").write_text("")  # serve the files as they are (no Jekyll processing)
    shutil.copy(ROOT / "docs" / "architektur.svg", site / "assets" / "architektur.svg")
    (site / "index.html").write_text(overview(environment), encoding="utf-8")
    (site / "aufbau.html").write_text(architecture_page(environment), encoding="utf-8")
    selector = RuleBook.load()
    for key, story in STORIES.items():
        if progress:
            print(f"Geschichte {story.icon} {story.title} …", flush=True)
        results = compute_story_results(key, selector)
        document, race_document = story_page(story, results, environment)
        (site / f"{key}.html").write_text(document, encoding="utf-8")
        if race_document:
            (site / f"rennen-{key}.html").write_text(race_document, encoding="utf-8")


def main() -> None:
    warning = throttling_warning()
    if warning:
        print(warning)
    build(SITE)
    print(f"fertig: {SITE.relative_to(ROOT)}/ ({len(list(SITE.rglob('*')))} Dateien)")


if __name__ == "__main__":
    main()
