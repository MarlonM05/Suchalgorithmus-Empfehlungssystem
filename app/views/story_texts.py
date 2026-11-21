"""Explanatory texts of the stories, shared by the app and the static website.

Both the story page of the app (``story_page.py``) and the website builder
(``presentation/build_site.py``) take their wording from here, so the two
always say the same. Plain Python without Streamlit; Markdown-style ``**bold**``
is converted by each caller.
"""

from dataclasses import dataclass

from recommender.features import DatasetFeatures
from recommender.rules.decision import FEATURE_NAMES_DE, DecisionStep
from recommender.viz.formatting import german, percent
from recommender.viz.graph import format_value

from .stories import Story

BOOLEAN_FEATURES = {"is_list", "sorted_known", "looks_sorted", "position_front"}


@dataclass(frozen=True)
class ProfileCard:
    """One tile of the profile ("Steckbrief")."""

    label: str
    value: str
    verdict: str
    help: str = ""


def data_text(story: Story) -> list[str]:
    """What the picture of the data shows, as short paragraphs (with **bold**)."""
    lines = [
        "Die Punkte steigen von links nach rechts: **die Daten sind sortiert**."
        if story.dataset.order == "sorted"
        else "Die Punkte liegen kreuz und quer: **die Daten sind nicht sortiert**."
    ]
    shape = {
        "uniform": "Die Werte verteilen sich gleichmäßig über den ganzen Bereich.",
        "lognormal": "Die meisten Werte sind klein, nur wenige sehr groß: Die Kurve bleibt lange flach und schießt am Ende hoch.",
    }.get(story.dataset.distribution)
    if shape:
        lines.append(shape)
    if story.dataset.structure == "linked_list":
        lines.append("In einer **verketteten Liste** kennt jedes Element nur seinen Nachfolger. Man kommt nur Schritt für Schritt voran.")
    return lines


def profile_cards(features: DatasetFeatures, story: Story) -> list[ProfileCard]:
    """The four tiles of the profile: size, order, distribution, effort of linear search."""
    workload = story.workload
    if features.ks_uniform < 0.05:
        shape = "✅ gleichmäßig"
    elif features.ks_uniform >= 0.5:
        shape = "⚠️ stark schief"
    else:
        shape = "leicht ungleichmäßig"
    return [
        ProfileCard(
            "Größe",
            f"n = {german(features.n)}",
            "✅ sehr klein: Vorbereitung lohnt kaum" if features.n <= 100 else "groß: die Wahl des Verfahrens zählt",
        ),
        ProfileCard(
            "Ordnung",
            f"{percent(features.disorder, 1)} falsch",
            "✅ wirkt sortiert (wird vor der Suche geprüft)" if features.looks_sorted else "❌ nicht sortiert",
            "Anteil der Nachbarpaare in falscher Reihenfolge, geschätzt auf einer Stichprobe.",
        ),
        ProfileCard(
            "Verteilung",
            f"KS = {german(round(features.ks_uniform, 2))}",
            shape,
            "Abstand zur Gleichverteilung: 0 = ganz gleichmäßig, nahe 1 = stark schief.",
        ),
        ProfileCard(
            "Durchläufe linear",
            f"≈ {german(round(workload.linear_scans, 1))}",
            "Treffer liegen vorne" if workload.position == "front" else "Treffer überall verteilt",
            "So oft müsste die lineare Suche für alle Anfragen zusammen komplett durch die Daten laufen.",
        ),
    ]


def answer_parts(step: DecisionStep) -> tuple[bool, str]:
    """Answer of one decision step: (yes?, measured value in words or "" for yes/no questions)."""
    if step.feature in BOOLEAN_FEATURES:
        return step.answer, ""
    return step.answer, f"{FEATURE_NAMES_DE.get(step.feature, step.feature)} = {format_value(step.feature, step.value)}"


def agreement_text(recommended: str, fastest: str, ratio: float) -> str:
    """How the recommendation compares with the measurement (names in German, ratio >= 1)."""
    if recommended == fastest:
        return "✅ Laut Messung ist das auch die schnellste Strategie."
    if ratio < 1.15:
        return f"Laut Messung praktisch gleichauf mit „{fastest}“ ({percent(ratio - 1, 0)} Unterschied)."
    return f"⚠️ Laut Messung wäre „{fastest}“ {german(round(ratio, 1))}-mal so schnell."
