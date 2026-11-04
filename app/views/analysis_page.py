"""Page 3 – Analyse: estimated features and the cost of estimating them."""

import numpy as np
import streamlit as st

from recommender.features import DatasetFeatures, verify_sorted

from . import charts
from .common import current_features, format_ns, german, invalidate_results, keys_in_order, next_step, page_header, percent

SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000]


def _uniformity_text(ks: float) -> str:
    if ks < 0.05:
        return "praktisch gleichverteilt"
    if ks < 0.2:
        return "leicht ungleich verteilt"
    if ks < 0.5:
        return "deutlich ungleich verteilt"
    return "stark schief verteilt"


def _consequences(features: DatasetFeatures) -> list[str]:
    """What the estimated features mean for the search algorithms (plain language)."""
    lines = []
    if features.structure == "linked_list":
        lines.append(":material/link: **Verkettete Liste:** nur sequenzieller Zugriff. Binäre, exponentielle, Fibonacci- und Interpolationssuche scheiden aus.")
    if features.looks_sorted:
        lines.append(":material/sort: **Wirkt sortiert:** Verfahren für sortierte Daten kommen in Frage. Vorher wird vollständig geprüft, außer du garantierst die Sortierung.")
    else:
        lines.append(
            f":material/shuffle: **Nicht sortiert** ({percent(features.disorder)} der Nachbarpaare falsch herum): Ohne Vorverarbeitung bleibt nur die lineare Suche; "
            "sonst Sortieren oder eine Hashtabelle."
        )
    if features.ks_uniform < 0.05:
        lines.append(":material/straighten: **Gleichverteilte Werte:** ideal für die Interpolationssuche, die die Position aus dem Wert schätzt.")
    elif features.ks_uniform >= 0.2:
        lines.append(":material/warning: **Schiefe Verteilung:** Die Interpolationssuche schätzt hier schlecht und kann bis zu O(n) Schritte brauchen.")
    if features.duplicate_share:
        lines.append(f":material/content_copy: **Duplikate** (etwa {percent(features.duplicate_share)}): Alle Verfahren liefern das erste Vorkommen.")
    return lines


def _illustration_sample(sample_size: int) -> np.ndarray:
    """A random sample of values for the distribution chart (same size as the analysis)."""
    values = keys_in_order(st.session_state.data)
    if values.size <= sample_size:
        return values
    return values[np.random.default_rng(0).integers(0, values.size, sample_size)]


def render() -> None:
    page_header(
        3,
        "Die Merkmale werden auf einer **Stichprobe** geschätzt, denn eine vollständige Prüfung kostet so viel wie eine lineare Suche [kerschke2019]. "
        "Die Kosten der Analyse werden mitgemessen.",
    )
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    sample_size = left.select_slider(
        "Stichprobengröße",
        options=SAMPLE_SIZES,
        value=st.session_state.get("sample_size", 1_000),
        format_func=german,
        help="Größere Stichproben schätzen genauer, kosten aber mehr Zeit. Die Wahl gilt auch für Empfehlung und Ausführung.",
    )
    if sample_size != st.session_state.get("sample_size"):
        st.session_state.sample_size = sample_size
        invalidate_results()
    if right.button("Analyse neu ausführen", icon=":material/refresh:", width="stretch"):
        st.session_state.pop("features", None)
    features = current_features()

    a, b, c = st.columns(3)
    a.metric("Datengröße n", german(features.n), border=True)
    b.metric("Speicherstruktur", "verkettete Liste" if features.structure == "linked_list" else "Array", border=True)
    c.metric("Kosten der Analyse", format_ns(features.analysis_time_ns), border=True, help="Gemessene Zeit für die Schätzung aller Merkmale (ohne einmalige Kompilierung).")

    st.subheader("Geschätzte Merkmale")
    a, b, c = st.columns(3)
    a.metric(
        "Falsch geordnete Nachbarpaare",
        percent(features.disorder, 2),
        border=True,
        help=f"Geschätzt auf {german(features.sample_size)} zufälligen Nachbarpaaren. 0 % bei sortierten Daten, etwa 50 % bei zufälliger Reihenfolge.",
    )
    b.metric(
        "Duplikatanteil",
        "nicht schätzbar" if features.duplicate_share is None else percent(features.duplicate_share),
        border=True,
        help="Nur bei sortiert wirkenden Daten schätzbar (gleiche Nachbarn). Bei unsortierten Daten lässt sich die Anzahl verschiedener Werte aus einer Stichprobe nicht zuverlässig schätzen [charikar2000].",
    )
    c.metric(
        "KS-Abstand zur Gleichverteilung",
        german(round(features.ks_uniform, 3)),
        border=True,
        help="Kolmogorow-Smirnow-Abstand: größte Lücke zwischen der Verteilung der Stichprobe und einer Gleichverteilung. 0 = perfekt gleichverteilt, nahe 1 = stark schief.",
    )
    verdict = "wirkt **sortiert**" if features.looks_sorted else "ist **nicht sortiert**"
    st.write(f"Die Stichprobe {verdict}; die Werte sind {_uniformity_text(features.ks_uniform)}.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Was das für die Suchverfahren bedeutet**")
        for line in _consequences(features):
            st.markdown(f"- {line}")
    with right:
        if features.n > 1:
            st.markdown("**Verteilung der Stichprobe gegenüber der Gleichverteilung**")
            st.altair_chart(charts.sample_cdf(_illustration_sample(st.session_state.sample_size)), width="stretch")
            st.caption("Blau: Anteil der Werte bis x. Grau gestrichelt: perfekte Gleichverteilung. Der größte senkrechte Abstand ist der KS-Abstand.")

    st.subheader("Vollständige Sortierprüfung")
    st.write(
        "Eine Stichprobe kann Sortierung nicht beweisen: Ein einziges vertauschtes Paar würde die binäre Suche falsch machen. "
        "Bevor ein Verfahren für sortierte Daten läuft, wird deshalb vollständig geprüft (O(n), bricht beim ersten Fehler ab)."
    )
    if st.button("Jetzt vollständig prüfen", icon=":material/fact_check:"):
        st.session_state.sort_check = verify_sorted(st.session_state.data)
    check = st.session_state.get("sort_check")
    if check is not None:
        if check.is_sorted:
            st.success(f"Die Daten sind sortiert. Die Prüfung dauerte {format_ns(check.time_ns)}.", icon=":material/check_circle:")
        else:
            st.warning(f"Die Daten sind nicht sortiert. Die Prüfung brach nach {format_ns(check.time_ns)} beim ersten Fehler ab.", icon=":material/warning:")
    next_step(3)
