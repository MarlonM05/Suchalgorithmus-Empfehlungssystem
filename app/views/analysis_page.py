"""Page 3 – Analyse: estimated features and the cost of estimating them."""

import streamlit as st

from recommender.features import extract_features, verify_sorted

from .common import format_ns, german

SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000]


def _uniformity_text(ks: float) -> str:
    if ks < 0.05:
        return "praktisch gleichverteilt"
    if ks < 0.2:
        return "leicht ungleich verteilt"
    if ks < 0.5:
        return "deutlich ungleich verteilt"
    return "stark schief verteilt"


def render() -> None:
    st.title("3 · Analyse")
    st.write(
        "Die Merkmale werden auf einer **Stichprobe** geschätzt, denn eine vollständige Prüfung kostet so viel "
        "wie eine lineare Suche (siehe [kerschke2019]). Die Kosten der Analyse werden mitgemessen."
    )
    sample_size = st.select_slider("Stichprobengröße", options=SAMPLE_SIZES, value=1_000, format_func=german)
    if st.button("Analyse neu ausführen") or "features" not in st.session_state or st.session_state.get("sample_size") != sample_size:
        st.session_state.features = extract_features(st.session_state.data, sample_size=sample_size)
        st.session_state.sample_size = sample_size
    features = st.session_state.features

    a, b, c = st.columns(3)
    a.metric("Datengröße n", german(features.n))
    b.metric("Speicherstruktur", "verkettete Liste" if features.structure == "linked_list" else "Array")
    c.metric("Kosten der Analyse", format_ns(features.analysis_time_ns))

    st.subheader("Geschätzte Merkmale")
    a, b, c = st.columns(3)
    a.metric(
        "Falsch geordnete Nachbarpaare",
        f"{german(round(features.disorder * 100, 2))} %",
        help=f"Geschätzt auf {german(features.sample_size)} zufälligen Nachbarpaaren. 0 % bei sortierten Daten, etwa 50 % bei zufälliger Reihenfolge.",
    )
    b.metric(
        "Duplikatanteil",
        "nicht schätzbar" if features.duplicate_share is None else f"{german(round(features.duplicate_share * 100, 1))} %",
        help="Nur bei sortiert wirkenden Daten schätzbar (gleiche Nachbarn). Bei unsortierten Daten lässt sich die Anzahl verschiedener Werte aus einer Stichprobe nicht zuverlässig schätzen [charikar2000].",
    )
    c.metric(
        "KS-Abstand zur Gleichverteilung",
        german(round(features.ks_uniform, 3)),
        help="0 = perfekt gleichverteilt, nahe 1 = stark schief. Entscheidend für die Interpolationssuche.",
    )
    verdict = "wirkt **sortiert**" if features.looks_sorted else "ist **nicht sortiert**"
    st.write(f"Die Stichprobe {verdict}; die Werte sind {_uniformity_text(features.ks_uniform)}.")

    st.subheader("Vollständige Sortierprüfung")
    st.write(
        "Eine Stichprobe kann Sortierung nicht beweisen: Ein einziges vertauschtes Paar würde die binäre Suche falsch machen. "
        "Bevor ein Verfahren für sortierte Daten läuft, wird deshalb vollständig geprüft (O(n), bricht beim ersten Fehler ab)."
    )
    if st.button("Jetzt vollständig prüfen"):
        st.session_state.sort_check = verify_sorted(st.session_state.data)
    check = st.session_state.get("sort_check")
    if check is not None:
        if check.is_sorted:
            st.success(f"Die Daten sind sortiert. Die Prüfung dauerte {format_ns(check.time_ns)}.")
        else:
            st.warning(f"Die Daten sind nicht sortiert. Die Prüfung brach nach {format_ns(check.time_ns)} beim ersten Fehler ab.")
