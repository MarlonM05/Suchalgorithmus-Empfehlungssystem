"""Interactive charts of the app (Altair, rendered by ``st.altair_chart``).

Altair is installed with Streamlit. Compared with Matplotlib images the charts
follow the light/dark theme of the app, have tooltips with exact German
numbers and keep the numeric axis order (the previous histogram used text
labels such as ``1.5e+15`` as categories).

Colours follow the project palette (``recommender.viz.plots``): blue for the
highlighted element, a muted grey for the rest.
"""

import altair as alt
import numpy as np
import pandas as pd

from recommender.viz.formatting import format_ns, german

HIGHLIGHT = "#2a78d6"
SECOND = "#eb6834"
MUTED = "#a3a29c"
MAX_POINTS = 1_500  # points of the position chart; more only slow the browser down
# Vega formats 10000 as "10,000"; this expression turns it into the German "10.000".
GERMAN_THOUSANDS = "replace(format(datum.value, ','), regexp(',', 'g'), '.')"


def histogram(values: np.ndarray, bins: int = 40) -> alt.Chart:
    """Histogram of the key values with numeric x axis."""
    counts, edges = np.histogram(values, bins=bins)
    frame = pd.DataFrame(
        {
            "von": edges[:-1],
            "bis": edges[1:],
            "Anzahl": counts,
            "Bereich": [f"{german(round(a))} bis {german(round(b))}" for a, b in zip(edges[:-1], edges[1:])],
        }
    )
    return (
        alt.Chart(frame, height=230)
        .mark_bar(color=HIGHLIGHT, opacity=0.85)
        .encode(
            x=alt.X("von:Q", bin="binned", title="Wert", axis=alt.Axis(format="~s")),
            x2="bis:Q",
            y=alt.Y("Anzahl:Q", title="Anzahl", axis=alt.Axis(labelExpr=GERMAN_THOUSANDS)),
            tooltip=["Bereich", alt.Tooltip("Anzahl:Q", format=",")],
        )
    )


def value_over_position(values: np.ndarray) -> alt.Chart:
    """Key value against its position (evenly thinned out for large data).

    Sorted data gives a rising curve, a straight line means uniformly spread
    values (the assumption of interpolation search), unsorted data a cloud.
    """
    positions = np.unique(np.linspace(0, values.size - 1, min(values.size, MAX_POINTS)).astype(np.int64))
    frame = pd.DataFrame({"Position": positions, "Wert": values[positions]})
    frame["Wert (exakt)"] = [german(int(v)) for v in frame["Wert"]]
    return (
        alt.Chart(frame, height=230)
        .mark_circle(size=14, color=HIGHLIGHT, opacity=0.7)
        .encode(
            x=alt.X("Position:Q", title="Position im Speicher", axis=alt.Axis(format="~s")),
            y=alt.Y("Wert:Q", title="Wert", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Position:Q", format=","), "Wert (exakt)"],
        )
    )


def sample_cdf(sample: np.ndarray) -> alt.LayerChart:
    """Empirical distribution function of a sample against the uniform distribution.

    The x axis is scaled to [0, 1] between sample minimum and maximum. The
    largest vertical gap between the two curves is the KS distance.
    """
    values = np.sort(sample.astype(np.float64))
    span = values[-1] - values[0] if values.size > 1 and values[-1] > values[0] else 1.0
    frame = pd.DataFrame({"x": (values - values[0]) / span, "F": np.arange(1, values.size + 1) / values.size})
    empirical = (
        alt.Chart(frame, height=230)
        .mark_line(color=HIGHLIGHT, interpolate="step-after")
        .encode(
            x=alt.X("x:Q", title="Wert, skaliert auf [0, 1]"),
            y=alt.Y("F:Q", title="Anteil der Werte ≤ x"),
        )
    )
    uniform = alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "F": [0.0, 1.0]})).mark_line(color=MUTED, strokeDash=[5, 4]).encode(x="x:Q", y="F:Q")
    return uniform + empirical


def strategy_comparison(table: pd.DataFrame, queries: int) -> alt.Chart:
    """Horizontal bars of the estimated total time per strategy (log scale).

    ``table`` needs the columns ``Strategie``, ``total_ns`` and ``empfohlen``.
    Bars on a log scale cannot start at 0, so every bar starts at a common
    baseline (the power of ten below the fastest strategy).
    """
    frame = table[["Strategie", "total_ns", "empfohlen"]].copy()
    frame["Gesamtzeit"] = [format_ns(v) for v in frame["total_ns"]]
    frame["Rolle"] = np.where(frame["empfohlen"], "empfohlen", "Alternative")
    frame["Basis"] = 10.0 ** np.floor(np.log10(max(float(frame["total_ns"].min()), 1.0)))
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("Strategie:N", sort=alt.EncodingSortField("total_ns", order="ascending"), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("total_ns:Q", scale=alt.Scale(type="log"), title=f"Geschätzte Gesamtzeit für {german(queries)} Anfragen [ns, logarithmisch]", axis=alt.Axis(labelExpr=GERMAN_THOUSANDS)),
            x2="Basis:Q",
            color=alt.Color("Rolle:N", scale=alt.Scale(domain=["empfohlen", "Alternative"], range=[HIGHLIGHT, MUTED]), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["Strategie", "Gesamtzeit", "Rolle"],
        )
        .properties(height=alt.Step(30))
    )


def step_counts(frame: pd.DataFrame) -> alt.Chart:
    """Number of steps of every traced algorithm for the same key (learning mode).

    ``frame`` needs the columns ``Verfahren``, ``Schritte`` and ``aktiv`` (the selected algorithm).
    """
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("Verfahren:N", sort=alt.EncodingSortField("Schritte", order="ascending"), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("Schritte:Q", title="Vergleiche bis zum Ergebnis", axis=alt.Axis(tickMinStep=1)),
            color=alt.condition(alt.datum.aktiv, alt.value(HIGHLIGHT), alt.value(MUTED)),
            tooltip=["Verfahren", "Schritte"],
        )
        .properties(height=alt.Step(28))
    )


# Axis labels in readable units: 10 ns, 1 µs, 100 ms, 10 s, ...
TIME_LABELS = (
    "datum.value >= 1e9 ? format(datum.value / 1e9, '~g') + ' s' : "
    "datum.value >= 1e6 ? format(datum.value / 1e6, '~g') + ' ms' : "
    "datum.value >= 1e3 ? format(datum.value / 1e3, '~g') + ' µs' : format(datum.value, '~g') + ' ns'"
)


def _powers_of_ten(low: float, high: float) -> list[float]:
    return [10.0**e for e in range(int(np.floor(np.log10(max(low, 1e-9)))), int(np.ceil(np.log10(max(high, 1.0)))) + 1)]


def cost_curves(frame: pd.DataFrame, queries: int, max_queries: int = 10_000_000) -> alt.LayerChart:
    """Total cost T(q) = preparation + q * time per query of every strategy (log-log).

    ``frame`` has one row per strategy with ``Strategie``, ``prep_ns``,
    ``query_ns`` and ``empfohlen``. The recommended strategy is drawn in blue
    on top of the alternatives (grey), every line is labelled at its right end
    (no legend to decode), and a dashed rule marks the chosen number of queries.
    """
    q_values = np.unique(np.logspace(0, np.log10(max_queries), 80).round().astype(np.int64))
    rows = []
    for _, row in frame.iterrows():
        for q in q_values:
            total = float(row["prep_ns"]) + q * float(row["query_ns"])
            rows.append({"Strategie": row["Strategie"], "q": int(q), "total": total, "Gesamtzeit": format_ns(total), "empfohlen": bool(row["empfohlen"])})
    curves = pd.DataFrame(rows)
    x = alt.X(
        "q:Q",
        scale=alt.Scale(type="log", domain=[1, max_queries]),
        title="Anzahl Anfragen q",
        axis=alt.Axis(values=_powers_of_ten(1, max_queries), labelExpr=GERMAN_THOUSANDS, grid=True),
    )
    y = alt.Y(
        "total:Q",
        scale=alt.Scale(type="log"),
        title="Gesamtzeit",
        axis=alt.Axis(values=_powers_of_ten(curves["total"].min(), curves["total"].max()), labelExpr=TIME_LABELS),
    )

    def layer(data: pd.DataFrame, colour: str, width: float) -> alt.LayerChart:
        lines = alt.Chart(data).mark_line(color=colour, strokeWidth=width).encode(
            x=x, y=y, detail="Strategie:N", tooltip=["Strategie", alt.Tooltip("q:Q", format=","), "Gesamtzeit"]
        )
        ends = data[data["q"] == data["q"].max()]
        labels = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=13, color=colour, fontWeight="bold" if width > 3 else "normal").encode(
            x=x, y=y, text="Strategie:N"
        )
        return lines + labels

    alternatives = layer(curves[~curves["empfohlen"]], MUTED, 2.5)
    recommended = layer(curves[curves["empfohlen"]], HIGHLIGHT, 4.5)
    marker = alt.Chart(pd.DataFrame({"q": [queries]})).mark_rule(color=SECOND, strokeWidth=2, strokeDash=[6, 4]).encode(x="q:Q")
    at_q = frame.assign(q=queries, total=frame["prep_ns"] + queries * frame["query_ns"])
    at_q["Gesamtzeit"] = [format_ns(v) for v in at_q["total"]]
    points = (
        alt.Chart(at_q)
        .mark_circle(size=120, opacity=1)
        .encode(
            x=x,
            y=y,
            color=alt.condition(alt.datum.empfohlen, alt.value(HIGHLIGHT), alt.value(MUTED)),
            tooltip=["Strategie", "Gesamtzeit"],
        )
    )
    return (alternatives + recommended + marker + points).properties(height=380, padding={"right": 300})


def mean_comparisons(frame: pd.DataFrame) -> alt.Chart:
    """Mean number of key comparisons per search on the full data (horizontal bars).

    ``frame`` needs ``Verfahren``, ``Vergleiche`` and ``empfohlen``.
    """
    data = frame.copy()
    data["Rolle"] = np.where(data["empfohlen"], "empfohlen", "Alternative")
    data["Mittel"] = [german(round(v, 1)) for v in data["Vergleiche"]]
    # Logarithmic axis: 17 against 50.000 comparisons would otherwise leave the short bar invisible.
    data["Basis"] = 0.8
    high = max(float(data["Vergleiche"].max()), 10.0)
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("Verfahren:N", sort=alt.EncodingSortField("Vergleiche", order="ascending"), title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            x=alt.X(
                "Vergleiche:Q",
                scale=alt.Scale(type="log", domain=[0.8, high * 3]),
                title="Vergleiche pro Suche (Mittel, logarithmisch)",
                axis=alt.Axis(values=_powers_of_ten(1, high), labelExpr=GERMAN_THOUSANDS),
            ),
            x2="Basis:Q",
            color=alt.Color("Rolle:N", scale=alt.Scale(domain=["empfohlen", "Alternative"], range=[HIGHLIGHT, MUTED]), legend=None),
            tooltip=["Verfahren", "Mittel"],
        )
    )
    text = alt.Chart(data).mark_text(align="left", dx=6, fontSize=14, fontWeight="bold").encode(
        y=alt.Y("Verfahren:N", sort=alt.EncodingSortField("Vergleiche", order="ascending")), x="Vergleiche:Q", text="Mittel:N"
    )
    return (bars + text).properties(height=alt.Step(40))
