"""Draw the architecture diagram for the paper (docs/architektur.pdf and .svg).

    python docs/architektur_diagramm.py

The diagram maps the modules onto the algorithm selection framework of Rice
[rice1976]: problem space -> feature space -> selection mapping -> algorithm
space -> performance measure (top row, online for every user request), with
the offline measurement pipeline that derives the rules (middle row) and the
user interface that uses all modules (bottom row).
"""

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from recommender.viz import plots

HERE = Path(__file__).parent
ONLINE = {"face": "#e3eefb", "edge": plots.PALETTE[0]}
OFFLINE = {"face": "#f4f3ef", "edge": plots.INK_MUTED}
INTERFACE = {"face": "#ffffff", "edge": plots.INK_SECONDARY}
W, H = 1.9, 1.15  # box size


def box(axes, x, y, title, subtitle, module, style, width=W, height=H):
    axes.add_patch(
        FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=style["face"], edgecolor=style["edge"], linewidth=1.2)
    )
    cx = x + width / 2
    axes.text(cx, y + height - 0.22, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=plots.INK)
    axes.text(cx, y + height / 2 - 0.05, subtitle, ha="center", va="center", fontsize=7, color=plots.INK_SECONDARY, linespacing=1.3)
    axes.text(cx, y + 0.17, module, ha="center", va="center", fontsize=6.5, family="monospace", color=plots.INK_SECONDARY)


def arrow(axes, start, end, label="", dashed=False, color=plots.INK_SECONDARY, rad=0.0, label_offset=(0, 0.1)):
    axes.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=color,
            linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        )
    )
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        axes.text(mx, my, label, ha="center", va="bottom", fontsize=6.5, color=color, style="italic")


def main() -> None:
    plots.apply_style()
    fig, axes = plots.plt.subplots(figsize=(10.4, 5.4))
    axes.set_xlim(0, 10.4)
    axes.set_ylim(0, 5.6)
    axes.axis("off")

    # Row 1: online path of one recommendation, along Rice's framework
    top = 3.65
    xs = [0.1, 2.2, 4.3, 6.4, 8.5]
    axes.text(0.1, 5.35, "Online: eine Empfehlung (pro Datenmenge und Zugriffsmuster)", fontsize=9, fontweight="bold", color=plots.PALETTE[0])
    for x, label in zip(xs, ["Problemraum", "Merkmalsraum", "Auswahlabbildung S", "Algorithmenraum", "Leistungsmaß"]):
        axes.text(x + W / 2, top + H + 0.12, label, ha="center", fontsize=7.5, color=plots.PALETTE[0])
    box(axes, xs[0], top, "Daten + Anfragen", "Datenmenge (Array/Liste)\nZugriffsmuster, q", "data/", ONLINE)
    box(axes, xs[1], top, "Merkmalsanalyse", "Stichprobe: Sortierung,\nVerteilung, Duplikate", "features/", ONLINE)
    box(axes, xs[2], top, "Empfehlung", "Regelwerk (YAML) oder\nEntscheidungsbaum", "rules/", ONLINE)
    box(axes, xs[3], top, "Strategie", "Vorverarbeitung +\nSuchverfahren (Numba)", "algorithms/", ONLINE)
    box(axes, xs[4], top, "Ausführung", "Index, Laufzeit,\nVergleiche, Zugriffe", "executor.py", ONLINE)
    for a, b in zip(xs, xs[1:]):
        arrow(axes, (a + W, top + H / 2), (b, top + H / 2))

    # Row 2: offline measurements that produce the rules and the paper's results (right to left)
    mid = 1.6
    axes.text(0.1, 3.0, "Offline: Messungen (einmalig)", fontsize=9, fontweight="bold", color=plots.INK_SECONDARY)
    box(axes, xs[3], mid, "Messumgebung", "Blockmessung, Statistik,\nUmgebungsprotokoll", "benchmark/", OFFLINE)
    box(axes, xs[2], mid, "Experimente", "Messreihen M3,\nMessraster M4, Bewertung", "experiments/", OFFLINE)
    box(axes, xs[1], mid, "Ergebnisse", "Rohdaten (CSV),\nDiagramme (PDF), Baum", "results/full/", OFFLINE)
    arrow(axes, (xs[3] + W / 2, top), (xs[3] + W / 2, mid + H))
    axes.text(xs[3] + W / 2 + 0.08, (top + mid + H) / 2, "misst alle Verfahren\nauf erzeugten Daten", fontsize=6.5, va="center", color=plots.INK_SECONDARY, style="italic")
    arrow(axes, (xs[3], mid + H / 2), (xs[2] + W, mid + H / 2))
    arrow(axes, (xs[2], mid + H / 2), (xs[1] + W, mid + H / 2))
    arrow(axes, (xs[1] + W - 0.2, mid + H), (xs[2] + 0.35, top), dashed=True, color=plots.PALETTE[0])
    axes.text(xs[2] + 0.45, (top + mid + H) / 2 + 0.02, "Schwellen der Regeln,\nTraining des Baums", fontsize=6.5, va="center", ha="left", color=plots.PALETTE[0], style="italic")

    # Row 3: user interface
    bottom = 0.15
    axes.add_patch(FancyBboxPatch((0.1, bottom), 10.2, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=INTERFACE["face"], edgecolor=INTERFACE["edge"], linewidth=1.2))
    axes.text(5.2, bottom + 0.65, "Oberfläche (Streamlit) und Visualisierung – nutzt alle Module", ha="center", fontsize=8.5, fontweight="bold", color=plots.INK)
    axes.text(
        5.2, bottom + 0.3,
        "Daten · Zugriffsmuster · Analyse · Empfehlung mit Entscheidungsweg (Graphviz) · Ausführung und Vergleich · Lernmodus",
        ha="center", fontsize=7, color=plots.INK_SECONDARY,
    )
    axes.text(10.2, bottom + 0.1, "app/, viz/", ha="right", fontsize=6.5, family="monospace", color=plots.INK_SECONDARY)

    for suffix in (".pdf", ".svg"):
        fig.savefig(HERE / f"architektur{suffix}", bbox_inches="tight")
    print("written: docs/architektur.pdf, docs/architektur.svg")


if __name__ == "__main__":
    main()
