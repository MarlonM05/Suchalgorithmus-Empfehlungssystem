"""Shared Matplotlib style and helpers for the figures of the paper.

All figures are exported as vector PDF with embedded fonts. Labels are German,
because the figures go into the German paper.

Design rules:

* **Every algorithm keeps one colour and one marker in all figures**
  (colour follows the entity). The eight colours are a categorical palette
  that was checked for colour-vision deficiencies (adjacent pairs). Because
  three of them have low contrast on white, every series also has its own
  marker and every figure has a legend, so identity never depends on colour
  alone, also in black-and-white print.
* Linked-list variants use the colour of the same algorithm on arrays with a
  dashed line.
* Thin lines, hairline grid, no second y-axis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Categorical palette, fixed order (validated: CVD-safe for adjacent series).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# key -> (German label, colour, marker, line style)
# The entries follow the palette order, so neighbouring series in the legend
# are exactly the validated adjacent pairs. Linear, binary and hash search
# (the three strategies of the amortization figure) take the first three
# slots, which are distinguishable in every combination; jump search takes
# violet, which is distinguishable from linear and binary search in every
# combination (structure figure).
ALGORITHM_STYLE: dict[str, tuple[str, str, str, str]] = {
    "linear": ("Lineare Suche", PALETTE[0], "o", "-"),
    "binary": ("Binäre Suche", PALETTE[1], "s", "-"),
    "hash": ("Hash-Lookup", PALETTE[2], "p", "-"),
    "binary_branchless": ("Binäre Suche (verzweigungsfrei)", PALETTE[3], "D", "-"),
    "exponential": ("Exponentielle Suche", PALETTE[4], "v", "-"),
    "fibonacci": ("Fibonacci-Suche", PALETTE[5], "P", "-"),
    "jump": ("Sprungsuche", PALETTE[6], "^", "-"),
    "interpolation": ("Interpolationssuche", PALETTE[7], "X", "-"),
    "list_linear": ("Lineare Suche (Liste)", PALETTE[0], "o", "--"),
    "list_jump": ("Sprungsuche (Liste)", PALETTE[6], "^", "--"),
    # Analysis steps are not search algorithms: neutral inks instead of a 9th hue.
    "features": ("Merkmalsanalyse (Stichprobe)", INK, "h", ":"),
    "verify_sorted": ("Vollständige Sortierprüfung", INK_MUTED, "8", ":"),
}
ALGORITHM_ORDER = list(ALGORITHM_STYLE)


def apply_style() -> None:
    """Global Matplotlib settings for all figures of the paper."""
    plt.rcParams.update(
        {
            "figure.figsize": (6.3, 3.6),  # fits the text width of an A4 page
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,  # embed TrueType fonts (selectable text in the PDF)
            "axes.formatter.use_locale": False,
        }
    )


def label(algorithm: str) -> str:
    return ALGORITHM_STYLE[algorithm][0]


def style_kwargs(algorithm: str) -> dict:
    """Matplotlib keyword arguments for the lines/markers of one algorithm."""
    _, color, marker, linestyle = ALGORITHM_STYLE[algorithm]
    return {
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "markeredgecolor": "white",  # surface ring keeps overlapping markers legible
        "markeredgewidth": 0.8,
        "label": label(algorithm),
    }


def sort_algorithms(keys) -> list[str]:
    """Algorithms in the fixed legend order."""
    return sorted(set(keys), key=ALGORITHM_ORDER.index)


def german_number(value: float) -> str:
    """Format numbers with German separators (1.000.000; 0,5)."""
    if float(value).is_integer():
        return f"{int(value):,}".replace(",", ".")
    return f"{value:g}".replace(".", ",")


def legend_below(fig: plt.Figure, axes, columns: int = 3) -> None:
    """One shared legend below the plot area (keeps the data area free)."""
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=columns, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.05 * ((len(labels) - 1) // columns), 1, 1))


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save as vector PDF and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_over_size(summary, value: str, ylabel: str, title: str, path: Path, x: str = "n", band: bool = True) -> Path:
    """Log-log line chart of ``value`` over the data size, one line per algorithm.

    ``summary`` has one row per (x, algorithm). If ``band`` is set, the 95 %
    bootstrap confidence interval of the median (``ci_low``/``ci_high``) is drawn
    as a light band around the line.
    """
    apply_style()
    fig, axes = plt.subplots()
    for algorithm in sort_algorithms(summary["algorithm"]):
        rows = summary[summary["algorithm"] == algorithm].sort_values(x)
        axes.plot(rows[x], rows[value], **style_kwargs(algorithm))
        if band and "ci_low" in rows:
            axes.fill_between(rows[x], rows["ci_low"], rows["ci_high"], color=ALGORITHM_STYLE[algorithm][1], alpha=0.12, linewidth=0)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Datengröße n")
    axes.set_ylabel(ylabel)
    axes.set_title(title, loc="left")
    axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: german_number(v) if v >= 1 else ""))
    legend_below(fig, axes)
    return save_figure(fig, path)


def plot_by_category(
    summary,
    category: str,
    categories: list,
    category_labels: list[str],
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool = True,
) -> Path:
    """Dot plot: one column per category, one dot (with CI whisker) per algorithm."""
    apply_style()
    fig, axes = plt.subplots()
    algorithms = sort_algorithms(summary["algorithm"])
    width = 0.7
    for index, algorithm in enumerate(algorithms):
        offset = (index - (len(algorithms) - 1) / 2) * width / max(len(algorithms), 1)
        rows = summary[summary["algorithm"] == algorithm].set_index(category).reindex(categories)
        positions = [i + offset for i in range(len(categories))]
        kwargs = style_kwargs(algorithm)
        kwargs.update(linestyle="none", markersize=7)
        axes.errorbar(
            positions,
            rows["median"],
            yerr=[rows["median"] - rows["ci_low"], rows["ci_high"] - rows["median"]],
            elinewidth=0.8,
            capsize=0,
            **kwargs,
        )
    axes.set_xticks(range(len(categories)), category_labels)
    axes.grid(axis="x", visible=False)
    if log_y:
        axes.set_yscale("log")
    axes.set_ylabel(ylabel)
    axes.set_title(title, loc="left")
    legend_below(fig, axes)
    return save_figure(fig, path)
