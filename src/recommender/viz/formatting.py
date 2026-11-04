"""German number and duration formatting for the user interface and the graphs.

All numbers shown to the user use the German convention: decimal comma and a
dot as thousands separator (``1.234.567`` and ``0,018``). Durations get a
readable unit from nanoseconds up to hours.

Pitfall fixed here: plain ``f"{x:.3g}"`` switches to scientific notation as
soon as a value has more digits than ``digits`` (``1234.5`` -> ``1.23e+03``,
``999.99`` -> ``1e+03``). Such values are shown as rounded integers instead.
"""

import math

import numpy as np

DASH = "–"  # shown for missing values (None, NaN)


def _integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def german(value: float | int | None, digits: int = 3) -> str:
    """Number with German decimal comma and thousands dots.

    Integers are shown exactly (also beyond 2^53, where floats lose digits).
    Other values are shown with ``digits`` significant digits, values that do
    not fit into ``digits`` digits as rounded integers.
    """
    if value is None:
        return DASH
    if isinstance(value, (bool, np.bool_)):
        return "ja" if value else "nein"
    if isinstance(value, (int, np.integer)):
        return _integer(int(value))
    value = float(value)
    if math.isnan(value):
        return DASH
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if value.is_integer() and abs(value) >= 1:
        return _integer(int(value))
    text = f"{value:.{digits}g}"
    if "e+" in text:  # too many digits before the comma: no scientific notation
        return _integer(round(value))
    return text.replace(".", ",")


# (unit, nanoseconds per unit), largest first
_UNITS = (("h", 3_600e9), ("min", 60e9), ("s", 1e9), ("ms", 1e6), ("µs", 1e3))


def format_ns(ns: float | None) -> str:
    """Duration in nanoseconds with a readable unit (ns, µs, ms, s, min, h)."""
    if ns is None or math.isnan(ns):
        return DASH
    for unit, factor in _UNITS:
        if abs(ns) >= factor:
            return f"{german(round(ns / factor, 2))} {unit}"
    return f"{german(round(ns, 1))} ns"


def percent(share: float, digits: int = 1) -> str:
    """Share in [0, 1] as German percentage, e.g. ``0.125`` -> ``12,5 %``."""
    if share is None or math.isnan(share):
        return DASH
    return f"{german(round(share * 100, digits))} %"
