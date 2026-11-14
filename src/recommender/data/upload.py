"""Read search keys from an uploaded CSV or text file.

The user interface promises: "one column of integers is enough". Real files
differ in three ways, and all of them are handled here:

* **No header.** A file ``5\\n3\\n8`` has no column name. A plain
  ``pandas.read_csv`` would take ``5`` as the column name and silently drop the
  first value. A header is therefore only assumed if the first row is not
  completely numeric.
* **Separator.** German spreadsheet programs export with ``;``, others with
  ``,`` or tabs. The separator is detected from the first non-empty line.
* **Invalid values.** Empty cells are skipped (and counted); decimals,
  infinite values, values outside the ``int64`` range and yes/no columns are
  rejected with a German error message, because all algorithms work on
  ``int64`` keys (docs/annahmen.md A2).

The order of the file is kept, so the user can upload unsorted data.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

INT64_LIMIT = 2.0**63  # floats at or above this value do not fit into int64


@dataclass(frozen=True)
class UploadedKeys:
    """Keys read from one column of an uploaded file."""

    keys: np.ndarray  # contiguous int64 array in file order
    skipped: int  # number of empty cells that were left out


def _is_number(text: object) -> bool:
    try:
        float(str(text).strip())
    except ValueError:
        return False
    return True


def _separator(text: str) -> str:
    """Field separator of the file, detected from the first non-empty line."""
    first = next((line for line in text.splitlines() if line.strip()), "")
    for separator in (";", "\t", ","):
        if separator in first:
            return separator
    return r"\s+" if len(first.split()) > 1 else ","


def read_table(raw: bytes) -> pd.DataFrame:
    """Parse the uploaded bytes into a table (with or without header row).

    Raises ``ValueError`` with a German message if the file is empty or cannot
    be parsed.
    """
    text = raw.decode("utf-8-sig", errors="replace")  # utf-8-sig drops an Excel BOM
    if not text.strip():
        raise ValueError("Die Datei ist leer.")
    separator = _separator(text)
    engine = "python" if separator == r"\s+" else "c"
    try:
        frame = pd.read_csv(io.StringIO(text), sep=separator, engine=engine)
        if all(_is_number(name) for name in frame.columns):
            # The first row consists of numbers: it is data, not a header.
            frame = pd.read_csv(io.StringIO(text), sep=separator, engine=engine, header=None)
            frame.columns = [f"Spalte {i + 1}" for i in range(frame.shape[1])]
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError(f"Die Datei konnte nicht gelesen werden: {error}") from error
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that contain numbers (yes/no columns do not count)."""
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]


def column_keys(series: pd.Series) -> UploadedKeys:
    """Convert one column into int64 search keys; raises ``ValueError`` if impossible."""
    values = series.dropna()
    skipped = int(series.size - values.size)
    if values.empty:
        raise ValueError("Die Spalte enthält keine Werte.")
    if pd.api.types.is_integer_dtype(values):
        if values.max() > np.iinfo(np.int64).max:
            raise ValueError("Die Spalte enthält Zahlen, die größer als 2⁶³ − 1 sind. Unterstützt wird der Wertebereich von int64.")
        return UploadedKeys(np.ascontiguousarray(values.to_numpy(dtype=np.int64)), skipped)

    floats = values.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(floats)):
        raise ValueError("Die Spalte enthält unendliche Werte (inf).")
    if not np.all(floats == np.round(floats)):
        raise ValueError("Die Spalte enthält Nachkommastellen. Unterstützt werden nur ganze Zahlen (siehe Annahme A2).")
    if np.any(np.abs(floats) >= INT64_LIMIT):
        raise ValueError("Die Spalte enthält Zahlen außerhalb des Wertebereichs von int64 (±9,2 · 10¹⁸).")
    return UploadedKeys(np.ascontiguousarray(floats.astype(np.int64)), skipped)
