"""Record the measurement environment and store results reproducibly.

Every result file documents the hardware and software it was measured on
[hoefler2015]. Results are stored as CSV; the environment is written as
comment lines (``# key: value``) at the top of the same file, so a single file
contains everything needed to interpret it. :func:`read_results` reads both.
"""

import datetime as dt
import os
import platform
import subprocess
from pathlib import Path

import numba
import numpy as np
import pandas as pd


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    if platform.system() == "Darwin":
        value = _run(["sysctl", "-n", "hw.memsize"])
        return int(value) if value else None
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    return None


def collect_environment() -> dict[str, str]:
    """Describe hardware, software and code version of the current run."""
    ram = _ram_bytes()
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu": _cpu_name(),
        "cpu_count": str(os.cpu_count()),
        "ram_gb": f"{ram / 2**30:.1f}" if ram else "unknown",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "pandas": pd.__version__,
        "git_commit": (commit or "unknown") + (" (uncommitted changes)" if dirty else ""),
    }


def write_results(frame: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    """Write ``frame`` as CSV, preceded by ``metadata`` as ``# key: value`` lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for key, value in metadata.items():
            file.write(f"# {key}: {value}\n")
        frame.to_csv(file, index=False)


def read_results(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a result file written by :func:`write_results`."""
    metadata = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.startswith("# "):
                break
            key, value = line[2:].rstrip("\n").split(": ", 1)
            metadata[key] = value
    return pd.read_csv(path, comment="#"), metadata
