"""Standalone cache file I/O helpers for the ICON Forecast Explorer.

All functions in this module are pure file-system operations that do not depend
on the :class:`~weather_data.ForecastStore` instance or any other application
state.  They are extracted here so they can be unit-tested and reused without
importing the full ``weather_data`` module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

LOGGER = logging.getLogger("icon_forecast.weather_cache")


# ---------------------------------------------------------------------------
# Field cache file I/O
# ---------------------------------------------------------------------------


def load_cached_field_file(path: Path) -> np.ndarray | None:
    """Load a compressed field array from *path*, returning ``None`` on error."""
    try:
        return np.load(path)["field"]
    except (EOFError, OSError, zipfile.BadZipFile, KeyError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        LOGGER.warning("Dropped corrupt/incomplete field cache file path=%s", path)
        return None


def save_cached_field_file(path: Path, field: np.ndarray) -> None:
    """Atomically write *field* as a compressed ``.npz`` file to *path*."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as tmp_file:
        np.savez_compressed(tmp_file, field=field)
    os.replace(tmp_path, path)
    LOGGER.debug(
        "Saved field cache path=%s bytes=%s",
        path,
        path.stat().st_size if path.exists() else -1,
    )


# ---------------------------------------------------------------------------
# Wind vector cache file I/O
# ---------------------------------------------------------------------------


def load_cached_wind_vector_file(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    """Load ``(u, v)`` wind arrays from *path*, returning ``None`` on error."""
    try:
        payload = np.load(path)
        return payload["u"], payload["v"]
    except (EOFError, OSError, zipfile.BadZipFile, KeyError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        LOGGER.warning("Dropped corrupt/incomplete wind vector cache file path=%s", path)
        return None


def save_cached_wind_vector_file(path: Path, u_field: np.ndarray, v_field: np.ndarray) -> None:
    """Atomically write *(u, v)* arrays as a compressed ``.npz`` file to *path*."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as tmp_file:
        np.savez_compressed(tmp_file, u=u_field, v=v_field)
    os.replace(tmp_path, path)
    LOGGER.debug(
        "Saved wind vector cache path=%s bytes=%s",
        path,
        path.stat().st_size if path.exists() else -1,
    )


# ---------------------------------------------------------------------------
# Field debug info I/O
# ---------------------------------------------------------------------------


def field_debug_path(field_cache_path: Path) -> Path:
    """Return the sidecar ``.json`` path for *field_cache_path*."""
    return field_cache_path.with_suffix(".json")


def load_field_debug_info(field_cache_path: Path) -> Dict[str, object] | None:
    """Read and return the field debug JSON sidecar, or ``None`` if absent/invalid."""
    path = field_debug_path(field_cache_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def save_field_debug_info(field_cache_path: Path, debug_info: Dict[str, object]) -> None:
    """Write *debug_info* as JSON to the field debug sidecar path."""
    path = field_debug_path(field_cache_path)
    try:
        path.write_text(json.dumps(debug_info))
    except OSError:
        LOGGER.warning("Failed to write field debug info path=%s", path)


# ---------------------------------------------------------------------------
# Filesystem utilities
# ---------------------------------------------------------------------------


def safe_unlink(path: Path) -> None:
    """Delete *path*, ignoring ``OSError``."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def parse_iso_duration_hours(value: str) -> int | None:
    """Parse an ISO-8601 duration string and return the total hours as an int."""
    match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return None
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
    return int(round(total_hours))


def init_to_iso(init_str: str) -> str:
    """Convert an ``YYYYMMDDhh`` init string to an ISO-8601 UTC timestamp."""
    dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")
