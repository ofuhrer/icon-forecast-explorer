"""Series-caching subsystem for ICON Forecast Explorer.

Thread-safe in-memory cache for /api/series responses, keyed by dataset, variable,
init time, requested types, lat/lon (snapped to grid cell), and time operator.

Note on test-patching: `_series_prune_orphan_locks`, `_series_cache_put`, and
`_series_cache_get` resolve `SERIES_CACHE_MAX_ENTRIES` / `SERIES_CACHE_TTL_SECONDS`
at call time via `_resolve_max_entries()` / `_resolve_ttl()`.  These helpers check
``sys.modules['app']`` first so that ``unittest.mock.patch.object(app_module, ...)``
patches are honoured transparently.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
from fastapi import HTTPException

from grid_utils import _coord_key, _dataset_grid_shape, _effective_grid_bounds

SERIES_CACHE_TTL_SECONDS = float(os.getenv("SERIES_CACHE_TTL_SECONDS", "1200"))
SERIES_CACHE_MAX_ENTRIES = int(os.getenv("SERIES_CACHE_MAX_ENTRIES", "512"))
SERIES_LATLON_BUCKET_DEG = float(os.getenv("SERIES_LATLON_BUCKET_DEG", "0.01"))

_SERIES_CACHE: Dict[tuple, tuple[float, Dict[str, object]]] = {}
_SERIES_CACHE_GUARD = threading.Lock()
_SERIES_KEY_LOCKS: Dict[tuple, threading.Lock] = {}
_SERIES_KEY_LOCKS_GUARD = threading.Lock()


def _resolve_max_entries() -> int:
    """Return SERIES_CACHE_MAX_ENTRIES, preferring the app module's value.

    This indirection lets ``unittest.mock.patch.object(app_module,
    "SERIES_CACHE_MAX_ENTRIES", N)`` take effect inside this module.
    """
    _app = sys.modules.get("app")
    if _app is not None:
        v = getattr(_app, "SERIES_CACHE_MAX_ENTRIES", None)
        if v is not None:
            return int(v)
    return SERIES_CACHE_MAX_ENTRIES


def _resolve_ttl() -> float:
    """Return SERIES_CACHE_TTL_SECONDS, preferring the app module's value."""
    _app = sys.modules.get("app")
    if _app is not None:
        v = getattr(_app, "SERIES_CACHE_TTL_SECONDS", None)
        if v is not None:
            return float(v)
    return SERIES_CACHE_TTL_SECONDS


def _series_spatial_cache_token(dataset_id: str, lat: float, lon: float, store=None) -> tuple:
    # Cache by sampled grid point to preserve value correctness while still
    # allowing reuse across nearby request coordinates that map to the same cell.
    bounds = _effective_grid_bounds(dataset_id, store=store)
    shape = _dataset_grid_shape(dataset_id, store=store)
    if shape is None:
        return ("coord", _coord_key(lat), _coord_key(lon))
    try:
        min_lon = float(bounds["min_lon"])
        max_lon = float(bounds["max_lon"])
        min_lat = float(bounds["min_lat"])
        max_lat = float(bounds["max_lat"])
    except (TypeError, ValueError, KeyError):
        return ("coord", _coord_key(lat), _coord_key(lon))
    lon_span = max(1e-9, max_lon - min_lon)
    lat_span = max(1e-9, max_lat - min_lat)
    w, h = shape
    x = int(np.clip(round((float(lon) - min_lon) / lon_span * (w - 1)), 0, w - 1))
    y = int(np.clip(round((max_lat - float(lat)) / lat_span * (h - 1)), 0, h - 1))
    return ("grid", x, y)


def _series_cache_key(
    dataset_id: str,
    variable_id: str,
    init: str,
    req_types: List[str],
    cached_only: bool,
    lat: float,
    lon: float,
    time_operator: str,
    store=None,
) -> tuple:
    spatial_token = _series_spatial_cache_token(dataset_id, lat, lon, store=store)
    return (
        dataset_id,
        variable_id,
        init,
        tuple(req_types),
        bool(cached_only),
        spatial_token,
        str(time_operator),
    )


def _series_cache_get(key: tuple) -> Dict[str, object] | None:
    now = time.monotonic()
    ttl = _resolve_ttl()
    with _SERIES_CACHE_GUARD:
        item = _SERIES_CACHE.get(key)
        if item is None:
            return None
        ts, payload = item
        if now - ts > ttl:
            _SERIES_CACHE.pop(key, None)
            _series_prune_orphan_locks()
            return None
        return payload


def _series_cache_put(key: tuple, payload: Dict[str, object]) -> None:
    max_entries = _resolve_max_entries()
    now = time.monotonic()
    with _SERIES_CACHE_GUARD:
        _SERIES_CACHE[key] = (now, payload)
        if len(_SERIES_CACHE) <= max_entries:
            _series_prune_orphan_locks()
            return
        oldest_key = min(_SERIES_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _SERIES_CACHE.pop(oldest_key, None)
        _series_prune_orphan_locks()


def _series_key_lock(key: tuple) -> threading.Lock:
    with _SERIES_KEY_LOCKS_GUARD:
        lock = _SERIES_KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SERIES_KEY_LOCKS[key] = lock
        return lock


def _series_prune_orphan_locks() -> None:
    # Keep lock map bounded by dropping locks that no longer have cache entries.
    max_entries = _resolve_max_entries()
    with _SERIES_KEY_LOCKS_GUARD:
        if len(_SERIES_KEY_LOCKS) <= max(64, max_entries * 2):
            return
        active_keys = set(_SERIES_CACHE.keys())
        stale = [k for k in _SERIES_KEY_LOCKS.keys() if k not in active_keys]
        for key in stale:
            _SERIES_KEY_LOCKS.pop(key, None)


def _series_payload_for_request(
    payload: Dict[str, object],
    lat: float,
    lon: float,
    cache_hit: bool,
) -> Dict[str, object]:
    out = dict(payload)
    out["lat"] = lat
    out["lon"] = lon
    diag = dict(out.get("diagnostics", {}))
    diag["cache_hit"] = bool(cache_hit)
    out["diagnostics"] = diag
    return out


def _build_series_payload(
    dataset_id: str,
    variable_id: str,
    init: str,
    lat: float,
    lon: float,
    req_types: List[str],
    cached_only: bool,
    time_operator: str,
    store,
) -> Dict[str, object]:
    leads = store.lead_hours_for_init(dataset_id, init)
    if not leads:
        raise HTTPException(status_code=400, detail=f"No leads available for init {init}")
    try:
        base_dt = datetime.strptime(init, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid init format: {init}") from exc

    values_by_type: Dict[str, List[float | None]] = {t: [] for t in req_types}
    missing_counts: Dict[str, int] = {t: 0 for t in req_types}
    errors: List[Dict[str, object]] = []
    for lead in leads:
        for t in req_types:
            try:
                if cached_only:
                    v = store.get_cached_value(
                        dataset_id,
                        variable_id,
                        init,
                        lead,
                        lat=lat,
                        lon=lon,
                        type_id=t,
                        time_operator=time_operator,
                    )
                else:
                    v = store.get_value(
                        dataset_id,
                        variable_id,
                        init,
                        lead,
                        lat=lat,
                        lon=lon,
                        type_id=t,
                        time_operator=time_operator,
                    )
                if v is None:
                    missing_counts[t] += 1
                    values_by_type[t].append(None)
                else:
                    values_by_type[t].append(round(float(v), 2))
            except (RuntimeError, ValueError) as exc:
                missing_counts[t] += 1
                values_by_type[t].append(None)
                errors.append(
                    {
                        "type_id": t,
                        "lead_hour": int(lead),
                        "message": str(exc),
                    }
                )
    valid_times_utc = [(base_dt + timedelta(hours=int(lead))).strftime("%Y-%m-%dT%H:%M:%SZ") for lead in leads]
    return {
        "dataset_id": dataset_id,
        "variable_id": variable_id,
        "init": init,
        "lat": lat,
        "lon": lon,
        "lead_hours": [int(v) for v in leads],
        "valid_times_utc": valid_times_utc,
        "values": values_by_type,
        "time_operator": time_operator,
        "diagnostics": {
            "cached_only": cached_only,
            "missing_counts": missing_counts,
            "errors": errors[:25],
            "error_count": len(errors),
        },
    }
