"""Coordinate-transform and grid-shape utilities for ICON Forecast Explorer.

Pure mathematical helpers (_lon_to_x, _lat_to_y, _x_to_lon, _y_to_lat, _coord_key) plus
thin store-aware helpers (_dataset_grid_shape, _effective_grid_bounds) that accept an
optional *store* argument so callers can inject the live or test-patched store instance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from weather_data import SWISS_BOUNDS


def _coord_key(value: float) -> float:
    return round(float(value), 6)


def _lon_to_x(lon: float, width: int, bounds: Dict[str, float]) -> int:
    lon_span = max(1e-9, float(bounds["max_lon"] - bounds["min_lon"]))
    frac = (lon - bounds["min_lon"]) / lon_span
    return int(np.clip(round(frac * (width - 1)), 0, width - 1))


def _lat_to_y(lat: float, height: int, bounds: Dict[str, float]) -> int:
    lat_span = max(1e-9, float(bounds["max_lat"] - bounds["min_lat"]))
    frac = (bounds["max_lat"] - lat) / lat_span
    return int(np.clip(round(frac * (height - 1)), 0, height - 1))


def _x_to_lon(x: int, width: int, bounds: Dict[str, float]) -> float:
    frac = float(x) / float(max(1, width - 1))
    return bounds["min_lon"] + frac * (bounds["max_lon"] - bounds["min_lon"])


def _y_to_lat(y: int, height: int, bounds: Dict[str, float]) -> float:
    frac = float(y) / float(max(1, height - 1))
    return bounds["max_lat"] - frac * (bounds["max_lat"] - bounds["min_lat"])


def _dataset_grid_shape(dataset_id: str, store=None) -> tuple[int, int] | None:
    target_shape_fn = getattr(store, "target_grid_shape", None)
    if callable(target_shape_fn):
        try:
            shape = target_shape_fn(dataset_id)
            if isinstance(shape, (tuple, list)) and len(shape) == 2:
                w = int(shape[0] or 0)
                h = int(shape[1] or 0)
                if w > 1 and h > 1:
                    return (w, h)
        except Exception:
            pass
    metas = getattr(store, "dataset_metas", None)
    if not metas:
        return None
    for ds in metas:
        if getattr(ds, "dataset_id", None) == dataset_id:
            w = int(getattr(ds, "target_grid_width", 0) or 0)
            h = int(getattr(ds, "target_grid_height", 0) or 0)
            if w > 1 and h > 1:
                return (w, h)
    return None


def _effective_grid_bounds(dataset_id: str | None = None, store=None) -> Dict[str, float]:
    grid_bounds_fn = getattr(store, "grid_bounds", None)
    if callable(grid_bounds_fn):
        try:
            if dataset_id is None:
                bounds = grid_bounds_fn()
            else:
                try:
                    bounds = grid_bounds_fn(dataset_id)
                except TypeError:
                    bounds = grid_bounds_fn()
            if isinstance(bounds, dict):
                return bounds
        except Exception:
            pass
    return dict(SWISS_BOUNDS)
