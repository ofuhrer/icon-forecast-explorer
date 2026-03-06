"""Tile rendering and colormap utilities for ICON Forecast Explorer.

Pure functions with no FastAPI dependency; safe to import anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

TILE_SIZE = 256
COLORMAP_MANIFEST_PATH = Path(__file__).resolve().parent / "colormaps.json"


def _load_project_colormaps(
    path: Path,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    if not path.exists():
        return {}, {}

    payload = json.loads(path.read_text())
    registry: Dict[str, Dict[str, object]] = {}
    var_to_cmap: Dict[str, str] = {}

    for entry in payload.get("colormaps", []):
        cmap_id = entry.get("id")
        levels = entry.get("levels", [])
        colors = entry.get("colors", [])
        variables = entry.get("variables", [])
        if not cmap_id or not isinstance(levels, list) or not isinstance(colors, list):
            continue
        if len(colors) != len(levels) + 1:
            continue

        level_arr = np.array(levels, dtype=np.float32)
        color_arr = np.array(colors, dtype=np.uint8)
        if color_arr.ndim != 2 or color_arr.shape[1] != 3:
            continue

        registry[cmap_id] = {
            "id": cmap_id,
            "name": entry.get("name", cmap_id),
            "unit": entry.get("unit", ""),
            "levels": level_arr,
            "colors": color_arr,
        }
        for variable_id in variables:
            var_to_cmap[str(variable_id)] = cmap_id

    return registry, var_to_cmap


COLORMAP_REGISTRY, VARIABLE_TO_COLORMAP = _load_project_colormaps(COLORMAP_MANIFEST_PATH)


def _colormap_for_variable(variable_id: str) -> Dict[str, object] | None:
    cmap_id = VARIABLE_TO_COLORMAP.get(variable_id)
    if cmap_id is None:
        return None
    return COLORMAP_REGISTRY.get(cmap_id)


def _legend_for_variable(variable_id: str) -> Dict[str, object] | None:
    cmap = _colormap_for_variable(variable_id)
    if cmap is not None:
        return {
            "type": "discrete",
            "thresholds": [float(v) for v in cmap["levels"].tolist()],
            "colors": [[int(c) for c in row] for row in cmap["colors"].tolist()],
        }
    return None


def apply_colormap(values: np.ndarray, variable_id: str) -> np.ndarray:
    cmap = _colormap_for_variable(variable_id)
    if cmap is not None:
        thresholds = cmap["levels"]
        colors = cmap["colors"]
        idx = np.digitize(values, thresholds, right=False)
        idx = np.clip(idx, 0, colors.shape[0] - 1)
        return colors[idx]

    min_v, max_v = float(np.nanmin(values)), float(np.nanmax(values))
    if max_v <= min_v:
        max_v = min_v + 1.0

    normalized = np.clip((values - min_v) / (max_v - min_v), 0.0, 1.0)
    r = np.interp(normalized, [0.0, 0.35, 0.7, 1.0], [35, 57, 242, 203])
    g = np.interp(normalized, [0.0, 0.35, 0.7, 1.0], [63, 145, 201, 24])
    b = np.interp(normalized, [0.0, 0.35, 0.7, 1.0], [140, 228, 75, 29])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def render_tile_rgba(
    field: np.ndarray,
    z: int,
    x: int,
    y: int,
    variable_id: str,
    bounds: Dict[str, float],
) -> np.ndarray:
    px, py = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE))
    global_x = x * TILE_SIZE + px
    global_y = y * TILE_SIZE + py

    world_size = TILE_SIZE * (2**z)
    lon = global_x / world_size * 360.0 - 180.0
    lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * global_y / world_size))))

    lat_mask = (lat >= bounds["min_lat"]) & (lat <= bounds["max_lat"])
    lon_mask = (lon >= bounds["min_lon"]) & (lon <= bounds["max_lon"])
    domain_mask = lat_mask & lon_mask

    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    if not np.any(domain_mask):
        return rgba

    h, w = field.shape
    lon_span = max(1e-9, float(bounds["max_lon"] - bounds["min_lon"]))
    lat_span = max(1e-9, float(bounds["max_lat"] - bounds["min_lat"]))
    # Bilinear sampling reduces zoom-level aliasing compared to nearest-cell lookup.
    x_f = np.clip((lon - bounds["min_lon"]) / lon_span * (w - 1), 0.0, float(w - 1))
    y_f = np.clip((bounds["max_lat"] - lat) / lat_span * (h - 1), 0.0, float(h - 1))
    x0 = np.floor(x_f).astype(np.int32)
    y0 = np.floor(y_f).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = x_f - x0
    dy = y_f - y0

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]
    sampled = v00 * (1.0 - dx) * (1.0 - dy) + v10 * dx * (1.0 - dy) + v01 * (1.0 - dx) * dy + v11 * dx * dy
    colors = apply_colormap(sampled, variable_id)

    rgba[domain_mask, :3] = colors[domain_mask]
    rgba[domain_mask, 3] = 170
    return rgba
