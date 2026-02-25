from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from weather_data import SWISS_BOUNDS, ForecastStore

TILE_SIZE = 256
COLORMAP_MANIFEST_PATH = Path(__file__).resolve().parent / "colormaps" / "manifest.json"


app = FastAPI(title="ICON Forecast Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ForecastStore()
COLORMAP_REGISTRY = {}
VARIABLE_TO_COLORMAP = {}
FORECAST_TYPES = [
    {"type_id": "control", "display_name": "Control"},
    {"type_id": "mean", "display_name": "Mean"},
    {"type_id": "median", "display_name": "Median"},
    {"type_id": "p10", "display_name": "10% Percentile"},
    {"type_id": "p90", "display_name": "90% Percentile"},
]


@app.on_event("startup")
def _startup() -> None:
    store.start_background_prewarm()


@app.on_event("shutdown")
def _shutdown() -> None:
    store.stop_background_prewarm()


@app.get("/api/metadata")
def metadata() -> Dict[str, object]:
    # Return immediately with current catalog snapshot; refresh in background.
    store.refresh_catalog(force=False, blocking=False)
    datasets_payload = []
    for ds in store.dataset_metas:
        init_times = store.init_times(ds.dataset_id)
        init_to_leads = store.init_to_leads(ds.dataset_id)
        expected_init_to_leads = {
            init: store.expected_lead_hours_for_init(ds.dataset_id, init) for init in init_times
        }
        default_leads = store.lead_hours_for_init(ds.dataset_id, init_times[0]) if init_times else []
        variables = []
        for var in store.variable_metas:
            variables.append(
                {
                    "variable_id": var.variable_id,
                    "display_name": var.display_name,
                    "unit": var.unit,
                    "range": [var.min_value, var.max_value],
                    "default_colormap": VARIABLE_TO_COLORMAP.get(var.variable_id, "default"),
                    "legend": _legend_for_variable(var.variable_id),
                }
            )
        datasets_payload.append(
            {
                "dataset_id": ds.dataset_id,
                "display_name": ds.display_name,
                "types": FORECAST_TYPES,
                "refresh": store.refresh_status(ds.dataset_id),
                "variables": variables,
                "lead_hours": default_leads,
                "init_to_leads": init_to_leads,
                "expected_init_to_leads": expected_init_to_leads,
                "init_times": init_times,
            }
        )

    first = datasets_payload[0] if datasets_payload else {"init_times": [], "lead_hours": [], "init_to_leads": {}}
    return {
        "datasets": datasets_payload,
        "init_times": first["init_times"],
        "lead_hours": first["lead_hours"],
        "init_to_leads": first["init_to_leads"],
        "bounds": SWISS_BOUNDS,
    }


@app.get("/api/value")
def value(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> Dict[str, object]:
    try:
        val = store.get_value(dataset_id, variable_id, init, lead, lat=lat, lon=lon, type_id=type_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "dataset_id": dataset_id,
        "type_id": type_id,
        "variable_id": variable_id,
        "init": init,
        "lead": lead,
        "lat": lat,
        "lon": lon,
        "value": round(val, 2),
    }


@app.get("/api/prefetch")
def prefetch(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
) -> Dict[str, object]:
    try:
        store.get_field(dataset_id, variable_id, init, lead, type_id=type_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/series")
def series(
    dataset_id: str = Query(...),
    variable_id: str = Query(...),
    init: str = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    types: str = Query("control,mean,median,p10,p90"),
    cached_only: bool = Query(True),
) -> Dict[str, object]:
    leads = store.lead_hours_for_init(dataset_id, init)
    if not leads:
        raise HTTPException(status_code=400, detail=f"No leads available for init {init}")

    req_types: List[str] = [t.strip() for t in types.split(",") if t.strip()]
    if not req_types:
        req_types = ["control"]

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
                        dataset_id, variable_id, init, lead, lat=lat, lon=lon, type_id=t
                    )
                else:
                    v = store.get_value(dataset_id, variable_id, init, lead, lat=lat, lon=lon, type_id=t)
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

    valid_times_utc = [
        (base_dt + timedelta(hours=int(lead))).strftime("%Y-%m-%dT%H:%M:%SZ")
        for lead in leads
    ]
    return {
        "dataset_id": dataset_id,
        "variable_id": variable_id,
        "init": init,
        "lat": lat,
        "lon": lon,
        "lead_hours": [int(v) for v in leads],
        "valid_times_utc": valid_times_utc,
        "values": values_by_type,
        "diagnostics": {
            "cached_only": cached_only,
            "missing_counts": missing_counts,
            "errors": errors[:25],
            "error_count": len(errors),
        },
    }


@app.get("/api/tiles/{dataset_id}/{variable_id}/{init}/{lead}/{z}/{x}/{y}.png")
def tiles(
    dataset_id: str,
    variable_id: str,
    init: str,
    lead: int,
    z: int,
    x: int,
    y: int,
    type_id: str = Query("control"),
) -> Response:
    try:
        field = store.get_field(dataset_id, variable_id, init, lead, type_id=type_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rgba = render_tile_rgba(field, z, x, y, variable_id)
    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


def render_tile_rgba(field: np.ndarray, z: int, x: int, y: int, variable_id: str) -> np.ndarray:
    px, py = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE))
    global_x = x * TILE_SIZE + px
    global_y = y * TILE_SIZE + py

    world_size = TILE_SIZE * (2**z)
    lon = global_x / world_size * 360.0 - 180.0
    lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * global_y / world_size))))

    lat_mask = (lat >= SWISS_BOUNDS["min_lat"]) & (lat <= SWISS_BOUNDS["max_lat"])
    lon_mask = (lon >= SWISS_BOUNDS["min_lon"]) & (lon <= SWISS_BOUNDS["max_lon"])
    swiss_mask = lat_mask & lon_mask

    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    if not np.any(swiss_mask):
        return rgba

    h, w = field.shape
    x_idx = np.clip(
        np.round((lon - SWISS_BOUNDS["min_lon"]) / (SWISS_BOUNDS["max_lon"] - SWISS_BOUNDS["min_lon"]) * (w - 1)).astype(int),
        0,
        w - 1,
    )
    y_idx = np.clip(
        np.round((SWISS_BOUNDS["max_lat"] - lat) / (SWISS_BOUNDS["max_lat"] - SWISS_BOUNDS["min_lat"]) * (h - 1)).astype(int),
        0,
        h - 1,
    )

    sampled = field[y_idx, x_idx]
    colors = apply_colormap(sampled, variable_id)

    rgba[swiss_mask, :3] = colors[swiss_mask]
    rgba[swiss_mask, 3] = 170
    return rgba


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


def _legend_for_variable(variable_id: str) -> Dict[str, object] | None:
    cmap = _colormap_for_variable(variable_id)
    if cmap is not None:
        return {
            "type": "discrete",
            "thresholds": [float(v) for v in cmap["levels"].tolist()],
            "colors": [[int(c) for c in row] for row in cmap["colors"].tolist()],
        }
    return None


def _colormap_for_variable(variable_id: str) -> Dict[str, object] | None:
    cmap_id = VARIABLE_TO_COLORMAP.get(variable_id)
    if cmap_id is None:
        return None
    return COLORMAP_REGISTRY.get(cmap_id)


def _load_project_colormaps(path: Path) -> tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
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
