from __future__ import annotations

from io import BytesIO
import json
import logging
import os
import threading
import time
import math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from weather_data import ForecastStore, SUPPORTED_TIME_OPERATORS, TIME_OPERATORS, SWISS_BOUNDS

TILE_SIZE = 256
COLORMAP_MANIFEST_PATH = Path(__file__).resolve().parent / "colormaps.json"
SERIES_CACHE_TTL_SECONDS = float(os.getenv("SERIES_CACHE_TTL_SECONDS", "1200"))
SERIES_CACHE_MAX_ENTRIES = int(os.getenv("SERIES_CACHE_MAX_ENTRIES", "512"))
SERIES_LATLON_BUCKET_DEG = float(os.getenv("SERIES_LATLON_BUCKET_DEG", "0.01"))


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", os.getenv("ICON_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("icon_forecast")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    log_file = os.getenv("ICON_LOG_FILE", "logs/icon_forecast.log").strip()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    logger.info("Logger configured level=%s file=%s", logging.getLevelName(level), log_file or "disabled")
    return logger


LOGGER = _configure_logging()


app = FastAPI(title="ICON Forecast Explorer")


def _allowed_cors_origins() -> List[str]:
    if os.getenv("ICON_EXPLORER_ALLOW_ALL_CORS", "").strip() == "1":
        return ["*"]
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
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
FORECAST_TYPE_IDS = {item["type_id"] for item in FORECAST_TYPES}
SERIES_FORECAST_TYPE_IDS = FORECAST_TYPE_IDS | {"min", "max"}
_SERIES_CACHE: Dict[tuple, tuple[float, Dict[str, object]]] = {}
_SERIES_CACHE_GUARD = threading.Lock()
_SERIES_KEY_LOCKS: Dict[tuple, threading.Lock] = {}
_SERIES_KEY_LOCKS_GUARD = threading.Lock()


def _validate_time_operator(value: str) -> str:
    if not isinstance(value, str):
        default = getattr(value, "default", None)
        if isinstance(default, str):
            value = default
    time_operator = str(value or "none").strip() or "none"
    if time_operator not in SUPPORTED_TIME_OPERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown time_operator: {time_operator}")
    return time_operator


def _parse_requested_types(types: str) -> List[str]:
    req_types = [t.strip() for t in str(types).split(",") if t.strip()]
    if not req_types:
        req_types = ["control"]
    unknown_types = [t for t in req_types if t not in SERIES_FORECAST_TYPE_IDS]
    if unknown_types:
        raise HTTPException(status_code=400, detail=f"Unknown type_id values: {', '.join(unknown_types)}")
    return req_types


def _parse_requested_variables(variables: str) -> List[str]:
    req_variables = [v.strip() for v in str(variables).split(",") if v.strip()]
    if not req_variables:
        raise HTTPException(status_code=400, detail="No variables requested")
    known_variables = {meta.variable_id for meta in getattr(store, "variable_metas", [])}
    if known_variables:
        unknown_variables = [v for v in req_variables if v not in known_variables]
        if unknown_variables:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown variable_id values: {', '.join(unknown_variables)}",
            )
    # preserve request order while removing duplicates
    out: List[str] = []
    for variable_id in req_variables:
        if variable_id not in out:
            out.append(variable_id)
    return out


@app.on_event("startup")
def _startup() -> None:
    LOGGER.info("App startup")
    store.start_background_prewarm()


@app.on_event("shutdown")
def _shutdown() -> None:
    LOGGER.info("App shutdown")
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
            grib_name_fn = getattr(store, "variable_grib_name", None)
            long_name_fn = getattr(store, "variable_long_name", None)
            standard_unit_fn = getattr(store, "variable_standard_unit", None)
            grib_name = (
                grib_name_fn(var.variable_id)
                if callable(grib_name_fn)
                else str(var.ogd_variable or var.variable_id).upper()
            )
            long_name = long_name_fn(var.variable_id) if callable(long_name_fn) else var.display_name
            standard_unit = standard_unit_fn(var.variable_id) if callable(standard_unit_fn) else var.unit
            variables.append(
                {
                    "variable_id": var.variable_id,
                    "display_name": var.display_name,
                    "long_name": long_name,
                    "grib_name": grib_name,
                    "standard_unit": standard_unit,
                    "display_unit": var.unit,
                    "unit": var.unit,
                    "range": [var.min_value, var.max_value],
                    "lead_time_display_offset_hours": int(store.variable_lead_display_offset_hours(var.variable_id)),
                    "default_colormap": VARIABLE_TO_COLORMAP.get(var.variable_id, "default"),
                    "legend": _legend_for_variable(var.variable_id),
                }
            )
        datasets_payload.append(
            {
                "dataset_id": ds.dataset_id,
                "display_name": ds.display_name,
                "target_grid_width": int(ds.target_grid_width),
                "target_grid_height": int(ds.target_grid_height),
                "types": FORECAST_TYPES,
                "time_operators": TIME_OPERATORS,
                "refresh": store.refresh_status(ds.dataset_id),
                "variables": variables,
                "lead_hours": default_leads,
                "init_to_leads": init_to_leads,
                "expected_init_to_leads": expected_init_to_leads,
                "init_times": init_times,
            }
        )

    first = datasets_payload[0] if datasets_payload else {"init_times": [], "lead_hours": [], "init_to_leads": {}}
    LOGGER.debug("Metadata served datasets=%d", len(datasets_payload))
    bounds = _effective_grid_bounds()
    return {
        "datasets": datasets_payload,
        "time_operators": TIME_OPERATORS,
        "init_times": first["init_times"],
        "lead_hours": first["lead_hours"],
        "init_to_leads": first["init_to_leads"],
        "bounds": bounds,
    }


@app.get("/api/value")
def value(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        val = store.get_cached_value(
            dataset_id,
            variable_id,
            init,
            lead,
            lat=lat,
            lon=lon,
            type_id=type_id,
            time_operator=time_operator,
        )
        if val is None:
            queued = store.queue_field_fetch(
                dataset_id, variable_id, init, lead, type_id=type_id, time_operator=time_operator
            )
            LOGGER.debug(
                "Value cache miss dataset=%s type=%s var=%s init=%s lead=%s queued=%s",
                dataset_id,
                type_id,
                variable_id,
                init,
                lead,
                queued,
            )
            raise HTTPException(status_code=503, detail="Value not cached yet")
    except ValueError as exc:
        LOGGER.warning("Value request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.warning("Value request runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "dataset_id": dataset_id,
        "type_id": type_id,
        "variable_id": variable_id,
        "init": init,
        "lead": lead,
        "time_operator": time_operator,
        "lat": lat,
        "lon": lon,
        "value": round(val, 2),
    }


@app.get("/api/model-elevation")
def model_elevation(
    dataset_id: str = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> Dict[str, object]:
    try:
        elevation_m = store.get_model_elevation(dataset_id, lat=lat, lon=lon)
    except ValueError as exc:
        LOGGER.warning("Model elevation request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.debug("Model elevation request runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("Model elevation request unexpected error: %s", exc)
        raise HTTPException(status_code=503, detail="Model elevation unavailable") from exc
    return {
        "dataset_id": dataset_id,
        "lat": lat,
        "lon": lon,
        "model_elevation_m": round(float(elevation_m), 2),
    }


@app.get("/api/prefetch")
def prefetch(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        queued = store.queue_field_fetch(
            dataset_id, variable_id, init, lead, type_id=type_id, time_operator=time_operator
        )
        LOGGER.debug(
            "Prefetch request dataset=%s type=%s var=%s init=%s lead=%s queued=%s",
            dataset_id,
            type_id,
            variable_id,
            init,
            lead,
            queued,
        )
    except ValueError as exc:
        LOGGER.warning("Prefetch request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.warning("Prefetch request runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "queued": bool(queued)}


@app.get("/api/meteogram-warmup/start")
def meteogram_warmup_start(
    dataset_id: str = Query(...),
    init: str = Query(...),
    variables: str = Query("clct,tot_prec,vmax_10m,t_2m"),
    types: str = Query("control,median,p10,p90,min,max"),
    time_operator: str = Query("none"),
) -> Dict[str, object]:
    try:
        req_variables = _parse_requested_variables(variables)
        req_types = _parse_requested_types(types)
        time_operator = _validate_time_operator(time_operator)
        payload = store.start_meteogram_warmup(
            dataset_id=dataset_id,
            init_str=init,
            variable_ids=req_variables,
            type_ids=req_types,
            time_operator=time_operator,
        )
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/meteogram-warmup/status")
def meteogram_warmup_status(job_id: str = Query(...)) -> Dict[str, object]:
    try:
        payload = store.get_meteogram_warmup(job_id)
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown warmup job_id: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/field-debug")
def field_debug(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        info = store.get_cached_field_debug_info(
            dataset_id=dataset_id,
            variable_id=variable_id,
            init_str=init,
            lead_hour=lead,
            type_id=type_id,
            time_operator=time_operator,
        )
    except ValueError as exc:
        LOGGER.warning("Field debug request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if info is None:
        failure = store.get_field_failure(
            dataset_id=dataset_id,
            variable_id=variable_id,
            init_str=init,
            lead_hour=lead,
            type_id=type_id,
            time_operator=time_operator,
        )
        if failure is not None:
            return {
                "dataset_id": dataset_id,
                "type_id": type_id,
                "variable_id": variable_id,
                "init": init,
                "lead": lead,
                "time_operator": time_operator,
                "status": "error",
                "debug": failure,
            }
        LOGGER.debug(
            "Field debug cache miss dataset=%s type=%s var=%s init=%s lead=%s",
            dataset_id,
            type_id,
            variable_id,
            init,
            lead,
        )
        store.queue_field_fetch(dataset_id, variable_id, init, lead, type_id=type_id, time_operator=time_operator)
        return {
            "dataset_id": dataset_id,
            "type_id": type_id,
            "variable_id": variable_id,
            "init": init,
            "lead": lead,
            "time_operator": time_operator,
            "status": "loading",
            "debug": None,
        }
    return {
        "dataset_id": dataset_id,
        "type_id": type_id,
        "variable_id": variable_id,
        "init": init,
        "lead": lead,
        "time_operator": time_operator,
        "status": "ready",
        "debug": info,
    }


@app.get("/api/series")
def series(
    dataset_id: str = Query(...),
    variable_id: str = Query(...),
    init: str = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    types: str = Query("control,mean,median,p10,p90"),
    cached_only: bool = Query(True),
    time_operator: str = Query("none"),
) -> Dict[str, object]:
    req_types = _parse_requested_types(types)
    time_operator = _validate_time_operator(time_operator)

    key = _series_cache_key(dataset_id, variable_id, init, req_types, cached_only, lat, lon, time_operator)
    cached_payload = _series_cache_get(key)
    if cached_payload is not None:
        return _series_payload_for_request(cached_payload, lat=lat, lon=lon, cache_hit=True)

    key_lock = _series_key_lock(key)
    with key_lock:
        cached_payload = _series_cache_get(key)
        if cached_payload is not None:
            return _series_payload_for_request(cached_payload, lat=lat, lon=lon, cache_hit=True)
        payload = _build_series_payload(
            dataset_id=dataset_id,
            variable_id=variable_id,
            init=init,
            lat=lat,
            lon=lon,
            req_types=req_types,
            cached_only=cached_only,
            time_operator=time_operator,
        )
        _series_cache_put(key, payload)
        return _series_payload_for_request(payload, lat=lat, lon=lon, cache_hit=False)


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
    time_operator: str = Query("none"),
) -> Response:
    try:
        time_operator = _validate_time_operator(time_operator)
        field = store.get_cached_field(
            dataset_id, variable_id, init, lead, type_id=type_id, time_operator=time_operator
        )
        if field is None:
            failure = store.get_field_failure(
                dataset_id=dataset_id,
                variable_id=variable_id,
                init_str=init,
                lead_hour=lead,
                type_id=type_id,
                time_operator=time_operator,
            )
            if failure is not None:
                raise RuntimeError(str(failure.get("message", "Asset unavailable")))
            # Keep map interaction responsive even under background queue pressure:
            # fetch requested field synchronously (single-flight guarded in store).
            LOGGER.info(
                "Tile cache miss; sync-fetch dataset=%s type=%s var=%s init=%s lead=%s z=%s x=%s y=%s",
                dataset_id,
                type_id,
                variable_id,
                init,
                lead,
                z,
                x,
                y,
            )
            field = store.get_field(
                dataset_id, variable_id, init, lead, type_id=type_id, time_operator=time_operator
            )
    except ValueError as exc:
        LOGGER.warning("Tile request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.warning("Tile request runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception(
            "Tile request unexpected failure dataset=%s type=%s var=%s init=%s lead=%s z=%s x=%s y=%s",
            dataset_id,
            type_id,
            variable_id,
            init,
            lead,
            z,
            x,
            y,
        )
        raise

    rgba = render_tile_rgba(field, z, x, y, variable_id, _effective_grid_bounds())
    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/wind-vectors")
def wind_vectors(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    min_lat: float = Query(...),
    max_lat: float = Query(...),
    min_lon: float = Query(...),
    max_lon: float = Query(...),
    zoom: float = Query(7.0),
    time_operator: str = Query("none"),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        # Match tile behavior: fetch synchronously on cache miss so vectors
        # appear immediately after switching to wind (no zoom needed).
        if hasattr(store, "get_wind_vectors"):
            u_field, v_field = store.get_wind_vectors(
                dataset_id, init, lead, type_id=type_id, time_operator=time_operator
            )
        else:
            cached = store.get_cached_wind_vectors(
                dataset_id, init, lead, type_id=type_id, time_operator=time_operator
            )
            if cached is None:
                store.queue_wind_vector_fetch(
                    dataset_id, init, lead, type_id=type_id, time_operator=time_operator
                )
                return {"status": "loading", "vectors": [], "unit": "km/h"}
            u_field, v_field = cached
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    z = float(zoom)
    base_stride = int(max(6, min(18, round(18 - z))))
    bounds = _effective_grid_bounds()
    grid_height, grid_width = int(u_field.shape[0]), int(u_field.shape[1])
    y0 = _lat_to_y(max_lat, grid_height, bounds)
    y1 = _lat_to_y(min_lat, grid_height, bounds)
    x0 = _lon_to_x(min_lon, grid_width, bounds)
    x1 = _lon_to_x(max_lon, grid_width, bounds)
    y_start, y_end = sorted((y0, y1))
    x_start, x_end = sorted((x0, x1))
    ny = max(1, y_end - y_start + 1)
    nx = max(1, x_end - x_start + 1)
    # Keep a denser vector field at full-domain views so arrows are visible
    # immediately after switching to wind, without requiring a zoom interaction.
    target_vectors = 2500
    adaptive_stride = int(max(1, math.ceil(math.sqrt((nx * ny) / float(target_vectors)))))
    stride = max(base_stride, adaptive_stride)
    def _sample_vectors(sample_stride: int) -> List[Dict[str, float]]:
        sampled: List[Dict[str, float]] = []
        for yy in range(y_start, y_end + 1, sample_stride):
            for xx in range(x_start, x_end + 1, sample_stride):
                u = float(u_field[yy, xx])
                v = float(v_field[yy, xx])
                speed = math.sqrt(u * u + v * v)
                if not np.isfinite(speed):
                    continue
                lat = _y_to_lat(yy, grid_height, bounds)
                lon = _x_to_lon(xx, grid_width, bounds)
                sampled.append({"lat": lat, "lon": lon, "u": u, "v": v, "speed": speed})
        return sampled

    vectors: List[Dict[str, float]] = _sample_vectors(stride)
    # Robustness: if full-domain sampling is too sparse for the current flow
    # pattern, automatically densify so arrows appear without requiring zoom.
    if len(vectors) < 80 and stride > 1:
        vectors = _sample_vectors(max(1, stride // 2))
    if len(vectors) < 80 and stride > 2:
        vectors = _sample_vectors(1)
    return {"status": "ready", "vectors": vectors, "unit": "km/h"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


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


def render_tile_rgba(field: np.ndarray, z: int, x: int, y: int, variable_id: str, bounds: Dict[str, float]) -> np.ndarray:
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
    x_idx = np.clip(
        np.round((lon - bounds["min_lon"]) / lon_span * (w - 1)).astype(int),
        0,
        w - 1,
    )
    y_idx = np.clip(
        np.round((bounds["max_lat"] - lat) / lat_span * (h - 1)).astype(int),
        0,
        h - 1,
    )

    sampled = field[y_idx, x_idx]
    colors = apply_colormap(sampled, variable_id)

    rgba[domain_mask, :3] = colors[domain_mask]
    rgba[domain_mask, 3] = 170
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


def _coord_key(value: float) -> float:
    return round(float(value), 6)


def _dataset_grid_shape(dataset_id: str) -> tuple[int, int] | None:
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


def _effective_grid_bounds() -> Dict[str, float]:
    grid_bounds_fn = getattr(store, "grid_bounds", None)
    if callable(grid_bounds_fn):
        try:
            bounds = grid_bounds_fn()
            if isinstance(bounds, dict):
                return bounds
        except Exception:
            pass
    return dict(SWISS_BOUNDS)


def _series_spatial_cache_token(dataset_id: str, lat: float, lon: float) -> tuple:
    # Cache by sampled grid point to preserve value correctness while still
    # allowing reuse across nearby request coordinates that map to the same cell.
    bounds = _effective_grid_bounds()
    shape = _dataset_grid_shape(dataset_id)
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
) -> tuple:
    spatial_token = _series_spatial_cache_token(dataset_id, lat, lon)
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
    with _SERIES_CACHE_GUARD:
        item = _SERIES_CACHE.get(key)
        if item is None:
            return None
        ts, payload = item
        if now - ts > SERIES_CACHE_TTL_SECONDS:
            _SERIES_CACHE.pop(key, None)
            _series_prune_orphan_locks()
            return None
        return payload


def _series_cache_put(key: tuple, payload: Dict[str, object]) -> None:
    now = time.monotonic()
    with _SERIES_CACHE_GUARD:
        _SERIES_CACHE[key] = (now, payload)
        if len(_SERIES_CACHE) <= SERIES_CACHE_MAX_ENTRIES:
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
    with _SERIES_KEY_LOCKS_GUARD:
        if len(_SERIES_KEY_LOCKS) <= max(64, SERIES_CACHE_MAX_ENTRIES * 2):
            return
        active_keys = set(_SERIES_CACHE.keys())
        stale = [k for k in _SERIES_KEY_LOCKS.keys() if k not in active_keys]
        for key in stale:
            _SERIES_KEY_LOCKS.pop(key, None)


def _series_payload_for_request(payload: Dict[str, object], lat: float, lon: float, cache_hit: bool) -> Dict[str, object]:
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
                        dataset_id, variable_id, init, lead, lat=lat, lon=lon, type_id=t, time_operator=time_operator
                    )
                else:
                    v = store.get_value(
                        dataset_id, variable_id, init, lead, lat=lat, lon=lon, type_id=t, time_operator=time_operator
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
        "time_operator": time_operator,
        "diagnostics": {
            "cached_only": cached_only,
            "missing_counts": missing_counts,
            "errors": errors[:25],
            "error_count": len(errors),
        },
    }
