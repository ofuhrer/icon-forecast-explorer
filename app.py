from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

import numpy as np

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from grid_utils import (
    _dataset_grid_shape,
    _effective_grid_bounds,
    _lat_to_y,
    _lon_to_x,
    _x_to_lon,
    _y_to_lat,
)
from input_validation import (
    FORECAST_TYPES,
    _parse_requested_types,
    _parse_requested_variables,
    _validate_time_operator,
)
from series_cache import (
    SERIES_CACHE_MAX_ENTRIES,  # noqa: F401 – re-exported for test patching
    SERIES_CACHE_TTL_SECONDS,  # noqa: F401 – re-exported for test patching
    SERIES_LATLON_BUCKET_DEG,  # noqa: F401 – re-exported for callers
    _SERIES_CACHE,  # noqa: F401 – re-exported; tests call .clear() on it
    _SERIES_CACHE_GUARD,  # noqa: F401 – re-exported for callers
    _SERIES_KEY_LOCKS,  # noqa: F401 – re-exported; tests call .clear() on it
    _SERIES_KEY_LOCKS_GUARD,  # noqa: F401 – re-exported for callers
    _build_series_payload,
    _series_cache_get,
    _series_cache_key,
    _series_cache_put,
    _series_key_lock,
    _series_payload_for_request,
    _series_prune_orphan_locks,  # noqa: F401 – re-exported for callers
    _series_spatial_cache_token,  # noqa: F401 – re-exported for callers
)
from tile_rendering import (
    VARIABLE_TO_COLORMAP,
    _legend_for_variable,
    render_tile_rgba,
)
from weather_data import (
    SWISS_BOUNDS,
    TIME_OPERATORS,
    ForecastStore,
)


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
    logger.info(
        "Logger configured level=%s file=%s",
        logging.getLevelName(level),
        log_file or "disabled",
    )
    return logger


LOGGER = _configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("App startup")
    store.start_background_prewarm()
    yield
    LOGGER.info("App shutdown")
    store.stop_background_prewarm()


app = FastAPI(title="ICON Forecast Explorer", lifespan=lifespan)


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


def _request_view_token(request: Request) -> str:
    header_token = str(request.headers.get("x-view-token", "")).strip()
    if header_token:
        return header_token
    query_token = str(request.query_params.get("_vt", "")).strip()
    if query_token:
        return query_token
    return "-"


@app.middleware("http")
async def _api_trace_middleware(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    start = time.perf_counter()
    view_token = _request_view_token(request)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(getattr(response, "status_code", 500))
        return response
    except asyncio.CancelledError:
        # Expected during client disconnects and server shutdown.
        # Returning 499 avoids noisy cancellation tracebacks in logs.
        status_code = 499
        return Response(status_code=499)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Keep high-volume tile traces at DEBUG while surfacing slow/failing API calls.
        if path.startswith("/api/tiles/"):
            LOGGER.debug(
                "API trace path=%s status=%s ms=%.1f vt=%s",
                path,
                status_code,
                elapsed_ms,
                view_token,
            )
        elif status_code == 499:
            LOGGER.debug(
                "API trace path=%s status=%s ms=%.1f vt=%s",
                path,
                status_code,
                elapsed_ms,
                view_token,
            )
        elif status_code >= 400 or elapsed_ms >= 2000:
            LOGGER.info(
                "API trace path=%s status=%s ms=%.1f vt=%s",
                path,
                status_code,
                elapsed_ms,
                view_token,
            )
        else:
            LOGGER.debug(
                "API trace path=%s status=%s ms=%.1f vt=%s",
                path,
                status_code,
                elapsed_ms,
                view_token,
            )


store = ForecastStore()


@app.get("/api/metadata")
def metadata() -> Dict[str, object]:
    # Return immediately with current catalog snapshot; refresh in background.
    store.refresh_catalog(force=False, blocking=False)
    datasets_payload = []
    for ds in store.dataset_metas:
        ds_bounds = _effective_grid_bounds(ds.dataset_id, store=store)
        ds_shape = _dataset_grid_shape(ds.dataset_id, store=store) or (
            int(getattr(ds, "target_grid_width", 0) or 0),
            int(getattr(ds, "target_grid_height", 0) or 0),
        )
        init_times = store.init_times(ds.dataset_id)
        init_to_leads = store.init_to_leads(ds.dataset_id)
        expected_init_to_leads = {init: store.expected_lead_hours_for_init(ds.dataset_id, init) for init in init_times}
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
            level_selector_fn = getattr(store, "variable_level_selector_payload", None)
            level_selector = (
                level_selector_fn(var.variable_id)
                if callable(level_selector_fn)
                else {
                    "enabled": bool(var.supported_level_kinds),
                    "supported_kinds": list(var.supported_level_kinds),
                    "default_kind": var.default_level_kind,
                    "levels": {},
                }
            )
            variables.append(
                {
                    "variable_id": var.variable_id,
                    "display_name": var.display_name,
                    "group_id": var.group_id,
                    "group_display_name": var.group_display_name,
                    "long_name": long_name,
                    "grib_name": grib_name,
                    "standard_unit": standard_unit,
                    "display_unit": var.unit,
                    "unit": var.unit,
                    "range": [var.min_value, var.max_value],
                    "lead_time_display_offset_hours": int(store.variable_lead_display_offset_hours(var.variable_id)),
                    "level_selector": level_selector,
                    "default_colormap": VARIABLE_TO_COLORMAP.get(var.variable_id, "default"),
                    "legend": _legend_for_variable(var.variable_id),
                }
            )
        datasets_payload.append(
            {
                "dataset_id": ds.dataset_id,
                "display_name": ds.display_name,
                "target_grid_width": int(ds_shape[0]),
                "target_grid_height": int(ds_shape[1]),
                "bounds": ds_bounds,
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
    bounds = (
        dict(first.get("bounds", {}))
        if datasets_payload and isinstance(first.get("bounds"), dict)
        else dict(SWISS_BOUNDS)
    )
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
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        sample_fn = getattr(store, "_sample_field_value", None)
        get_cached_field_fn = getattr(store, "get_cached_field", None)
        if callable(sample_fn) and callable(get_cached_field_fn):
            field = get_cached_field_fn(
                dataset_id,
                variable_id,
                init,
                lead,
                type_id=type_id,
                time_operator=time_operator,
                level_kind=level_kind,
                level_value=level_value,
            )
            if field is None:
                queued = store.queue_field_fetch(
                    dataset_id,
                    variable_id,
                    init,
                    lead,
                    type_id=type_id,
                    time_operator=time_operator,
                    level_kind=level_kind,
                    level_value=level_value,
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
            val = sample_fn(dataset_id, field, lat=lat, lon=lon)
            if val is None:
                raise HTTPException(status_code=404, detail="No value at location")
        else:
            val = store.get_cached_value(
                dataset_id,
                variable_id,
                init,
                lead,
                lat=lat,
                lon=lon,
                type_id=type_id,
                time_operator=time_operator,
                level_kind=level_kind,
                level_value=level_value,
            )
            if val is None:
                queued = store.queue_field_fetch(
                    dataset_id,
                    variable_id,
                    init,
                    lead,
                    type_id=type_id,
                    time_operator=time_operator,
                    level_kind=level_kind,
                    level_value=level_value,
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


@app.get("/api/prefetch")
def prefetch(
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        queued = store.queue_field_fetch(
            dataset_id,
            variable_id,
            init,
            lead,
            type_id=type_id,
            time_operator=time_operator,
            level_kind=level_kind,
            level_value=level_value,
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
        req_variables = _parse_requested_variables(variables, store=store)
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
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
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
            level_kind=level_kind,
            level_value=level_value,
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
            level_kind=level_kind,
            level_value=level_value,
        )
        if failure is not None:
            return {
                "dataset_id": dataset_id,
                "type_id": type_id,
                "variable_id": variable_id,
                "init": init,
                "lead": lead,
                "time_operator": time_operator,
                "level_kind": level_kind,
                "level_value": level_value,
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
        store.queue_field_fetch(
            dataset_id,
            variable_id,
            init,
            lead,
            type_id=type_id,
            time_operator=time_operator,
            level_kind=level_kind,
            level_value=level_value,
        )
        return {
            "dataset_id": dataset_id,
            "type_id": type_id,
            "variable_id": variable_id,
            "init": init,
            "lead": lead,
            "time_operator": time_operator,
            "level_kind": level_kind,
            "level_value": level_value,
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
        "level_kind": level_kind,
        "level_value": level_value,
        "status": "ready",
        "debug": info,
    }


@app.get("/api/field-progress")
async def field_progress(
    response: Response = None,
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
) -> Dict[str, object]:
    try:
        time_operator = _validate_time_operator(time_operator)
        level_kind_value = level_kind if isinstance(level_kind, str) and level_kind else None
        level_value_value = level_value if isinstance(level_value, str) and level_value else None
        progress = store.get_field_progress(
            dataset_id=dataset_id,
            variable_id=variable_id,
            init_str=init,
            lead_hour=lead,
            type_id=type_id,
            time_operator=time_operator,
            level_kind=level_kind_value,
            level_value=level_value_value,
        )
        if response is not None:
            response.headers["Cache-Control"] = "no-store"
        return {
            "dataset_id": dataset_id,
            "type_id": type_id,
            "variable_id": variable_id,
            "init": init,
            "lead": lead,
            "time_operator": time_operator,
            "level_kind": level_kind_value,
            "level_value": level_value_value,
            "progress": progress,
        }
    except ValueError as exc:
        LOGGER.warning("Field progress request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/field-progress-stream")
async def field_progress_stream(
    request: Request,
    dataset_id: str = Query(...),
    type_id: str = Query("control"),
    variable_id: str = Query(...),
    init: str = Query(...),
    lead: int = Query(..., ge=0),
    time_operator: str = Query("none"),
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
) -> StreamingResponse:
    try:
        time_operator = _validate_time_operator(time_operator)
        level_kind_value = level_kind if isinstance(level_kind, str) and level_kind else None
        level_value_value = level_value if isinstance(level_value, str) and level_value else None
    except ValueError as exc:
        LOGGER.warning("Field progress stream request invalid: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_stream():
        last_payload = ""
        while True:
            if await request.is_disconnected():
                break
            progress = store.get_field_progress(
                dataset_id=dataset_id,
                variable_id=variable_id,
                init_str=init,
                lead_hour=lead,
                type_id=type_id,
                time_operator=time_operator,
                level_kind=level_kind_value,
                level_value=level_value_value,
            )
            payload = json.dumps(
                {
                    "dataset_id": dataset_id,
                    "type_id": type_id,
                    "variable_id": variable_id,
                    "init": init,
                    "lead": lead,
                    "time_operator": time_operator,
                    "level_kind": level_kind_value,
                    "level_value": level_value_value,
                    "progress": progress,
                }
            )
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if progress.get("status") in {"ready", "error"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
) -> Dict[str, object]:
    req_types = _parse_requested_types(types)
    time_operator = _validate_time_operator(time_operator)

    key = _series_cache_key(
        dataset_id,
        variable_id,
        init,
        req_types,
        cached_only,
        lat,
        lon,
        time_operator,
        level_kind,
        level_value,
        store=store,
    )
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
            level_kind=level_kind,
            level_value=level_value,
            store=store,
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
    level_kind: str | None = Query(None),
    level_value: str | None = Query(None),
) -> Response:
    try:
        time_operator = _validate_time_operator(time_operator)
        field = store.get_cached_field(
            dataset_id,
            variable_id,
            init,
            lead,
            type_id=type_id,
            time_operator=time_operator,
            level_kind=level_kind,
            level_value=level_value,
        )
        if field is None:
            failure = store.get_field_failure(
                dataset_id=dataset_id,
                variable_id=variable_id,
                init_str=init,
                lead_hour=lead,
                type_id=type_id,
                time_operator=time_operator,
                level_kind=level_kind,
                level_value=level_value,
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
                dataset_id,
                variable_id,
                init,
                lead,
                type_id=type_id,
                time_operator=time_operator,
                level_kind=level_kind,
                level_value=level_value,
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

    rgba = render_tile_rgba(field, z, x, y, variable_id, _effective_grid_bounds(dataset_id, store=store))
    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


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
            cached = store.get_cached_wind_vectors(dataset_id, init, lead, type_id=type_id, time_operator=time_operator)
            if cached is None:
                store.queue_wind_vector_fetch(dataset_id, init, lead, type_id=type_id, time_operator=time_operator)
                return {"status": "loading", "vectors": [], "unit": "km/h"}
            u_field, v_field = cached
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    z = float(zoom)
    base_stride = int(max(6, min(18, round(18 - z))))
    bounds = _effective_grid_bounds(dataset_id, store=store)
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
