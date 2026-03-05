from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import re
import threading
import sys
import time
import math
from urllib.parse import urlparse
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Re-export everything from the focused sub-modules so that existing callers
# (app.py, tests, etc.) that import from weather_data continue to work.
# ---------------------------------------------------------------------------
from weather_models import (  # noqa: F401
    SWISS_BOUNDS,
    STAC_SEARCH_URL,
    CATALOG_REFRESH_SECONDS,
    FIELD_CACHE_VERSION,
    MS_TO_KMH,
    FIELD_CACHE_RETENTION_HOURS,
    FIELD_CACHE_CLEANUP_INTERVAL_SECONDS,
    FIELD_CACHE_MAX_ENTRIES,
    GRIB_ASSET_KEY_LOCKS_MAX_ENTRIES,
    GRIB_ASSET_CACHE_ENABLED,
    GRIB_ASSET_CACHE_TTL_HOURS,
    GRIB_ASSET_CACHE_MAX_BYTES,
    GRIB_DOWNLOAD_WORKERS,
    METEOGRAM_WARM_WORKERS,
    METEOGRAM_WARM_JOB_TTL_SECONDS,
    METEOGRAM_WARM_PREFETCH_ASSETS,
    SUPPORTED_FORECAST_TYPES,
    TIME_OPERATORS,
    SUPPORTED_TIME_OPERATORS,
    HOT_PREWARM_INTERVAL_SECONDS,
    HOT_PREWARM_VARIABLES,
    HOT_PREWARM_TYPES,
    HOT_PREWARM_LEADS,
    HOT_PREWARM_ALL_LEADS,
    HOT_PREWARM_ENABLED,
    OGD_FETCH_RETRIES,
    OGD_FETCH_BASE_BACKOFF_SECONDS,
    OGD_HORIZON_FALLBACK_STEPS,
    BACKGROUND_FETCH_WORKERS,
    FIELD_FAILURE_TTL_SECONDS,
    DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS,
    DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS,
    OGD_PARAMETER_INFO,
    OGDIngestionError,
    OGDRequestError,
    OGDDecodeError,
    VariableMeta,
    DatasetMeta,
)
from weather_cache import (
    load_cached_field_file,
    save_cached_field_file,
    load_cached_wind_vector_file,
    save_cached_wind_vector_file,
    load_field_debug_info,
    save_field_debug_info,
    field_debug_path,
    safe_unlink,
    parse_iso_duration_hours,
    init_to_iso,
)
from weather_grib import (
    decode_param_candidates,
    pick_best_array,
    ogd_variable_candidates,
    horizon_candidates,
    reduce_members,
    member_axis,
    fill_nan_with_neighbors,
    ensure_eccodes_definition_path,
    field_end_step,
    deaggregate_from_reference,
)

LOGGER = logging.getLogger("icon_forecast.weather_data")


class ForecastStore:
    """ICON-CH1/CH2 EPS control ingestion and cache pipeline."""

    def __init__(self) -> None:
        self._variables: Dict[str, VariableMeta] = {
            "t_2m": VariableMeta(
                variable_id="t_2m",
                display_name="2 m temperature",
                unit="°C",
                min_value=-20.0,
                max_value=40.0,
                ogd_variable="T_2M",
            ),
            "td_2m": VariableMeta(
                variable_id="td_2m",
                display_name="2 m dew point",
                unit="°C",
                min_value=-30.0,
                max_value=30.0,
                ogd_variable="TD_2M",
            ),
            "pres_sfc": VariableMeta(
                variable_id="pres_sfc",
                display_name="Surface pressure",
                unit="hPa",
                min_value=960.0,
                max_value=1040.0,
                ogd_variable="PS",
            ),
            "wind_speed_10m": VariableMeta(
                variable_id="wind_speed_10m",
                display_name="10 m wind speed",
                unit="km/h",
                min_value=0.0,
                max_value=200.0,
                ogd_components=("U_10M", "V_10M"),
            ),
            "vmax_10m": VariableMeta(
                variable_id="vmax_10m",
                display_name="10 m wind gust",
                unit="km/h",
                min_value=0.0,
                max_value=240.0,
                ogd_variable="VMAX_10M",
            ),
            "tot_prec": VariableMeta(
                variable_id="tot_prec",
                display_name="Total precipitation",
                unit="mm",
                min_value=0.0,
                max_value=80.0,
                ogd_variable="TOT_PREC",
            ),
            "rain_gsp": VariableMeta(
                variable_id="rain_gsp",
                display_name="Large-scale rain",
                unit="mm",
                min_value=0.0,
                max_value=80.0,
                ogd_variable="RAIN_GSP",
            ),
            "clct": VariableMeta(
                variable_id="clct",
                display_name="Total cloud cover",
                unit="%",
                min_value=0.0,
                max_value=100.0,
                ogd_variable="CLCT",
            ),
            "clcl": VariableMeta(
                variable_id="clcl",
                display_name="Low cloud cover",
                unit="%",
                min_value=0.0,
                max_value=100.0,
                ogd_variable="CLCL",
            ),
            "clcm": VariableMeta(
                variable_id="clcm",
                display_name="Mid cloud cover",
                unit="%",
                min_value=0.0,
                max_value=100.0,
                ogd_variable="CLCM",
            ),
            "clch": VariableMeta(
                variable_id="clch",
                display_name="High cloud cover",
                unit="%",
                min_value=0.0,
                max_value=100.0,
                ogd_variable="CLCH",
            ),
            "ceiling": VariableMeta(
                variable_id="ceiling",
                display_name="Cloud ceiling",
                unit="m",
                min_value=0.0,
                max_value=9000.0,
                ogd_variable="CEILING",
            ),
            "hzerocl": VariableMeta(
                variable_id="hzerocl",
                display_name="Freezing level",
                unit="m",
                min_value=0.0,
                max_value=8000.0,
                ogd_variable="HZEROCL",
            ),
            "w_snow": VariableMeta(
                variable_id="w_snow",
                display_name="Snow water equivalent",
                unit="mm",
                min_value=0.0,
                max_value=120.0,
                ogd_variable="W_SNOW",
            ),
            "snow_gsp": VariableMeta(
                variable_id="snow_gsp",
                display_name="Large-scale snowfall",
                unit="mm",
                min_value=0.0,
                max_value=200.0,
                ogd_variable="SNOW_GSP",
            ),
            "snowlmt": VariableMeta(
                variable_id="snowlmt",
                display_name="Snowfall limit",
                unit="m",
                min_value=0.0,
                max_value=4000.0,
                ogd_variable="SNOWLMT",
            ),
            "dursun": VariableMeta(
                variable_id="dursun",
                display_name="Sunshine duration",
                unit="min",
                min_value=0.0,
                max_value=60.0,
                ogd_variable="DURSUN",
            ),
            "asob_s": VariableMeta(
                variable_id="asob_s",
                display_name="Net shortwave radiation",
                unit="W/m^2",
                min_value=-200.0,
                max_value=1200.0,
                ogd_variable="ASOB_S",
            ),
            "athb_s": VariableMeta(
                variable_id="athb_s",
                display_name="Net longwave radiation",
                unit="W/m^2",
                min_value=-300.0,
                max_value=300.0,
                ogd_variable="ATHB_S",
            ),
            "aswdir_s": VariableMeta(
                variable_id="aswdir_s",
                display_name="Direct shortwave radiation",
                unit="W/m^2",
                min_value=0.0,
                max_value=1200.0,
                ogd_variable="ASWDIR_S",
            ),
            "aswdifd_s": VariableMeta(
                variable_id="aswdifd_s",
                display_name="Diffuse shortwave radiation",
                unit="W/m^2",
                min_value=0.0,
                max_value=1000.0,
                ogd_variable="ASWDIFD_S",
            ),
            "ashfl_s": VariableMeta(
                variable_id="ashfl_s",
                display_name="Sensible heat flux",
                unit="W/m^2",
                min_value=-500.0,
                max_value=500.0,
                ogd_variable="ASHFL_S",
            ),
            "alhfl_s": VariableMeta(
                variable_id="alhfl_s",
                display_name="Latent heat flux",
                unit="W/m^2",
                min_value=-700.0,
                max_value=700.0,
                ogd_variable="ALHFL_S",
            ),
            "cape_ml": VariableMeta(
                variable_id="cape_ml",
                display_name="CAPE (mixed layer)",
                unit="J/kg",
                min_value=0.0,
                max_value=4000.0,
                ogd_variable="CAPE_ML",
            ),
            "cin_ml": VariableMeta(
                variable_id="cin_ml",
                display_name="CIN (mixed layer)",
                unit="J/kg",
                min_value=-500.0,
                max_value=0.0,
                ogd_variable="CIN_ML",
            ),
        }
        self._catalog_reference_ogd_variable = "T_2M"

        self._dataset_configs = {
            "icon-ch1-eps-control": DatasetMeta(
                dataset_id="icon-ch1-eps-control",
                display_name="ICON-CH1-EPS",
                collection_id="ch.meteoschweiz.ogd-forecasting-icon-ch1",
                ogd_collection="ogd-forecasting-icon-ch1",
                expected_members_total=11,
                fallback_cycle_hours=3,
                fallback_lead_hours=list(range(0, 49)),
                target_grid_width=540,
                target_grid_height=380,
                target_grid_spacing_km=0.7,
            ),
            "icon-ch2-eps-control": DatasetMeta(
                dataset_id="icon-ch2-eps-control",
                display_name="ICON-CH2-EPS",
                collection_id="ch.meteoschweiz.ogd-forecasting-icon-ch2",
                ogd_collection="ogd-forecasting-icon-ch2",
                expected_members_total=21,
                fallback_cycle_hours=6,
                fallback_lead_hours=list(range(0, 121)),
                target_grid_width=270,
                target_grid_height=190,
                target_grid_spacing_km=1.4,
            ),
        }

        self._grid_width = 540
        self._grid_height = 380
        self._grid_bounds: Dict[str, Dict[str, float]] = {
            dataset_id: dict(SWISS_BOUNDS) for dataset_id in self._dataset_configs
        }
        self._grid_bounds_locked: Dict[str, bool] = {dataset_id: False for dataset_id in self._dataset_configs}
        self._computed_target_shapes: Dict[str, Tuple[int, int]] = {}
        self._grid_bounds_guard = threading.Lock()

        self._cache_dir = Path("cache")
        self._field_cache_dir = self._cache_dir / "fields"
        self._vector_cache_dir = self._cache_dir / "vectors"
        self._catalog_cache_dir = self._cache_dir / "catalogs"
        self._grib_asset_cache_dir = self._cache_dir / "grib_assets"
        self._constant_cache_dir = self._cache_dir / "constants"
        self._field_cache_dir.mkdir(parents=True, exist_ok=True)
        self._vector_cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_cache_dir.mkdir(parents=True, exist_ok=True)
        self._grib_asset_cache_dir.mkdir(parents=True, exist_ok=True)
        self._constant_cache_dir.mkdir(parents=True, exist_ok=True)

        self._field_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._wind_vector_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self._constant_fields: Dict[Tuple[str, str], np.ndarray] = {}
        self._field_debug_info: Dict[Tuple[str, ...], Dict[str, object]] = {}
        self._field_failures: Dict[Tuple[str, ...], Dict[str, object]] = {}
        self._field_failure_guard = threading.Lock()
        self._key_locks: Dict[Tuple[str, ...], threading.Lock] = {}
        self._key_locks_guard = threading.Lock()
        self._catalog_guard = threading.Lock()
        self._constant_guard = threading.Lock()
        self._constants_asset_guard = threading.Lock()
        self._refresh_state_guard = threading.Lock()
        self._cleanup_guard = threading.Lock()
        self._grib_asset_cache_guard = threading.Lock()
        self._grib_asset_key_locks: Dict[str, threading.Lock] = {}
        self._grib_asset_key_locks_guard = threading.Lock()
        self._grib_download_futures: Dict[str, Future] = {}
        self._grib_download_futures_guard = threading.Lock()
        self._grib_download_executor = ThreadPoolExecutor(
            max_workers=max(1, GRIB_DOWNLOAD_WORKERS),
            thread_name_prefix="grib-download",
        )
        self._prewarm_guard = threading.Lock()
        self._display_offset_guard = threading.Lock()
        self._background_fetch_guard = threading.Lock()
        self._background_fetch_inflight: set[Tuple[str, ...]] = set()
        self._background_fetch_executor = ThreadPoolExecutor(
            max_workers=max(1, BACKGROUND_FETCH_WORKERS),
            thread_name_prefix="field-fetch",
        )
        self._meteogram_warm_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="meteogram-warm-job",
        )
        self._meteogram_warm_guard = threading.Lock()
        self._meteogram_warm_jobs: Dict[str, Dict[str, object]] = {}
        self._meteogram_warm_index: Dict[Tuple[str, ...], str] = {}
        self._meteogram_warm_seq = 0

        self._catalogs: Dict[str, Dict[str, object]] = {}
        self._constants_asset_urls: Dict[str, str] = {}
        self._catalog_refreshed_at: Dict[str, datetime] = {}
        self._refresh_inflight: set[str] = set()
        self._variable_lead_display_offsets: Dict[str, int] = {
            variable_id: int(meta.lead_time_display_offset_hours) for variable_id, meta in self._variables.items()
        }
        self._prewarm_started = False
        self._prewarm_thread: threading.Thread | None = None
        self._prewarm_stop = threading.Event()
        now = datetime.now(timezone.utc)
        stale = now - timedelta(seconds=CATALOG_REFRESH_SECONDS + 1)
        for dataset_id, cfg in self._dataset_configs.items():
            # Avoid network calls during app import/startup:
            # 1) use last known on-disk catalog if available
            # 2) otherwise use deterministic fallback
            cached = self._load_catalog_cache(cfg)
            self._catalogs[dataset_id] = cached if cached is not None else self._fallback_catalog(cfg)
            self._catalog_refreshed_at[dataset_id] = stale
        self._last_cleanup_at = datetime.min.replace(tzinfo=timezone.utc)
        self._last_grib_asset_cleanup_at = datetime.min.replace(tzinfo=timezone.utc)
        self._cleanup_field_cache(force=True)

    def start_background_prewarm(self) -> None:
        if not HOT_PREWARM_ENABLED:
            LOGGER.info("Hot prewarm disabled by HOT_PREWARM_ENABLED")
            return
        with self._prewarm_guard:
            if self._prewarm_started:
                return
            self._prewarm_started = True
            self._prewarm_stop.clear()
            self._prewarm_thread = threading.Thread(
                target=self._prewarm_loop,
                name="hot-prewarm",
                daemon=True,
            )
            self._prewarm_thread.start()
            LOGGER.info("Started hot prewarm thread")

    def stop_background_prewarm(self) -> None:
        with self._prewarm_guard:
            self._prewarm_stop.set()
        self._background_fetch_executor.shutdown(wait=False, cancel_futures=True)
        self._grib_download_executor.shutdown(wait=False, cancel_futures=True)
        self._meteogram_warm_executor.shutdown(wait=False, cancel_futures=True)
        LOGGER.info("Stopped background workers")

    def queue_field_fetch(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> bool:
        self._validate_request(dataset_id, variable_id, init_str, lead_hour, type_id, time_operator=time_operator)
        key = (dataset_id, type_id, variable_id, time_operator, init_str, lead_hour)
        if self._has_recent_field_failure(key):
            return False
        if key in self._field_cache:
            return False
        if self._field_cache_path(
            dataset_id, type_id, variable_id, init_str, lead_hour, time_operator=time_operator
        ).exists():
            return False

        with self._background_fetch_guard:
            if key in self._background_fetch_inflight:
                return False
            self._background_fetch_inflight.add(key)

        self._background_fetch_executor.submit(self._background_fetch_job, key)
        LOGGER.debug("Queued field fetch key=%s", key)
        return True

    def queue_wind_vector_fetch(
        self,
        dataset_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> bool:
        self._validate_request(
            dataset_id, "wind_speed_10m", init_str, lead_hour, type_id, time_operator=time_operator
        )
        key = (dataset_id, type_id, "wind_vectors", time_operator, init_str, lead_hour)
        if key in self._wind_vector_cache:
            return False
        if self._wind_vector_cache_path(dataset_id, type_id, init_str, lead_hour, time_operator=time_operator).exists():
            return False

        with self._background_fetch_guard:
            if key in self._background_fetch_inflight:
                return False
            self._background_fetch_inflight.add(key)

        self._background_fetch_executor.submit(self._background_fetch_job, key)
        LOGGER.debug("Queued wind vector fetch key=%s", key)
        return True

    def start_meteogram_warmup(
        self,
        dataset_id: str,
        init_str: str,
        variable_ids: List[str],
        type_ids: List[str],
        time_operator: str = "none",
    ) -> Dict[str, object]:
        self._dataset_config(dataset_id)
        if time_operator not in SUPPORTED_TIME_OPERATORS:
            raise ValueError(f"Unknown time_operator: {time_operator}")
        if not variable_ids:
            raise ValueError("No variables requested for warmup")
        if not type_ids:
            raise ValueError("No forecast types requested for warmup")
        norm_variables: List[str] = []
        for variable_id in variable_ids:
            if variable_id not in self._variables:
                raise ValueError(f"Unknown variable_id: {variable_id}")
            if variable_id not in norm_variables:
                norm_variables.append(variable_id)
        norm_types: List[str] = []
        for type_id in type_ids:
            if type_id not in SUPPORTED_FORECAST_TYPES:
                raise ValueError(f"Unknown type_id: {type_id}")
            if type_id not in norm_types:
                norm_types.append(type_id)
        leads = [int(v) for v in self.lead_hours_for_init(dataset_id, init_str)]
        if not leads:
            raise ValueError(f"No leads available for init {init_str}")

        key = (
            dataset_id,
            init_str,
            time_operator,
            tuple(norm_variables),
            tuple(norm_types),
        )
        with self._meteogram_warm_guard:
            self._cleanup_meteogram_warm_jobs_locked()
            existing_job_id = self._meteogram_warm_index.get(key)
            if existing_job_id:
                existing = self._meteogram_warm_jobs.get(existing_job_id)
                if existing and str(existing.get("status", "")) in {"queued", "running", "done", "partial"}:
                    return self._meteogram_warm_payload(existing)
                self._meteogram_warm_index.pop(key, None)

        total_tasks, ready_tasks, missing_tasks = self._compute_meteogram_warm_tasks(
            dataset_id=dataset_id,
            init_str=init_str,
            variable_ids=norm_variables,
            type_ids=norm_types,
            leads=leads,
            time_operator=time_operator,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._meteogram_warm_guard:
            self._meteogram_warm_seq += 1
            job_id = f"mw-{int(time.time())}-{self._meteogram_warm_seq}"
            job = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "init": init_str,
                "variables": list(norm_variables),
                "types": list(norm_types),
                "time_operator": time_operator,
                "status": "queued",
                "created_at": now_iso,
                "started_at": None,
                "finished_at": None,
                "total_tasks": int(total_tasks),
                "completed_tasks": int(ready_tasks),
                "failed_tasks": 0,
                "errors": [],
                "key": key,
            }
            self._meteogram_warm_jobs[job_id] = job
            self._meteogram_warm_index[key] = job_id
            if not missing_tasks:
                job["status"] = "done"
                job["started_at"] = now_iso
                job["finished_at"] = now_iso
                return self._meteogram_warm_payload(job)
            self._meteogram_warm_executor.submit(self._run_meteogram_warm_job, job_id, missing_tasks)
            return self._meteogram_warm_payload(job)

    def get_meteogram_warmup(self, job_id: str) -> Dict[str, object]:
        with self._meteogram_warm_guard:
            self._cleanup_meteogram_warm_jobs_locked()
            job = self._meteogram_warm_jobs.get(str(job_id))
            if job is None:
                raise KeyError(str(job_id))
            return self._meteogram_warm_payload(job)

    def _compute_meteogram_warm_tasks(
        self,
        dataset_id: str,
        init_str: str,
        variable_ids: List[str],
        type_ids: List[str],
        leads: List[int],
        time_operator: str,
    ) -> Tuple[int, int, List[Tuple[str, str, str, int, str, str]]]:
        tasks: List[Tuple[str, str, str, int, str, str]] = []
        total = 0
        ready = 0
        for variable_id in variable_ids:
            for type_id in type_ids:
                for lead_hour in leads:
                    total += 1
                    cached = self.get_cached_field(
                        dataset_id=dataset_id,
                        variable_id=variable_id,
                        init_str=init_str,
                        lead_hour=int(lead_hour),
                        type_id=type_id,
                        time_operator=time_operator,
                    )
                    if cached is not None:
                        ready += 1
                    else:
                        tasks.append((dataset_id, variable_id, init_str, int(lead_hour), type_id, time_operator))
        return total, ready, tasks

    def _run_meteogram_warm_job(
        self,
        job_id: str,
        tasks: List[Tuple[str, str, str, int, str, str]],
    ) -> None:
        with self._meteogram_warm_guard:
            job = self._meteogram_warm_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            job["phase"] = "prefetch_assets" if METEOGRAM_WARM_PREFETCH_ASSETS else "fetch_fields"
            job["asset_total"] = 0
            job["asset_completed"] = 0
            job["asset_failed"] = 0
        if METEOGRAM_WARM_PREFETCH_ASSETS:
            self._prefetch_meteogram_assets(job_id, tasks)
        with self._meteogram_warm_guard:
            job = self._meteogram_warm_jobs.get(job_id)
            if job is None:
                return
            job["phase"] = "fetch_fields"
        worker_count = max(1, min(int(METEOGRAM_WARM_WORKERS), len(tasks)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="meteogram-warm-field") as executor:
            futures = [
                executor.submit(
                    self.get_field,
                    dataset_id,
                    variable_id,
                    init_str,
                    lead_hour,
                    type_id=type_id,
                    time_operator=time_operator,
                )
                for dataset_id, variable_id, init_str, lead_hour, type_id, time_operator in tasks
            ]
            for fut in as_completed(futures):
                error_text: str | None = None
                try:
                    fut.result()
                except Exception as exc:
                    error_text = str(exc)
                with self._meteogram_warm_guard:
                    job = self._meteogram_warm_jobs.get(job_id)
                    if job is None:
                        return
                    job["completed_tasks"] = int(job.get("completed_tasks", 0)) + 1
                    if error_text:
                        job["failed_tasks"] = int(job.get("failed_tasks", 0)) + 1
                        errors = job.setdefault("errors", [])
                        if isinstance(errors, list) and len(errors) < 15:
                            errors.append(error_text)
        with self._meteogram_warm_guard:
            job = self._meteogram_warm_jobs.get(job_id)
            if job is None:
                return
            failed = int(job.get("failed_tasks", 0))
            job["status"] = "done" if failed == 0 else "partial"
            job["phase"] = "done"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

    def _prefetch_meteogram_assets(
        self,
        job_id: str,
        tasks: List[Tuple[str, str, str, int, str, str]],
    ) -> None:
        urls = self._collect_meteogram_prefetch_urls(tasks)
        if not urls:
            return
        # De-duplicate by local cache target path (signed query strings can differ).
        unique: Dict[str, str] = {}
        for url in urls:
            path = self._grib_asset_path_for_url(str(url))
            unique.setdefault(str(path), str(url))
        urls_dedup = list(unique.values())
        with self._meteogram_warm_guard:
            job = self._meteogram_warm_jobs.get(job_id)
            if job is None:
                return
            job["asset_total"] = int(len(urls_dedup))
            job["asset_completed"] = 0
            job["asset_failed"] = 0
        futures: List[Future] = []
        for url in urls_dedup:
            cache_path = self._grib_asset_path_for_url(str(url))
            futures.append(self._ensure_grib_asset_cached(str(url), cache_path))
        for fut in as_completed(futures):
            err = None
            try:
                fut.result()
            except Exception as exc:
                err = str(exc)
            with self._meteogram_warm_guard:
                job = self._meteogram_warm_jobs.get(job_id)
                if job is None:
                    return
                job["asset_completed"] = int(job.get("asset_completed", 0)) + 1
                if err:
                    job["asset_failed"] = int(job.get("asset_failed", 0)) + 1
                    errors = job.setdefault("errors", [])
                    if isinstance(errors, list) and len(errors) < 15:
                        errors.append(f"asset prefetch failed: {err}")

    def _collect_meteogram_prefetch_urls(
        self, tasks: List[Tuple[str, str, str, int, str, str]]
    ) -> List[str]:
        if not tasks:
            return []

        class _Req:
            __slots__ = ("need_control", "need_perturbed", "need_prev")

            def __init__(self) -> None:
                self.need_control = False
                self.need_perturbed = False
                self.need_prev = False

        requirements: Dict[Tuple[str, str, str, int, str], _Req] = {}
        ensemble_types = {"mean", "median", "p10", "p90", "min", "max"}
        for dataset_id, variable_id, init_str, lead_hour, type_id, time_operator in tasks:
            key = (dataset_id, variable_id, init_str, int(lead_hour), str(time_operator))
            req = requirements.get(key)
            if req is None:
                req = _Req()
                requirements[key] = req
            req.need_control = True
            if str(type_id) in ensemble_types:
                req.need_perturbed = True
            if int(lead_hour) > 0 and (
                variable_id in DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS
                or variable_id in DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS
            ):
                req.need_prev = True

        out_urls: List[str] = []
        out_seen: set[str] = set()

        for (dataset_id, variable_id, init_str, lead_hour, _time_operator), req in requirements.items():
            cfg = self._dataset_config(dataset_id)
            variable = self._variables.get(variable_id)
            if variable is None:
                continue
            reference_iso = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            ogd_api = self._ogd_api(dataset_id, init_str)

            ogd_variables: List[str] = []
            if variable.ogd_components:
                ogd_variables.extend([str(v) for v in variable.ogd_components if str(v).strip()])
            elif variable.ogd_variable:
                ogd_variables.append(str(variable.ogd_variable))
            if not ogd_variables:
                continue

            leads = [int(lead_hour)]
            if req.need_prev:
                previous = self._previous_available_lead(dataset_id, init_str, int(lead_hour))
                if previous is not None:
                    leads.append(int(previous))
            # Stable ordering for determinism.
            leads = sorted(set(leads))

            for src_lead in leads:
                for ogd_var in ogd_variables:
                    if req.need_control:
                        for u in self._resolve_asset_urls_for_prefetch(
                            ogd_api=ogd_api,
                            ogd_collection=cfg.ogd_collection,
                            ogd_variable=ogd_var,
                            reference_iso=reference_iso,
                            lead_hour=src_lead,
                            perturbed=False,
                        ):
                            if u not in out_seen:
                                out_seen.add(u)
                                out_urls.append(u)
                    if req.need_perturbed:
                        for u in self._resolve_asset_urls_for_prefetch(
                            ogd_api=ogd_api,
                            ogd_collection=cfg.ogd_collection,
                            ogd_variable=ogd_var,
                            reference_iso=reference_iso,
                            lead_hour=src_lead,
                            perturbed=True,
                        ):
                            if u not in out_seen:
                                out_seen.add(u)
                                out_urls.append(u)
        return out_urls

    def _resolve_asset_urls_for_prefetch(
        self,
        ogd_api,
        ogd_collection: str,
        ogd_variable: str,
        reference_iso: str,
        lead_hour: int,
        perturbed: bool,
    ) -> List[str]:
        for effective_lead in horizon_candidates(int(lead_hour)):
            for candidate_variable in ogd_variable_candidates(ogd_variable):
                request = ogd_api.Request(
                    collection=ogd_collection,
                    variable=candidate_variable,
                    reference_datetime=reference_iso,
                    perturbed=bool(perturbed),
                    horizon=timedelta(hours=int(effective_lead)),
                )
                urls = self._safe_asset_urls_for_request(ogd_api, request)
                if urls:
                    return urls
        return []

    def _meteogram_warm_payload(self, job: Dict[str, object]) -> Dict[str, object]:
        total = max(0, int(job.get("total_tasks", 0)))
        completed = max(0, int(job.get("completed_tasks", 0)))
        failed = max(0, int(job.get("failed_tasks", 0)))
        asset_total = max(0, int(job.get("asset_total", 0)))
        asset_completed = max(0, int(job.get("asset_completed", 0)))
        asset_failed = max(0, int(job.get("asset_failed", 0)))
        percent = 100 if total == 0 else int(round((completed / float(total)) * 100))
        payload = {
            "job_id": str(job.get("job_id", "")),
            "dataset_id": str(job.get("dataset_id", "")),
            "init": str(job.get("init", "")),
            "variables": list(job.get("variables", [])),
            "types": list(job.get("types", [])),
            "time_operator": str(job.get("time_operator", "none")),
            "status": str(job.get("status", "queued")),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "remaining_tasks": max(0, total - completed),
            "percent_complete": max(0, min(100, percent)),
            "ready": total > 0 and completed >= total and failed == 0,
            "partial": total > 0 and completed >= total and failed > 0,
            "errors": list(job.get("errors", [])),
            "phase": str(job.get("phase", "queued")),
            "asset_total": asset_total,
            "asset_completed": asset_completed,
            "asset_failed": asset_failed,
            "asset_percent_complete": (
                0 if asset_total == 0 else max(0, min(100, int(round((asset_completed / float(asset_total)) * 100))))
            ),
        }
        return payload

    def _cleanup_meteogram_warm_jobs_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(60, METEOGRAM_WARM_JOB_TTL_SECONDS))
        stale_ids: List[str] = []
        for job_id, job in self._meteogram_warm_jobs.items():
            status = str(job.get("status", ""))
            if status in {"queued", "running"}:
                continue
            finished_raw = str(job.get("finished_at") or "")
            try:
                finished_at = datetime.fromisoformat(finished_raw)
            except (ValueError, TypeError):
                finished_at = datetime.min.replace(tzinfo=timezone.utc)
            if finished_at.tzinfo is None:
                finished_at = finished_at.replace(tzinfo=timezone.utc)
            if finished_at < cutoff:
                stale_ids.append(job_id)
        for job_id in stale_ids:
            job = self._meteogram_warm_jobs.pop(job_id, None)
            if not job:
                continue
            key = job.get("key")
            if key in self._meteogram_warm_index:
                self._meteogram_warm_index.pop(key, None)

    def _background_fetch_job(self, key: Tuple[str, ...]) -> None:
        if len(key) >= 6:
            dataset_id, type_id, variable_id, time_operator, init_str, lead_hour = key[:6]
        else:
            dataset_id, type_id, variable_id, init_str, lead_hour = key[:5]
            time_operator = "none"
        try:
            if str(variable_id) == "wind_vectors":
                self.get_wind_vectors(
                    dataset_id, init_str, int(lead_hour), type_id=str(type_id), time_operator=str(time_operator)
                )
            else:
                self.get_field(
                    dataset_id,
                    variable_id,
                    init_str,
                    int(lead_hour),
                    type_id=str(type_id),
                    time_operator=str(time_operator),
                )
        except Exception:
            LOGGER.exception("Background field fetch failed key=%s", key)
        finally:
            with self._background_fetch_guard:
                self._background_fetch_inflight.discard(key)

    def _prewarm_loop(self) -> None:
        while not self._prewarm_stop.is_set():
            try:
                self.prewarm_hot_cache_once()
            except Exception:
                # Keep background prewarm best-effort and non-fatal.
                pass
            self._prewarm_stop.wait(HOT_PREWARM_INTERVAL_SECONDS)

    def prewarm_hot_cache_once(self) -> None:
        # Prioritize interactive requests over proactive warming.
        if self._background_fetch_inflight:
            LOGGER.debug("Skipping hot prewarm cycle: interactive fetches in-flight=%d", len(self._background_fetch_inflight))
            return
        self.refresh_catalog(force=False, blocking=True)
        for ds in self.dataset_metas:
            init = self.latest_complete_init(ds.dataset_id)
            if not init:
                continue
            leads_available = set(self.lead_hours_for_init(ds.dataset_id, init))
            prewarm_variables = HOT_PREWARM_VARIABLES
            if HOT_PREWARM_ALL_LEADS:
                leads_to_prewarm = sorted(int(v) for v in leads_available)
            else:
                leads_to_prewarm = [int(v) for v in HOT_PREWARM_LEADS if int(v) in leads_available]
            for variable_id in prewarm_variables:
                if self._background_fetch_inflight:
                    LOGGER.debug(
                        "Aborting hot prewarm cycle early: interactive fetches in-flight=%d",
                        len(self._background_fetch_inflight),
                    )
                    return
                if variable_id not in self._variables:
                    continue
                for type_id in HOT_PREWARM_TYPES:
                    for lead in leads_to_prewarm:
                        if self._background_fetch_inflight:
                            LOGGER.debug(
                                "Aborting hot prewarm lead loop: interactive fetches in-flight=%d",
                                len(self._background_fetch_inflight),
                            )
                            return
                        try:
                            self.get_field(ds.dataset_id, variable_id, init, lead, type_id=type_id)
                        except Exception:
                            continue

    def latest_complete_init(self, dataset_id: str) -> str | None:
        init_times = self.init_times(dataset_id)
        init_to_leads = self.init_to_leads(dataset_id)
        for init in init_times:
            available = set(int(v) for v in init_to_leads.get(init, []))
            expected = set(self.expected_lead_hours_for_init(dataset_id, init))
            if expected and expected.issubset(available):
                return init
        return init_times[0] if init_times else None

    @property
    def variable_metas(self) -> List[VariableMeta]:
        return [self._variables[k] for k in sorted(self._variables.keys())]

    def variable_meta(self, variable_id: str) -> VariableMeta:
        meta = self._variables.get(variable_id)
        if meta is None:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        return meta

    def variable_grib_name(self, variable_id: str) -> str:
        meta = self.variable_meta(variable_id)
        if meta.ogd_variable:
            return meta.ogd_variable
        return " + ".join(meta.ogd_components)

    def variable_long_name(self, variable_id: str) -> str:
        meta = self.variable_meta(variable_id)
        if meta.ogd_variable:
            info = OGD_PARAMETER_INFO.get(meta.ogd_variable, {})
            return str(info.get("long_name", meta.display_name))
        if meta.ogd_components == ("U_10M", "V_10M"):
            return "Wind speed at 10 m (derived from U_10M and V_10M)"
        return meta.display_name

    def variable_standard_unit(self, variable_id: str) -> str:
        meta = self.variable_meta(variable_id)
        if meta.ogd_variable:
            info = OGD_PARAMETER_INFO.get(meta.ogd_variable, {})
            return str(info.get("standard_unit", meta.unit))
        if meta.ogd_components == ("U_10M", "V_10M"):
            return "m/s (derived)"
        return meta.unit

    def variable_lead_display_offset_hours(self, variable_id: str) -> int:
        self.variable_meta(variable_id)
        with self._display_offset_guard:
            return int(self._variable_lead_display_offsets.get(variable_id, 0))

    def _record_variable_lead_display_offset_hours(self, variable_id: str, offset_hours: int) -> None:
        self.variable_meta(variable_id)
        if not isinstance(offset_hours, int):
            return
        if abs(offset_hours) > 6:
            return
        with self._display_offset_guard:
            self._variable_lead_display_offsets[variable_id] = offset_hours

    @property
    def dataset_metas(self) -> List[DatasetMeta]:
        return [self._dataset_configs[k] for k in sorted(self._dataset_configs.keys())]

    def init_times(self, dataset_id: str) -> List[str]:
        return self._catalog_for(dataset_id)["init_times"]

    def init_to_leads(self, dataset_id: str) -> Dict[str, List[int]]:
        return self._catalog_for(dataset_id)["init_to_leads"]

    def lead_hours(self, dataset_id: str) -> List[int]:
        return self._catalog_for(dataset_id)["lead_hours"]

    def lead_hours_for_init(self, dataset_id: str, init_str: str) -> List[int]:
        catalog = self._catalog_for(dataset_id)
        return catalog["init_to_leads"].get(init_str, catalog["lead_hours"])

    def expected_lead_hours_for_init(self, dataset_id: str, init_str: str) -> List[int]:
        # Operational expectations used to label runs as complete/incomplete.
        if dataset_id == "icon-ch2-eps-control":
            return list(range(0, 121))
        if dataset_id == "icon-ch1-eps-control":
            try:
                hour = int(init_str[8:10])
            except (ValueError, IndexError):
                return list(range(0, 34))
            max_lead = 45 if hour == 3 else 33
            return list(range(0, max_lead + 1))
        return self.lead_hours_for_init(dataset_id, init_str)

    def refresh_status(self, dataset_id: str) -> Dict[str, object]:
        self._dataset_config(dataset_id)
        with self._refresh_state_guard:
            refreshing = dataset_id in self._refresh_inflight
        last = self._catalog_refreshed_at.get(dataset_id)
        return {
            "refreshing": refreshing,
            "last_refreshed_at": last.isoformat() if last else None,
        }

    def refresh_catalog(self, dataset_id: str | None = None, force: bool = False, blocking: bool = True) -> None:
        dataset_ids = [dataset_id] if dataset_id else list(self._dataset_configs.keys())
        now = datetime.now(timezone.utc)

        for ds in dataset_ids:
            if ds not in self._dataset_configs:
                continue
            last = self._catalog_refreshed_at.get(ds)
            if not force and last and (now - last).total_seconds() < CATALOG_REFRESH_SECONDS:
                continue

            if not blocking and not force:
                self._refresh_catalog_async(ds)
                continue

            with self._refresh_state_guard:
                now = datetime.now(timezone.utc)
                last = self._catalog_refreshed_at.get(ds)
                if not force and last and (now - last).total_seconds() < CATALOG_REFRESH_SECONDS:
                    continue
                if ds in self._refresh_inflight:
                    continue
                self._refresh_inflight.add(ds)

            try:
                fresh_catalog = self._load_or_refresh_catalog(self._dataset_configs[ds], force=True)
                with self._catalog_guard:
                    self._catalogs[ds] = fresh_catalog
                    self._catalog_refreshed_at[ds] = datetime.now(timezone.utc)
                self._cleanup_field_cache(force=False)
            finally:
                with self._refresh_state_guard:
                    self._refresh_inflight.discard(ds)

    def _refresh_catalog_async(self, dataset_id: str) -> None:
        with self._refresh_state_guard:
            if dataset_id in self._refresh_inflight:
                return
            self._refresh_inflight.add(dataset_id)

        def _worker() -> None:
            try:
                fresh_catalog = self._load_or_refresh_catalog(self._dataset_configs[dataset_id], force=True)
                with self._catalog_guard:
                    self._catalogs[dataset_id] = fresh_catalog
                    self._catalog_refreshed_at[dataset_id] = datetime.now(timezone.utc)
                self._cleanup_field_cache(force=False)
            finally:
                with self._refresh_state_guard:
                    self._refresh_inflight.discard(dataset_id)

        thread = threading.Thread(
            target=_worker,
            name=f"catalog-refresh-{dataset_id}",
            daemon=True,
        )
        thread.start()

    def get_field(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> np.ndarray:
        self._validate_request(dataset_id, variable_id, init_str, lead_hour, type_id, time_operator=time_operator)
        key = (dataset_id, type_id, variable_id, time_operator, init_str, lead_hour)
        recent_failure = self._recent_field_failure(key)
        if recent_failure is not None:
            raise RuntimeError(str(recent_failure.get("message", "Field unavailable")))
        if key in self._field_cache:
            return self._field_cache[key]

        key_lock = self._get_key_lock(key)
        with key_lock:
            recent_failure = self._recent_field_failure(key)
            if recent_failure is not None:
                raise RuntimeError(str(recent_failure.get("message", "Field unavailable")))
            if key in self._field_cache:
                return self._field_cache[key]

            disk_path = self._field_cache_path(
                dataset_id, type_id, variable_id, init_str, lead_hour, time_operator=time_operator
            )
            if disk_path.exists():
                loaded = load_cached_field_file(disk_path)
                if loaded is not None:
                    debug_info = load_field_debug_info(disk_path)
                    if debug_info is None:
                        # Backfill debug sidecars for legacy cache entries so
                        # provenance/source reporting remains consistent.
                        debug_info = self._build_field_debug_info_from_request(
                            dataset_id=dataset_id,
                            type_id=type_id,
                            variable_id=variable_id,
                            init_str=init_str,
                            lead_hour=lead_hour,
                            time_operator=time_operator,
                        )
                        if debug_info is not None:
                            save_field_debug_info(disk_path, debug_info)
                    # Guard against stale cached "none" fields for variables that
                    # require deaggregation from reference-time products.
                    needs_recompute = False
                    if (
                        time_operator == "none"
                        and int(lead_hour) > 0
                        and variable_id
                        in (DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS | DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS)
                    ):
                        if not debug_info or not debug_info.get("deaggregation_kind"):
                            needs_recompute = True
                    if needs_recompute:
                        try:
                            disk_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    else:
                        self._field_cache[key] = loaded
                        if debug_info is not None:
                            self._field_debug_info[key] = debug_info
                        self._enforce_memory_cache_limit()
                        self._clear_field_failure(key)
                        return loaded

            try:
                if time_operator == "none":
                    field, debug_info = self._fetch_and_regrid(
                        dataset_id, variable_id, init_str, lead_hour, type_id=type_id
                    )
                else:
                    field, debug_info = self._compute_time_operated_field(
                        dataset_id=dataset_id,
                        variable_id=variable_id,
                        init_str=init_str,
                        lead_hour=lead_hour,
                        type_id=type_id,
                        time_operator=time_operator,
                    )
                save_cached_field_file(disk_path, field)
                self._field_cache[key] = field
                if debug_info:
                    self._field_debug_info[key] = debug_info
                    save_field_debug_info(disk_path, debug_info)
                self._enforce_memory_cache_limit()
                self._clear_field_failure(key)
                return field
            except RuntimeError as exc:
                self._record_field_failure(key, str(exc))
                raise

    def get_cached_field_debug_info(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> Dict[str, object] | None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        if time_operator not in SUPPORTED_TIME_OPERATORS:
            raise ValueError(f"Unknown time_operator: {time_operator}")
        key = (dataset_id, type_id, variable_id, time_operator, init_str, lead_hour)
        info = self._field_debug_info.get(key)
        if info is not None:
            return dict(info)
        disk_path = self._field_cache_path(
            dataset_id, type_id, variable_id, init_str, lead_hour, time_operator=time_operator
        )
        if not disk_path.exists():
            return None
        info = load_field_debug_info(disk_path)
        if info is None:
            info = self._build_field_debug_info_from_request(
                dataset_id=dataset_id,
                type_id=type_id,
                variable_id=variable_id,
                init_str=init_str,
                lead_hour=lead_hour,
                time_operator=time_operator,
            )
            if info is None:
                return None
            save_field_debug_info(disk_path, info)
        self._field_debug_info[key] = info
        return dict(info)

    def get_field_failure(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> Dict[str, object] | None:
        key = (dataset_id, type_id, variable_id, time_operator, init_str, lead_hour)
        return self._recent_field_failure(key)

    def _build_field_debug_info_from_request(
        self,
        dataset_id: str,
        type_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        time_operator: str = "none",
    ) -> Dict[str, object] | None:
        if time_operator != "none":
            window_leads, kind = self._time_operator_window(dataset_id, init_str, lead_hour, time_operator)
            sources: set[str] = set()
            for src_lead in window_leads:
                src = self.get_cached_field_debug_info(
                    dataset_id=dataset_id,
                    variable_id=variable_id,
                    init_str=init_str,
                    lead_hour=src_lead,
                    type_id=type_id,
                    time_operator="none",
                )
                if src and isinstance(src.get("source_files"), list):
                    sources.update(str(v) for v in src["source_files"])
            return {
                "source_files": sorted(sources),
                "source_variables": [f"{variable_id}[{kind}]"],
                "mode": type_id,
                "time_operator": time_operator,
                "window_leads": window_leads,
                "synthetic": True,
            }

        try:
            from meteodatalab import ogd_api
        except Exception:
            return None

        cfg = self._dataset_config(dataset_id)
        variable = self.variable_meta(variable_id)
        reference_iso = init_to_iso(init_str)

        source_files: set[str] = set()
        source_variables: List[str] = []

        def _collect(ogd_variable: str, perturbed: bool) -> None:
            request = ogd_api.Request(
                collection=cfg.ogd_collection,
                variable=ogd_variable,
                reference_datetime=reference_iso,
                perturbed=perturbed,
                horizon=timedelta(hours=int(lead_hour)),
            )
            names = self._asset_filenames_for_request(ogd_api, request)
            source_files.update(names)
            source_variables.append(ogd_variable + ("(perturbed)" if perturbed else "(control)"))

        if variable.ogd_components:
            if tuple(variable.ogd_components) != ("U_10M", "V_10M"):
                return None
            if type_id == "control":
                _collect("U_10M", False)
                _collect("V_10M", False)
            else:
                _collect("U_10M", False)
                _collect("U_10M", True)
                _collect("V_10M", False)
                _collect("V_10M", True)
        else:
            if not variable.ogd_variable:
                return None
            if type_id == "control":
                _collect(variable.ogd_variable, False)
            else:
                _collect(variable.ogd_variable, False)
                _collect(variable.ogd_variable, True)
            if int(lead_hour) > 0 and (
                variable_id in DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS
                or variable_id in DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS
            ):
                previous_lead = self._previous_available_lead(dataset_id, init_str, int(lead_hour))
                if previous_lead is not None:
                    prev_request_ctrl = ogd_api.Request(
                        collection=cfg.ogd_collection,
                        variable=variable.ogd_variable,
                        reference_datetime=reference_iso,
                        perturbed=False,
                        horizon=timedelta(hours=int(previous_lead)),
                    )
                    source_files.update(self._asset_filenames_for_request(ogd_api, prev_request_ctrl))
                    source_variables.append(variable.ogd_variable + "(control-prev)")
                    if type_id != "control":
                        prev_request_ens = ogd_api.Request(
                            collection=cfg.ogd_collection,
                            variable=variable.ogd_variable,
                            reference_datetime=reference_iso,
                            perturbed=True,
                            horizon=timedelta(hours=int(previous_lead)),
                        )
                        source_files.update(self._asset_filenames_for_request(ogd_api, prev_request_ens))
                        source_variables.append(variable.ogd_variable + "(perturbed-prev)")

        if not source_files:
            return None
        return {
            "source_files": sorted(source_files),
            "source_variables": source_variables,
            "mode": type_id,
            "synthetic": True,
        }

    def get_value(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        lat: float,
        lon: float,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> float:
        field = self.get_field(
            dataset_id, variable_id, init_str, lead_hour, type_id=type_id, time_operator=time_operator
        )
        bounds = self.grid_bounds(dataset_id)
        lon_span = max(1e-9, float(bounds["max_lon"] - bounds["min_lon"]))
        lat_span = max(1e-9, float(bounds["max_lat"] - bounds["min_lat"]))

        lon_frac = (lon - bounds["min_lon"]) / lon_span
        lat_frac = (bounds["max_lat"] - lat) / lat_span

        h, w = field.shape
        x = int(np.clip(round(lon_frac * (w - 1)), 0, w - 1))
        y = int(np.clip(round(lat_frac * (h - 1)), 0, h - 1))
        return float(field[y, x])

    def get_cached_value(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        lat: float,
        lon: float,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> float | None:
        field = self.get_cached_field(
            dataset_id, variable_id, init_str, lead_hour, type_id=type_id, time_operator=time_operator
        )
        if field is None:
            return None
        bounds = self.grid_bounds(dataset_id)
        lon_span = max(1e-9, float(bounds["max_lon"] - bounds["min_lon"]))
        lat_span = max(1e-9, float(bounds["max_lat"] - bounds["min_lat"]))

        lon_frac = (lon - bounds["min_lon"]) / lon_span
        lat_frac = (bounds["max_lat"] - lat) / lat_span

        h, w = field.shape
        x = int(np.clip(round(lon_frac * (w - 1)), 0, w - 1))
        y = int(np.clip(round(lat_frac * (h - 1)), 0, h - 1))
        return float(field[y, x])

    def get_model_elevation(self, dataset_id: str, lat: float, lon: float) -> float:
        self._dataset_config(dataset_id)
        field = self._get_constant_field(dataset_id, "hsurf")
        bounds = self.grid_bounds(dataset_id)
        lon_span = max(1e-9, float(bounds["max_lon"] - bounds["min_lon"]))
        lat_span = max(1e-9, float(bounds["max_lat"] - bounds["min_lat"]))
        lon_frac = (float(lon) - float(bounds["min_lon"])) / lon_span
        lat_frac = (float(bounds["max_lat"]) - float(lat)) / lat_span
        h, w = field.shape
        x = int(np.clip(round(lon_frac * (w - 1)), 0, w - 1))
        y = int(np.clip(round(lat_frac * (h - 1)), 0, h - 1))
        return float(field[y, x])

    def grid_bounds(self, dataset_id: str | None = None) -> Dict[str, float]:
        if dataset_id is None:
            # Preserve legacy behavior for callers that do not pass dataset_id.
            dataset_id = next(iter(self._dataset_configs.keys()))
        self._dataset_config(dataset_id)
        with self._grid_bounds_guard:
            bounds = self._grid_bounds.get(dataset_id)
            if isinstance(bounds, dict):
                return dict(bounds)
        return dict(SWISS_BOUNDS)

    def target_grid_shape(self, dataset_id: str) -> Tuple[int, int]:
        return self._target_grid_shape(dataset_id)

    def get_cached_field(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> np.ndarray | None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        if time_operator not in SUPPORTED_TIME_OPERATORS:
            raise ValueError(f"Unknown time_operator: {time_operator}")
        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"]:
            return None
        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            return None
        key = (dataset_id, type_id, variable_id, time_operator, init_str, lead_hour)
        field = self._field_cache.get(key)
        if field is not None:
            return field

        disk_path = self._field_cache_path(
            dataset_id, type_id, variable_id, init_str, lead_hour, time_operator=time_operator
        )
        if not disk_path.exists():
            return None
        loaded = load_cached_field_file(disk_path)
        if loaded is None:
            return None
        self._field_cache[key] = loaded
        self._enforce_memory_cache_limit()
        return loaded

    def get_cached_wind_vectors(
        self,
        dataset_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        self._dataset_config(dataset_id)
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        if time_operator not in SUPPORTED_TIME_OPERATORS:
            raise ValueError(f"Unknown time_operator: {time_operator}")
        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"]:
            return None
        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            return None
        key = (dataset_id, type_id, "wind_vectors", time_operator, init_str, lead_hour)
        cached = self._wind_vector_cache.get(key)
        if cached is not None:
            return cached
        disk_path = self._wind_vector_cache_path(dataset_id, type_id, init_str, lead_hour, time_operator=time_operator)
        if not disk_path.exists():
            return None
        loaded = load_cached_wind_vector_file(disk_path)
        if loaded is None:
            return None
        self._wind_vector_cache[key] = loaded
        self._enforce_memory_cache_limit()
        return loaded

    def get_wind_vectors(
        self,
        dataset_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
        time_operator: str = "none",
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_request(
            dataset_id, "wind_speed_10m", init_str, lead_hour, type_id, time_operator=time_operator
        )
        key = (dataset_id, type_id, "wind_vectors", time_operator, init_str, lead_hour)
        cached = self._wind_vector_cache.get(key)
        if cached is not None:
            return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            cached = self._wind_vector_cache.get(key)
            if cached is not None:
                return cached

            disk_path = self._wind_vector_cache_path(
                dataset_id, type_id, init_str, lead_hour, time_operator=time_operator
            )
            if disk_path.exists():
                loaded = load_cached_wind_vector_file(disk_path)
                if loaded is not None:
                    self._wind_vector_cache[key] = loaded
                    return loaded

            if time_operator == "none":
                vectors = self._fetch_and_regrid_wind_vectors(dataset_id, init_str, lead_hour, type_id=type_id)
            else:
                window_leads, kind = self._time_operator_window(dataset_id, init_str, lead_hour, time_operator)
                u_stack: List[np.ndarray] = []
                v_stack: List[np.ndarray] = []
                for src_lead in window_leads:
                    u_field, v_field = self.get_wind_vectors(
                        dataset_id, init_str, src_lead, type_id=type_id, time_operator="none"
                    )
                    u_stack.append(u_field)
                    v_stack.append(v_field)
                u_arr = np.stack(u_stack, axis=0)
                v_arr = np.stack(v_stack, axis=0)
                if kind == "avg":
                    vectors = (np.nanmean(u_arr, axis=0).astype(np.float32), np.nanmean(v_arr, axis=0).astype(np.float32))
                else:
                    vectors = (np.nansum(u_arr, axis=0).astype(np.float32), np.nansum(v_arr, axis=0).astype(np.float32))
            save_cached_wind_vector_file(disk_path, vectors[0], vectors[1])
            self._wind_vector_cache[key] = vectors
            self._enforce_memory_cache_limit()
            return vectors

    def _catalog_for(self, dataset_id: str) -> Dict[str, object]:
        if dataset_id not in self._catalogs:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        return self._catalogs[dataset_id]

    def _dataset_config(self, dataset_id: str) -> DatasetMeta:
        cfg = self._dataset_configs.get(dataset_id)
        if cfg is None:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")
        return cfg

    def _validate_request(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str,
        time_operator: str = "none",
    ) -> None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        if time_operator not in SUPPORTED_TIME_OPERATORS:
            raise ValueError(f"Unknown time_operator: {time_operator}")

        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"] or lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            self.refresh_catalog(dataset_id=dataset_id, force=True)
            catalog = self._catalog_for(dataset_id)

        if init_str not in catalog["init_times"]:
            raise ValueError(f"Unknown init time: {init_str}")

        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            raise ValueError(f"Unknown lead hour {lead_hour} for init {init_str}")

    @staticmethod
    def _time_operator_cache_tag(time_operator: str) -> str:
        if time_operator == "none":
            return "opnone"
        return "op" + re.sub(r"[^a-z0-9]+", "", time_operator.lower())

    def _time_operator_window(
        self, dataset_id: str, init_str: str, lead_hour: int, time_operator: str
    ) -> Tuple[List[int], str]:
        if time_operator == "none":
            return [int(lead_hour)], "none"
        m = re.fullmatch(r"(avg|acc|min|max)_(\d+)h", str(time_operator))
        if not m:
            raise ValueError(f"Invalid time_operator format: {time_operator}")
        kind = m.group(1)
        hours = int(m.group(2))
        if hours <= 0:
            raise ValueError(f"Invalid time_operator window: {time_operator}")

        available = set(int(v) for v in self.lead_hours_for_init(dataset_id, init_str))
        start = max(0, int(lead_hour) - hours + 1)
        window = [h for h in range(start, int(lead_hour) + 1) if h in available]
        if not window:
            raise RuntimeError(
                f"No leads available to compute {time_operator} for init={init_str} lead={lead_hour}"
            )
        return window, kind

    def _compute_time_operated_field(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str,
        time_operator: str,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        window_leads, kind = self._time_operator_window(dataset_id, init_str, lead_hour, time_operator)
        stack: List[np.ndarray] = []
        source_files: set[str] = set()
        for src_lead in window_leads:
            field = self.get_field(
                dataset_id=dataset_id,
                variable_id=variable_id,
                init_str=init_str,
                lead_hour=src_lead,
                type_id=type_id,
                time_operator="none",
            )
            stack.append(field)
            src_key = (dataset_id, type_id, variable_id, "none", init_str, src_lead)
            src_info = self._field_debug_info.get(src_key)
            if src_info and isinstance(src_info.get("source_files"), list):
                source_files.update(str(v) for v in src_info["source_files"])

        data = np.stack(stack, axis=0)
        if kind == "avg":
            out = np.nanmean(data, axis=0).astype(np.float32)
        elif kind == "min":
            out = np.nanmin(data, axis=0).astype(np.float32)
        elif kind == "max":
            out = np.nanmax(data, axis=0).astype(np.float32)
        else:
            out = np.nansum(data, axis=0).astype(np.float32)
        debug_info = {
            "source_files": sorted(source_files),
            "source_variables": [variable_id],
            "mode": type_id,
            "time_operator": time_operator,
            "window_leads": window_leads,
            "synthetic": True,
        }
        return out, debug_info

    def _field_cache_path(
        self,
        dataset_id: str,
        type_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        time_operator: str = "none",
    ) -> Path:
        unit_tag = "u2" if variable_id in {"wind_speed_10m", "vmax_10m"} else "u1"
        if time_operator == "none":
            return self._field_cache_dir / (
                f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_{variable_id}_{unit_tag}_{init_str}_{lead_hour:03d}.npz"
            )
        op_tag = self._time_operator_cache_tag(time_operator)
        return self._field_cache_dir / (
            f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_{variable_id}_{unit_tag}_{op_tag}_{init_str}_{lead_hour:03d}.npz"
        )

    def _wind_vector_cache_path(
        self,
        dataset_id: str,
        type_id: str,
        init_str: str,
        lead_hour: int,
        time_operator: str = "none",
    ) -> Path:
        if time_operator == "none":
            return self._vector_cache_dir / (
                f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_wind_vectors_u2_{init_str}_{lead_hour:03d}.npz"
            )
        op_tag = self._time_operator_cache_tag(time_operator)
        return self._vector_cache_dir / (
            f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_wind_vectors_u2_{op_tag}_{init_str}_{lead_hour:03d}.npz"
        )

    def _constant_field_cache_path(self, dataset_id: str, constant_id: str) -> Path:
        return self._constant_cache_dir / f"{FIELD_CACHE_VERSION}_{dataset_id}_{constant_id}.npz"

    def _get_constant_field(self, dataset_id: str, constant_id: str) -> np.ndarray:
        key = (dataset_id, constant_id)
        cached = self._constant_fields.get(key)
        if cached is not None:
            return cached
        with self._constant_guard:
            cached = self._constant_fields.get(key)
            if cached is not None:
                return cached
            disk_path = self._constant_field_cache_path(dataset_id, constant_id)
            if disk_path.exists():
                loaded = load_cached_field_file(disk_path)
                if loaded is not None:
                    self._constant_fields[key] = loaded
                    return loaded
            field = self._fetch_constant_field(dataset_id, constant_id)
            save_cached_field_file(disk_path, field)
            self._constant_fields[key] = field
            return field

    def _fetch_constant_field(self, dataset_id: str, constant_id: str) -> np.ndarray:
        if constant_id != "hsurf":
            raise ValueError(f"Unknown constant field: {constant_id}")
        ensure_eccodes_definition_path()
        try:
            from meteodatalab import ogd_api
        except ImportError as exc:
            raise RuntimeError(
                "meteodata-lab is required for OGD ingestion. Install dependencies from requirements.txt"
            ) from exc

        cfg = self._dataset_config(dataset_id)
        init_times = self.init_times(dataset_id)
        if init_times:
            try:
                direct_field, _units, _display_offset, _info = self._fetch_direct_regridded(
                    ogd_api=ogd_api,
                    dataset_id=dataset_id,
                    ogd_collection=cfg.ogd_collection,
                    ogd_variable="HSURF",
                    reference_iso=init_to_iso(str(init_times[0])),
                    lead_hour=0,
                    perturbed=False,
                )
                return direct_field.astype(np.float32)
            except Exception as exc:
                # Fall back to constants-file loading path below, but keep a trace
                # for diagnosing elevation failures.
                LOGGER.debug(
                    "HSURF direct fetch failed for dataset=%s; falling back to constants asset: %s",
                    dataset_id,
                    exc,
                )

        source_urls: List[str] | None = None
        local_cached = self._find_cached_constants_asset_path(dataset_id)
        if local_cached is not None:
            source_urls = [local_cached.resolve().as_uri()]
        if not source_urls:
            url = self._discover_constants_asset_url(dataset_id, ogd_api=ogd_api)
            source_urls = self._materialize_asset_urls([url])
        source = ogd_api.data_source.URLDataSource(urls=source_urls)
        geo_coords = getattr(ogd_api, "_geo_coords", None)
        result = ogd_api.grib_decoder.load(source, {"param": "HSURF"}, geo_coords=geo_coords)
        if not result:
            result = ogd_api.grib_decoder.load(source, {"param": "hsurf"}, geo_coords=geo_coords)
        if not result:
            raise OGDDecodeError(f"Failed to decode HSURF from constants asset for dataset={dataset_id}")
        data_array = pick_best_array(result, "HSURF")
        field = self._regrid_data_array(data_array, dataset_id).astype(np.float32)
        return field

    def _discover_constants_asset_url(self, dataset_id: str, ogd_api=None) -> str:
        with self._constants_asset_guard:
            cached = self._constants_asset_urls.get(dataset_id)
            if cached:
                return cached

        # First fallback: derive constants URL from an OGD request asset listing.
        if ogd_api is not None:
            derived = self._derive_constants_url_from_ogd_assets(dataset_id, ogd_api)
            if derived:
                with self._constants_asset_guard:
                    self._constants_asset_urls[dataset_id] = derived
                return derived

        from_items = self._discover_constants_asset_url_from_collection_items(dataset_id)
        if from_items:
            with self._constants_asset_guard:
                self._constants_asset_urls[dataset_id] = from_items
            return from_items

        cfg = self._dataset_config(dataset_id)
        body = {"collections": [cfg.collection_id], "limit": 3000}
        features = self._search_stac_features(STAC_SEARCH_URL, body)
        token = "icon-ch1-eps" if "ch1" in dataset_id else "icon-ch2-eps"
        candidates: List[Tuple[str, str]] = []
        for feature in features:
            props = feature.get("properties", {}) if isinstance(feature, dict) else {}
            ref = str(
                props.get("forecast:reference_datetime")
                or props.get("datetime")
                or props.get("created")
                or ""
            )
            assets = feature.get("assets", {}) if isinstance(feature, dict) else {}
            if not isinstance(assets, dict):
                continue
            for asset in assets.values():
                href = str((asset or {}).get("href", "")).strip()
                if not href:
                    continue
                name = Path(urlparse(href).path).name.lower()
                if "horizontal_constants" in name and token in name:
                    candidates.append((ref, href))
        if not candidates:
            raise OGDRequestError(f"No constants asset found in STAC for dataset={dataset_id}")
        candidates.sort(key=lambda x: x[0], reverse=True)
        resolved = candidates[0][1]
        with self._constants_asset_guard:
            self._constants_asset_urls[dataset_id] = resolved
        return resolved

    def _discover_constants_asset_url_from_collection_items(self, dataset_id: str) -> str | None:
        cfg = self._dataset_config(dataset_id)
        token = "horizontal_constants_icon-ch1-eps" if "ch1" in dataset_id else "horizontal_constants_icon-ch2-eps"
        url = f"https://data.geo.admin.ch/api/stac/v1/collections/{cfg.collection_id}/items?limit=500"
        visited: set[str] = set()
        best_href = None
        best_ref = ""
        pages = 0
        while url and url not in visited and pages < 30:
            visited.add(url)
            pages += 1
            try:
                resp = requests.get(url, timeout=12)
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                break
            features = payload.get("features", [])
            if isinstance(features, list):
                for feature in features:
                    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
                    ref = str(
                        props.get("forecast:reference_datetime")
                        or props.get("datetime")
                        or props.get("created")
                        or ""
                    )
                    assets = feature.get("assets", {}) if isinstance(feature, dict) else {}
                    if not isinstance(assets, dict):
                        continue
                    for asset in assets.values():
                        href = str((asset or {}).get("href", "")).strip()
                        if not href:
                            continue
                        name = Path(urlparse(href).path).name.lower()
                        if token in name and ref >= best_ref:
                            best_ref = ref
                            best_href = href
            next_url = None
            for link in payload.get("links", []):
                if isinstance(link, dict) and link.get("rel") == "next":
                    next_url = str(link.get("href", "")).strip() or None
                    break
            url = next_url
        return best_href

    def _derive_constants_url_from_ogd_assets(self, dataset_id: str, ogd_api) -> str | None:
        cfg = self._dataset_config(dataset_id)
        init_times = self.init_times(dataset_id)
        if not init_times:
            return None
        init_str = str(init_times[0])
        leads = self.lead_hours_for_init(dataset_id, init_str)
        lead_hour = int(leads[0]) if leads else 0
        request = ogd_api.Request(
            collection=cfg.ogd_collection,
            variable=self._catalog_reference_ogd_variable,
            reference_datetime=init_to_iso(init_str),
            perturbed=False,
            horizon=timedelta(hours=lead_hour),
        )
        urls = self._safe_asset_urls_for_request(ogd_api, request)
        token = "horizontal_constants_icon-ch1-eps" if "ch1" in dataset_id else "horizontal_constants_icon-ch2-eps"
        for url in urls:
            name = Path(urlparse(str(url)).path).name.lower()
            if token in name:
                return str(url)
        return None

    def _find_cached_constants_asset_path(self, dataset_id: str) -> Path | None:
        token = "horizontal_constants_icon-ch1-eps" if "ch1" in dataset_id else "horizontal_constants_icon-ch2-eps"
        candidates: List[Path] = []
        for path in self._grib_asset_cache_dir.glob(f"*{token}*.grib2"):
            if path.is_file() and path.stat().st_size > 0:
                candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]


    def _record_field_failure(self, key: Tuple[str, ...], message: str) -> None:
        with self._field_failure_guard:
            self._field_failures[key] = {
                "message": str(message),
                "at": datetime.now(timezone.utc).isoformat(),
            }

    def _clear_field_failure(self, key: Tuple[str, ...]) -> None:
        with self._field_failure_guard:
            self._field_failures.pop(key, None)

    def _recent_field_failure(self, key: Tuple[str, ...]) -> Dict[str, object] | None:
        with self._field_failure_guard:
            info = self._field_failures.get(key)
            if not info:
                return None
            at_raw = str(info.get("at", ""))
            if at_raw:
                try:
                    at_dt = datetime.fromisoformat(at_raw)
                    if at_dt.tzinfo is None:
                        at_dt = at_dt.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - at_dt).total_seconds()
                    if age > FIELD_FAILURE_TTL_SECONDS:
                        self._field_failures.pop(key, None)
                        return None
                except ValueError:
                    pass
            return dict(info)

    def _has_recent_field_failure(self, key: Tuple[str, ...]) -> bool:
        return self._recent_field_failure(key) is not None


    def _fetch_and_regrid(
        self, dataset_id: str, variable_id: str, init_str: str, lead_hour: int, type_id: str = "control"
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        cfg = self._dataset_config(dataset_id)
        variable = self.variable_meta(variable_id)
        ensure_eccodes_definition_path()
        try:
            from meteodatalab import ogd_api
        except ImportError as exc:
            raise RuntimeError(
                "meteodata-lab is required for OGD ingestion. Install dependencies from requirements.txt"
            ) from exc

        reference_iso = init_to_iso(init_str)
        if type_id == "control":
            return self._fetch_control_field(
                ogd_api, cfg.dataset_id, cfg.ogd_collection, variable_id, variable, init_str, reference_iso, lead_hour
            )
        return self._fetch_ensemble_stat_field(
            ogd_api,
            cfg.dataset_id,
            cfg.ogd_collection,
            cfg.expected_members_total,
            variable_id,
            variable,
            init_str,
            reference_iso,
            lead_hour,
            type_id,
        )

    def _fetch_and_regrid_wind_vectors(
        self, dataset_id: str, init_str: str, lead_hour: int, type_id: str = "control"
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self._dataset_config(dataset_id)
        ensure_eccodes_definition_path()
        try:
            from meteodatalab import ogd_api
        except ImportError as exc:
            raise RuntimeError(
                "meteodata-lab is required for OGD ingestion. Install dependencies from requirements.txt"
            ) from exc

        reference_iso = init_to_iso(init_str)
        if type_id == "control":
            u_field, u_units, _u_offset, _u_info = self._fetch_direct_regridded(
                ogd_api, dataset_id, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_field, v_units, _v_offset, _v_info = self._fetch_direct_regridded(
                ogd_api, dataset_id, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
            )
            u_field = self._normalize_variable_units(u_field, "wind_speed_10m", units_hint=u_units).astype(np.float32)
            v_field = self._normalize_variable_units(v_field, "wind_speed_10m", units_hint=v_units).astype(np.float32)
            return u_field, v_field

        u_ctrl, u_ctrl_units, _u_ctrl_offset, _u_ctrl_info = self._fetch_direct_regridded(
            ogd_api, dataset_id, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
        )
        v_ctrl, v_ctrl_units, _v_ctrl_offset, _v_ctrl_info = self._fetch_direct_regridded(
            ogd_api, dataset_id, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
        )
        u_ens, u_ens_units, _u_ens_offset, _u_ens_info = self._fetch_direct_member_stack(
            ogd_api, dataset_id, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
        )
        v_ens, v_ens_units, _v_ens_offset, _v_ens_info = self._fetch_direct_member_stack(
            ogd_api, dataset_id, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
        )

        u_ctrl = self._normalize_variable_units(u_ctrl, "wind_speed_10m", units_hint=u_ctrl_units).astype(np.float32)
        v_ctrl = self._normalize_variable_units(v_ctrl, "wind_speed_10m", units_hint=v_ctrl_units).astype(np.float32)
        u_ens = self._normalize_variable_units(u_ens, "wind_speed_10m", units_hint=u_ens_units).astype(np.float32)
        v_ens = self._normalize_variable_units(v_ens, "wind_speed_10m", units_hint=v_ens_units).astype(np.float32)

        u_members = np.concatenate([u_ctrl[np.newaxis, ...], u_ens], axis=0)
        v_members = np.concatenate([v_ctrl[np.newaxis, ...], v_ens], axis=0)
        self._check_ensemble_member_count(
            dataset_id,
            cfg.expected_members_total,
            u_members.shape[0],
            "wind_speed_10m",
            init=reference_iso,
            lead_hour=lead_hour,
        )
        return reduce_members(u_members, type_id), reduce_members(v_members, type_id)

    def _fetch_control_field(
        self,
        ogd_api,
        dataset_id: str,
        ogd_collection: str,
        variable_id: str,
        variable: VariableMeta,
        init_str: str,
        reference_iso: str,
        lead_hour: int,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if variable.ogd_components:
            if tuple(variable.ogd_components) == ("U_10M", "V_10M"):
                u_field, u_units, _u_offset, u_info = self._fetch_direct_regridded(
                    ogd_api, dataset_id, ogd_collection, "U_10M", reference_iso, lead_hour
                )
                v_field, v_units, _v_offset, v_info = self._fetch_direct_regridded(
                    ogd_api, dataset_id, ogd_collection, "V_10M", reference_iso, lead_hour
                )
                u_field = self._normalize_variable_units(u_field, "wind_speed_10m", units_hint=u_units)
                v_field = self._normalize_variable_units(v_field, "wind_speed_10m", units_hint=v_units)
                field = np.sqrt(u_field * u_field + v_field * v_field).astype(np.float32)
                return field, {
                    "source_files": sorted(set(u_info.get("source_files", []) + v_info.get("source_files", []))),
                    "source_variables": ["U_10M", "V_10M"],
                    "mode": "control",
                }
            raise RuntimeError(f"Unsupported derived variable: {variable_id}")

        if not variable.ogd_variable:
            raise RuntimeError(f"Variable {variable_id} has no direct OGD mapping")

        field, units, display_offset, info = self._fetch_direct_regridded(
            ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        deagg_kind, deagg_note = self._deaggregate_kind_for_field(variable_id, info)
        if deagg_kind and deagg_note.startswith("Applied variable-level fallback"):
            LOGGER.debug(
                "Deaggregation fallback used variable=%s ogd=%s init=%s lead=%s reason=%s",
                variable_id,
                variable.ogd_variable,
                reference_iso,
                lead_hour,
                deagg_note,
            )
        if deagg_kind and lead_hour > 0:
            previous_lead = self._previous_available_lead(dataset_id, init_str, lead_hour)
            if previous_lead is not None:
                prev_field, _prev_units, _prev_display_offset, prev_info = self._fetch_direct_regridded(
                    ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, previous_lead
                )
                prev_end = field_end_step(prev_info, previous_lead)
                end_step = field_end_step(info, lead_hour)
                field = deaggregate_from_reference(field, prev_field, deagg_kind, end_step, prev_end)
                info["deaggregated"] = True
                info["deaggregation_kind"] = deagg_kind
                info["deaggregation_note"] = deagg_note
                info["deaggregation_previous_lead"] = int(previous_lead)
                info["deaggregation_window_hours"] = float(max(1.0, end_step - prev_end))
                info["source_files"] = sorted(
                    set(info.get("source_files", []) + prev_info.get("source_files", []))
                )
        self._record_variable_lead_display_offset_hours(variable_id, display_offset)
        field = self._normalize_variable_units(field, variable_id, units_hint=units)
        source_variable = str(info.get("source_variable", variable.ogd_variable))
        return field.astype(np.float32), {
            "source_files": info.get("source_files", []),
            "source_variables": [source_variable],
            "source_unit": units,
            "mode": "control",
            "display_offset_hours": display_offset,
            "aggregation_kind": info.get("aggregation_kind"),
            "aggregation_from_reference": info.get("aggregation_from_reference"),
            "aggregation_metadata_present": info.get("aggregation_metadata_present"),
            "start_step": info.get("start_step"),
            "end_step": info.get("end_step"),
            "deaggregation_kind": info.get("deaggregation_kind"),
            "deaggregation_note": info.get("deaggregation_note"),
            "deaggregation_previous_lead": info.get("deaggregation_previous_lead"),
            "deaggregation_window_hours": info.get("deaggregation_window_hours"),
        }

    def _fetch_ensemble_stat_field(
        self,
        ogd_api,
        dataset_id: str,
        ogd_collection: str,
        expected_members_total: int,
        variable_id: str,
        variable: VariableMeta,
        init_str: str,
        reference_iso: str,
        lead_hour: int,
        type_id: str,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if variable.ogd_components:
            if tuple(variable.ogd_components) != ("U_10M", "V_10M"):
                raise RuntimeError(f"Unsupported derived variable: {variable_id}")
            u_ctrl, u_ctrl_units, _u_ctrl_offset, u_ctrl_info = self._fetch_direct_regridded(
                ogd_api, dataset_id, ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_ctrl, v_ctrl_units, _v_ctrl_offset, v_ctrl_info = self._fetch_direct_regridded(
                ogd_api, dataset_id, ogd_collection, "V_10M", reference_iso, lead_hour
            )
            u_ens, u_ens_units, _u_ens_offset, u_ens_info = self._fetch_direct_member_stack(
                ogd_api, dataset_id, ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_ens, v_ens_units, _v_ens_offset, v_ens_info = self._fetch_direct_member_stack(
                ogd_api, dataset_id, ogd_collection, "V_10M", reference_iso, lead_hour
            )

            u_ctrl = self._normalize_variable_units(u_ctrl, "wind_speed_10m", units_hint=u_ctrl_units).astype(np.float32)
            v_ctrl = self._normalize_variable_units(v_ctrl, "wind_speed_10m", units_hint=v_ctrl_units).astype(np.float32)
            u_ens = self._normalize_variable_units(u_ens, "wind_speed_10m", units_hint=u_ens_units).astype(np.float32)
            v_ens = self._normalize_variable_units(v_ens, "wind_speed_10m", units_hint=v_ens_units).astype(np.float32)

            ctrl_speed = np.sqrt(u_ctrl * u_ctrl + v_ctrl * v_ctrl).astype(np.float32)
            ens_speed = np.sqrt(u_ens * u_ens + v_ens * v_ens).astype(np.float32)
            members = np.concatenate([ctrl_speed[np.newaxis, ...], ens_speed], axis=0)
            self._check_ensemble_member_count(dataset_id, expected_members_total, members.shape[0], variable_id, init=reference_iso, lead_hour=lead_hour)
            return reduce_members(members, type_id), {
                "source_files": sorted(
                    set(
                        u_ctrl_info.get("source_files", [])
                        + v_ctrl_info.get("source_files", [])
                        + u_ens_info.get("source_files", [])
                        + v_ens_info.get("source_files", [])
                    )
                ),
                "source_variables": ["U_10M", "V_10M"],
                "mode": type_id,
            }

        if not variable.ogd_variable:
            raise RuntimeError(f"Variable {variable_id} has no direct OGD mapping")

        ctrl, ctrl_units, ctrl_display_offset, ctrl_info = self._fetch_direct_regridded(
            ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        ens, ens_units, _ens_display_offset, ens_info = self._fetch_direct_member_stack(
            ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        deagg_kind, deagg_note = self._deaggregate_kind_for_field(variable_id, ctrl_info)
        if deagg_kind and deagg_note.startswith("Applied variable-level fallback"):
            LOGGER.debug(
                "Deaggregation fallback used variable=%s ogd=%s init=%s lead=%s reason=%s",
                variable_id,
                variable.ogd_variable,
                reference_iso,
                lead_hour,
                deagg_note,
            )
        if deagg_kind and lead_hour > 0:
            previous_lead = self._previous_available_lead(dataset_id, init_str, lead_hour)
            if previous_lead is not None:
                prev_ctrl, _pctrl_units, _pctrl_offset, prev_ctrl_info = self._fetch_direct_regridded(
                    ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, previous_lead
                )
                prev_ens, _pens_units, _pens_offset, prev_ens_info = self._fetch_direct_member_stack(
                    ogd_api, dataset_id, ogd_collection, variable.ogd_variable, reference_iso, previous_lead
                )
                prev_end = field_end_step(prev_ctrl_info, previous_lead)
                end_step = field_end_step(ctrl_info, lead_hour)
                ctrl = deaggregate_from_reference(ctrl, prev_ctrl, deagg_kind, end_step, prev_end)
                ens = deaggregate_from_reference(ens, prev_ens, deagg_kind, end_step, prev_end)
                ctrl_info["deaggregated"] = True
                ctrl_info["deaggregation_kind"] = deagg_kind
                ctrl_info["deaggregation_note"] = deagg_note
                ctrl_info["deaggregation_previous_lead"] = int(previous_lead)
                ctrl_info["deaggregation_window_hours"] = float(max(1.0, end_step - prev_end))
                ctrl_info["source_files"] = sorted(
                    set(ctrl_info.get("source_files", []) + prev_ctrl_info.get("source_files", []))
                )
                ens_info["source_files"] = sorted(
                    set(ens_info.get("source_files", []) + prev_ens_info.get("source_files", []))
                )
        self._record_variable_lead_display_offset_hours(variable_id, ctrl_display_offset)
        ctrl = self._normalize_variable_units(ctrl, variable_id, units_hint=ctrl_units).astype(np.float32)
        ens = self._normalize_variable_units(ens, variable_id, units_hint=ens_units).astype(np.float32)
        members = np.concatenate([ctrl[np.newaxis, ...], ens], axis=0)
        self._check_ensemble_member_count(dataset_id, expected_members_total, members.shape[0], variable_id, init=reference_iso, lead_hour=lead_hour)
        ctrl_source_variable = str(ctrl_info.get("source_variable", variable.ogd_variable))
        ens_source_variable = str(ens_info.get("source_variable", variable.ogd_variable))
        return reduce_members(members, type_id), {
            "source_files": sorted(set(ctrl_info.get("source_files", []) + ens_info.get("source_files", []))),
            "source_variables": sorted(set([ctrl_source_variable, ens_source_variable])),
            "source_unit": ctrl_units or ens_units,
            "mode": type_id,
            "display_offset_hours": ctrl_display_offset,
            "aggregation_kind": ctrl_info.get("aggregation_kind"),
            "aggregation_from_reference": ctrl_info.get("aggregation_from_reference"),
            "aggregation_metadata_present": ctrl_info.get("aggregation_metadata_present"),
            "start_step": ctrl_info.get("start_step"),
            "end_step": ctrl_info.get("end_step"),
            "deaggregation_kind": ctrl_info.get("deaggregation_kind"),
            "deaggregation_note": ctrl_info.get("deaggregation_note"),
            "deaggregation_previous_lead": ctrl_info.get("deaggregation_previous_lead"),
            "deaggregation_window_hours": ctrl_info.get("deaggregation_window_hours"),
        }

    @staticmethod
    def _check_ensemble_member_count(
        dataset_id: str,
        expected_total: int,
        actual_total: int,
        variable_id: str,
        init: str,
        lead_hour: int,
    ) -> None:
        if actual_total < expected_total:
            raise RuntimeError(
                "Incomplete ensemble for "
                f"{dataset_id} variable={variable_id} init={init} lead={lead_hour}: "
                f"expected {expected_total} members (including control), got {actual_total}."
            )

    def _fetch_direct_regridded(
        self,
        ogd_api,
        dataset_id: str,
        ogd_collection: str,
        ogd_variable: str,
        reference_iso: str,
        lead_hour: int,
        perturbed: bool = False,
    ) -> Tuple[np.ndarray, str, int, Dict[str, object]]:
        requested_lead = int(lead_hour)
        last_exc: Exception | None = None

        for effective_lead in horizon_candidates(requested_lead):
            for candidate_variable in ogd_variable_candidates(ogd_variable):
                request = ogd_api.Request(
                    collection=ogd_collection,
                    variable=candidate_variable,
                    reference_datetime=reference_iso,
                    perturbed=perturbed,
                    horizon=timedelta(hours=effective_lead),
                )
                for attempt in range(1, OGD_FETCH_RETRIES + 1):
                    try:
                        asset_urls = self._safe_asset_urls_for_request(ogd_api, request)
                        if not asset_urls:
                            raise OGDRequestError(
                                f"No OGD assets for variable={candidate_variable} ref={reference_iso} lead={effective_lead}"
                            )
                        data_array = self._load_with_decode_fallbacks(
                            ogd_api, request, original_error=None, asset_urls=asset_urls
                        )
                        field = self._regrid_data_array(data_array, dataset_id).astype(np.float32)
                        units = self._extract_units_hint(data_array, candidate_variable)
                        display_offset = self._extract_display_lead_offset_hours(data_array, requested_lead)
                        agg_meta = self._extract_aggregation_metadata(data_array)
                        return (
                            field,
                            units,
                            display_offset,
                            {
                                "source_files": self._asset_filenames_for_request(ogd_api, request),
                                "source_variable": candidate_variable,
                                "display_offset_hours": display_offset,
                                "aggregation_kind": agg_meta.get("kind"),
                                "aggregation_from_reference": agg_meta.get("from_reference"),
                                "start_step": agg_meta.get("start_step"),
                                "end_step": agg_meta.get("end_step"),
                                "aggregation_metadata_present": agg_meta.get("metadata_present"),
                            },
                        )
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= OGD_FETCH_RETRIES:
                            break
                        time.sleep(OGD_FETCH_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise OGDRequestError(
            f"OGD fetch failed for variable={ogd_variable} ref={reference_iso} lead={lead_hour} "
            f"after horizon fallbacks {OGD_HORIZON_FALLBACK_STEPS} and {OGD_FETCH_RETRIES} attempts each: {last_exc}"
        ) from last_exc

    def _fetch_direct_member_stack(
        self,
        ogd_api,
        dataset_id: str,
        ogd_collection: str,
        ogd_variable: str,
        reference_iso: str,
        lead_hour: int,
    ) -> Tuple[np.ndarray, str, int, Dict[str, object]]:
        requested_lead = int(lead_hour)
        last_exc: Exception | None = None

        for effective_lead in horizon_candidates(requested_lead):
            for candidate_variable in ogd_variable_candidates(ogd_variable):
                request = ogd_api.Request(
                    collection=ogd_collection,
                    variable=candidate_variable,
                    reference_datetime=reference_iso,
                    perturbed=True,
                    horizon=timedelta(hours=effective_lead),
                )
                for attempt in range(1, OGD_FETCH_RETRIES + 1):
                    try:
                        asset_urls = self._safe_asset_urls_for_request(ogd_api, request)
                        if not asset_urls:
                            raise OGDRequestError(
                                f"No OGD assets for variable={candidate_variable} ref={reference_iso} lead={effective_lead} (perturbed)"
                            )
                        data_array = self._load_with_decode_fallbacks(
                            ogd_api, request, original_error=None, asset_urls=asset_urls
                        )
                        members = self._regrid_member_stack(data_array, dataset_id).astype(np.float32)
                        units = self._extract_units_hint(data_array, candidate_variable)
                        display_offset = self._extract_display_lead_offset_hours(data_array, requested_lead)
                        agg_meta = self._extract_aggregation_metadata(data_array)
                        return (
                            members,
                            units,
                            display_offset,
                            {
                                "source_files": self._asset_filenames_for_request(ogd_api, request),
                                "source_variable": candidate_variable,
                                "display_offset_hours": display_offset,
                                "aggregation_kind": agg_meta.get("kind"),
                                "aggregation_from_reference": agg_meta.get("from_reference"),
                                "start_step": agg_meta.get("start_step"),
                                "end_step": agg_meta.get("end_step"),
                                "aggregation_metadata_present": agg_meta.get("metadata_present"),
                            },
                        )
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= OGD_FETCH_RETRIES:
                            break
                        time.sleep(OGD_FETCH_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise OGDRequestError(
            f"OGD ensemble fetch failed for variable={ogd_variable} ref={reference_iso} lead={lead_hour} "
            f"after horizon fallbacks {OGD_HORIZON_FALLBACK_STEPS} and {OGD_FETCH_RETRIES} attempts each: {last_exc}"
        ) from last_exc


    @staticmethod
    def _safe_asset_urls_for_request(ogd_api, request) -> List[str]:
        try:
            urls = ogd_api.get_asset_urls(request)
        except Exception:
            return []
        if not urls:
            return []
        return [str(u) for u in urls if str(u).strip()]

    def _materialize_asset_urls(self, asset_urls: List[str]) -> List[str]:
        if not GRIB_ASSET_CACHE_ENABLED:
            return [str(u) for u in asset_urls]
        work: List[Tuple[str, Path]] = []
        for url in asset_urls:
            url_text = str(url).strip()
            if not url_text:
                continue
            work.append((url_text, self._grib_asset_path_for_url(url_text)))
        futures: List[Future] = [self._ensure_grib_asset_cached(url_text, cache_path) for url_text, cache_path in work]
        for fut in futures:
            fut.result()
        resolved: List[str] = []
        for _, cache_path in work:
            resolved.append(cache_path.resolve().as_uri())
        self._cleanup_grib_asset_cache(force=False)
        return resolved

    def _grib_asset_lock_for_key(self, key: str) -> threading.Lock:
        with self._grib_asset_key_locks_guard:
            lock = self._grib_asset_key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._grib_asset_key_locks[key] = lock
            needs_prune = len(self._grib_asset_key_locks) > GRIB_ASSET_KEY_LOCKS_MAX_ENTRIES
        if needs_prune:
            self._prune_grib_asset_key_locks()
        return lock

    def _ensure_grib_asset_cached(self, url_text: str, cache_path: Path) -> Future:
        key = str(cache_path)
        lock = self._grib_asset_lock_for_key(cache_path.name)

        # Fast path: already cached.
        with lock:
            if cache_path.exists() and cache_path.stat().st_size > 0:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                done: Future = Future()
                done.set_result(cache_path)
                LOGGER.debug("GRIB asset cache hit path=%s", cache_path.name)
                return done

        with self._grib_download_futures_guard:
            existing = self._grib_download_futures.get(key)
            if existing is not None:
                return existing
            fut = self._grib_download_executor.submit(self._download_asset_to_cache, url_text, cache_path)
            self._grib_download_futures[key] = fut
            fut.add_done_callback(lambda _f, k=key: self._pop_grib_download_future(k))
            return fut

    def _pop_grib_download_future(self, key: str) -> None:
        with self._grib_download_futures_guard:
            self._grib_download_futures.pop(key, None)

    def _grib_asset_path_for_url(self, url_text: str) -> Path:
        parsed = urlparse(url_text)
        normalized = parsed.path or url_text
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]
        name_path = Path(parsed.path)
        suffix = name_path.suffix or ".bin"
        stem = name_path.stem or "asset"
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)[:80]
        return self._grib_asset_cache_dir / f"{digest}_{stem}{suffix}"

    def _download_asset_to_cache(self, url_text: str, cache_path: Path) -> None:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        lock = self._grib_asset_lock_for_key(cache_path.name)
        with lock:
            if cache_path.exists() and cache_path.stat().st_size > 0:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                LOGGER.debug("GRIB asset cache hit path=%s", cache_path.name)
                return
        try:
            resp = requests.get(url_text, stream=True, timeout=(10, 180))
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            with lock:
                os.replace(tmp_path, cache_path)
            LOGGER.debug(
                "Saved GRIB asset cache path=%s bytes=%s url=%s",
                cache_path,
                cache_path.stat().st_size if cache_path.exists() else -1,
                Path(urlparse(url_text).path).name,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_grib_asset_cache(self, force: bool = False) -> None:
        if not GRIB_ASSET_CACHE_ENABLED:
            return
        with self._grib_asset_cache_guard:
            if self._background_fetch_inflight:
                return
            with self._grib_download_futures_guard:
                if self._grib_download_futures:
                    return
            now = datetime.now(timezone.utc)
            if not force and (
                now - self._last_grib_asset_cleanup_at
            ).total_seconds() < FIELD_CACHE_CLEANUP_INTERVAL_SECONDS:
                return
            self._last_grib_asset_cleanup_at = now
            cutoff = now - timedelta(hours=GRIB_ASSET_CACHE_TTL_HOURS)
            files: List[Path] = []
            total_size = 0
            for path in self._grib_asset_cache_dir.glob("*"):
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    safe_unlink(path)
                    continue
                files.append(path)
                total_size += int(stat.st_size)
            if total_size <= GRIB_ASSET_CACHE_MAX_BYTES:
                return
            files.sort(key=lambda p: p.stat().st_atime if p.exists() else 0.0)
            for path in files:
                if total_size <= GRIB_ASSET_CACHE_MAX_BYTES:
                    break
                try:
                    size = int(path.stat().st_size)
                except OSError:
                    size = 0
                safe_unlink(path)
                total_size = max(0, total_size - size)

    def _previous_available_lead(self, dataset_id: str, init_str: str, lead_hour: int) -> int | None:
        leads = [int(v) for v in self.lead_hours_for_init(dataset_id, init_str)]
        previous = [v for v in leads if v < int(lead_hour)]
        if not previous:
            return None
        return int(max(previous))


    def _deaggregate_kind_for_field(self, variable_id: str, info: Dict[str, object]) -> Tuple[str | None, str]:
        aggregation = str(info.get("aggregation_kind", "")).strip().lower()
        from_reference = bool(info.get("aggregation_from_reference", False))
        metadata_present = bool(info.get("aggregation_metadata_present", False))
        if from_reference and aggregation in {"accum", "sum"}:
            return "accum", "GRIB metadata indicates accumulation/sum from reference time"
        if from_reference and aggregation in {"avg", "average", "mean"}:
            return "avg", "GRIB metadata indicates average/mean from reference time"
        # If GRIB metadata is present but does not indicate a from-reference
        # accumulation/average, trust metadata and do not force a variable fallback.
        if metadata_present:
            return None, ""
        if variable_id in DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS:
            return "accum", "Applied variable-level fallback accumulation de-aggregation"
        if variable_id in DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS:
            return "avg", "Applied variable-level fallback average de-aggregation"
        return None, ""


    def _normalize_variable_units(self, field: np.ndarray, variable_id: str, units_hint: str) -> np.ndarray:
        result = field.astype(np.float32)
        units = units_hint.lower().strip()
        units_compact = units.replace(" ", "")
        units_alnum = re.sub(r"[^a-z0-9]+", "", units)

        if variable_id in {"t_2m", "td_2m"}:
            if units in {"k", "kelvin"}:
                return result - 273.15
            finite = result[np.isfinite(result)]
            if finite.size == 0:
                return result
            if float(np.nanmedian(finite)) > 150.0:
                return result - 273.15
            return result

        if variable_id in {"wind_speed_10m", "vmax_10m"}:
            if units_compact in {"m/s", "ms-1", "m*s-1", "ms^-1", "meterpersecond", "metrepersecond"}:
                return result * MS_TO_KMH
            if units_alnum in {
                "ms1",
                "ms01",
                "ms001",
                "ms",
                "meterpersecond",
                "metrepersecond",
                "meterspersecond",
                "metrespersecond",
            }:
                return result * MS_TO_KMH
            if units_compact in {"kt", "knot", "knots"}:
                return result * 1.852
            # OGD wind diagnostics are expected in m/s; if unit metadata is missing/odd,
            # default to m/s to avoid under-colored wind fields with km/h color tables.
            if not units:
                return result * MS_TO_KMH
            return result

        if variable_id in {"tot_prec", "w_snow", "rain_gsp", "snow_gsp"}:
            if units_compact in {"m", "meter", "metre", "mwe", "mofwaterequivalent"}:
                return result * 1000.0
            if units_alnum in {"kgm2", "kgm02", "kgm002", "kgperm2", "kgpermeter2", "kgpersquaremeter"}:
                # 1 kg m-2 liquid water equivalent == 1 mm.
                return result
            return result

        if variable_id in {"clct", "clcl", "clcm", "clch"}:
            if units_compact in {"1", "fraction"}:
                return result * 100.0
            finite = result[np.isfinite(result)]
            if finite.size and float(np.nanmax(finite)) <= 1.2:
                return result * 100.0
            return result

        if variable_id == "pres_sfc":
            if units_compact in {"pa", "pascal", "pascals"}:
                return result / 100.0
            if units_compact in {"kpa"}:
                return result * 10.0
            return result

        if variable_id == "dursun":
            if units_compact in {"s", "sec", "second", "seconds"}:
                return result / 60.0
            if units_compact in {"h", "hr", "hour", "hours"}:
                return result * 60.0
            if not units:
                # OGD standard unit is seconds; when unit metadata is missing,
                # infer seconds for clearly non-minute ranges.
                finite = result[np.isfinite(result)]
                if finite.size and float(np.nanmax(finite)) > 120.0:
                    return result / 60.0
            return result

        return result

    def _extract_units_hint(self, data_array, ogd_variable: str) -> str:
        attrs = dict(getattr(data_array, "attrs", {}) or {})

        def _canon_key(key: object) -> str:
            text = str(key or "").strip().lower()
            if text.startswith("grib_"):
                text = text[5:]
            return text

        canon_attrs: Dict[str, object] = {}
        for key, value in attrs.items():
            ck = _canon_key(key)
            if ck and ck not in canon_attrs:
                canon_attrs[ck] = value

        for key in ("units", "unit"):
            value = canon_attrs.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text

        info = OGD_PARAMETER_INFO.get(str(ogd_variable).upper(), {})
        fallback = str(info.get("standard_unit", "")).strip()
        return fallback

    @staticmethod
    def _extract_display_lead_offset_hours(data_array, requested_lead_hour: int) -> int:
        attrs = dict(getattr(data_array, "attrs", {}) or {})

        def _parse_int_like(value) -> int | None:
            if value is None:
                return None
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, float):
                if np.isfinite(value):
                    return int(round(value))
                return None
            text = str(value).strip()
            if not text:
                return None
            if re.fullmatch(r"-?\d+", text):
                return int(text)
            iso_match = re.fullmatch(r"PT(\d+)H", text)
            if iso_match:
                return int(iso_match.group(1))
            return None

        end_step_keys = ("endStep", "end_step", "stepEnd", "step_end")
        for key in end_step_keys:
            if key in attrs:
                end_step = _parse_int_like(attrs.get(key))
                if end_step is not None:
                    offset = end_step - int(requested_lead_hour)
                    if abs(offset) <= 6:
                        return int(offset)

        step_range_keys = ("stepRange", "step_range")
        for key in step_range_keys:
            raw = attrs.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            matches = re.findall(r"\d+", text)
            if matches:
                end_step = int(matches[-1])
                offset = end_step - int(requested_lead_hour)
                if abs(offset) <= 6:
                    return int(offset)

        # Final fallback: infer from absolute timestamps if present.
        ref_raw = attrs.get("forecast_reference_time") or attrs.get("reference_time") or attrs.get("reference_datetime")
        valid_raw = attrs.get("valid_time") or attrs.get("valid_datetime")
        try:
            if ref_raw and valid_raw:
                ref_dt = datetime.fromisoformat(str(ref_raw).replace("Z", "+00:00"))
                valid_dt = datetime.fromisoformat(str(valid_raw).replace("Z", "+00:00"))
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
                else:
                    ref_dt = ref_dt.astimezone(timezone.utc)
                if valid_dt.tzinfo is None:
                    valid_dt = valid_dt.replace(tzinfo=timezone.utc)
                else:
                    valid_dt = valid_dt.astimezone(timezone.utc)
                inferred_lead = int(round((valid_dt - ref_dt).total_seconds() / 3600))
                offset = inferred_lead - int(requested_lead_hour)
                if abs(offset) <= 6:
                    return int(offset)
        except Exception:
            pass

        return 0

    @staticmethod
    def _extract_aggregation_metadata(data_array) -> Dict[str, object]:
        attrs = dict(getattr(data_array, "attrs", {}) or {})

        def _canon_key(key: object) -> str:
            text = str(key or "").strip().lower()
            if text.startswith("grib_"):
                text = text[5:]
            return text

        canon_attrs: Dict[str, object] = {}
        for key, value in attrs.items():
            ck = _canon_key(key)
            if ck and ck not in canon_attrs:
                canon_attrs[ck] = value

        def _get_attr(*keys: str):
            for key in keys:
                value = canon_attrs.get(_canon_key(key))
                if value is not None:
                    return value
            return None

        def _parse_number(value) -> float | None:
            if value is None:
                return None
            if isinstance(value, (int, float, np.integer, np.floating)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
            text = str(value).strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        kind_raw = _get_attr(
            "stepType",
            "step_type",
            "stepTypeInternal",
            "statistical_process",
            "aggregation",
            "typeOfStatisticalProcessing",
        ) or ""
        kind = str(kind_raw).strip().lower()
        if kind in {"accumulation"}:
            kind = "accum"
        elif kind in {"average", "mean"}:
            kind = "avg"
        elif kind in {"sum"}:
            kind = "sum"

        stat_proc = _get_attr("typeOfStatisticalProcessing")
        stat_proc_num = _parse_number(stat_proc)
        if not kind and stat_proc_num is not None:
            # ecCodes: 0 avg, 1 accum
            if int(stat_proc_num) == 0:
                kind = "avg"
            elif int(stat_proc_num) == 1:
                kind = "accum"

        start_step = _parse_number(
            _get_attr("startStep", "start_step", "stepStart", "step_start")
        )
        end_step = _parse_number(
            _get_attr("endStep", "end_step", "stepEnd", "step_end")
        )

        step_range_raw = _get_attr("stepRange", "step_range")
        if (start_step is None or end_step is None) and step_range_raw:
            matches = re.findall(r"-?\d+(?:\.\d+)?", str(step_range_raw))
            if len(matches) >= 2:
                if start_step is None:
                    start_step = _parse_number(matches[0])
                if end_step is None:
                    end_step = _parse_number(matches[-1])
            elif len(matches) == 1 and end_step is None:
                end_step = _parse_number(matches[0])

        from_reference = False
        if start_step is not None:
            from_reference = abs(float(start_step)) < 1e-6
        elif step_range_raw:
            text = str(step_range_raw).strip()
            from_reference = text.startswith("0-") or text == "0"

        metadata_present = bool(
            kind
            or stat_proc is not None
            or step_range_raw is not None
            or start_step is not None
            or end_step is not None
        )

        return {
            "kind": kind or None,
            "from_reference": bool(from_reference),
            "start_step": start_step,
            "end_step": end_step,
            "metadata_present": metadata_present,
        }

    @staticmethod
    def _asset_filenames_for_request(ogd_api, request) -> List[str]:
        try:
            urls = ogd_api.get_asset_urls(request)
        except Exception:
            return []
        out: List[str] = []
        for url in urls:
            path = urlparse(str(url)).path
            name = Path(path).name
            if name:
                out.append(name)
        return out

    def _load_with_decode_fallbacks(
        self,
        ogd_api,
        request,
        original_error: Exception | None,
        asset_urls: List[str] | None = None,
    ):
        urls = [str(u) for u in (asset_urls or ogd_api.get_asset_urls(request) or []) if str(u).strip()]
        if not urls:
            raise OGDRequestError(f"No OGD assets for variable={request.variable}")
        source_urls = self._materialize_asset_urls(urls)
        source = ogd_api.data_source.URLDataSource(urls=source_urls)
        geo_coords = getattr(ogd_api, "_geo_coords", None)

        candidates = decode_param_candidates(request.variable)
        for param in candidates:
            try:
                result = ogd_api.grib_decoder.load(source, {"param": param}, geo_coords=geo_coords)
            except Exception:
                continue
            if result:
                return pick_best_array(result, request.variable)

        try:
            result = ogd_api.grib_decoder.load(source, {}, geo_coords=geo_coords)
        except Exception as exc:
            try:
                return ogd_api.get_from_ogd(request)
            except Exception:
                raise OGDDecodeError(f"Failed to decode OGD asset for variable {request.variable}: {exc}") from exc

        if not result:
            try:
                return ogd_api.get_from_ogd(request)
            except Exception:
                raise OGDDecodeError(
                    f"Decoded OGD asset is empty for variable {request.variable}"
                ) from (original_error or RuntimeError("Empty decode result"))

        return pick_best_array(result, request.variable)



    def _get_key_lock(self, key: Tuple[str, ...]) -> threading.Lock:
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _regrid_data_array(self, data_array, dataset_id: str) -> np.ndarray:
        values = np.asarray(data_array).squeeze()
        lat, lon = self._extract_lat_lon(data_array, values.shape)
        return self._regrid_values(values, lat, lon, dataset_id)

    def _regrid_member_stack(self, data_array, dataset_id: str) -> np.ndarray:
        values = np.asarray(data_array)
        if values.ndim == 0:
            raise RuntimeError("Unexpected scalar ensemble payload")

        dims = list(getattr(data_array, "dims", ()))
        member_names = {"eps", "number", "member", "realization", "ensemble_member", "perturbationNumber"}
        for axis in range(values.ndim - 1, -1, -1):
            dim_name = str(dims[axis]) if axis < len(dims) else ""
            if values.shape[axis] == 1 and dim_name not in member_names:
                values = np.squeeze(values, axis=axis)
                if axis < len(dims):
                    dims.pop(axis)

        dims_t = tuple(dims)
        member_axis = member_axis(dims_t, values.ndim)
        if member_axis is None:
            raise RuntimeError(
                "Failed to identify ensemble member dimension. "
                f"dims={tuple(getattr(data_array, 'dims', ()))}, shape={values.shape}"
            )

        moved = np.moveaxis(values, member_axis, 0)
        spatial_shape = moved.shape[1:]
        lat, lon = self._extract_lat_lon(data_array, spatial_shape)
        regridded = [self._regrid_values(moved[idx], lat, lon, dataset_id) for idx in range(moved.shape[0])]
        return np.stack(regridded, axis=0).astype(np.float32)


    def _regrid_values(self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray, dataset_id: str) -> np.ndarray:
        values = np.asarray(values)

        flat_values = values.reshape(-1).astype(np.float64)
        flat_lat = lat.reshape(-1).astype(np.float64)
        flat_lon = lon.reshape(-1).astype(np.float64)

        finite = np.isfinite(flat_values) & np.isfinite(flat_lat) & np.isfinite(flat_lon)
        flat_values = flat_values[finite]
        flat_lat = flat_lat[finite]
        flat_lon = flat_lon[finite]

        self._lock_grid_bounds(dataset_id, flat_lat, flat_lon)
        bounds = self.grid_bounds(dataset_id)
        grid_width, grid_height = self._target_grid_shape(dataset_id)
        lat_edges = np.linspace(bounds["min_lat"], bounds["max_lat"], grid_height + 1)
        lon_edges = np.linspace(bounds["min_lon"], bounds["max_lon"], grid_width + 1)

        val_sum, _, _ = np.histogram2d(flat_lat, flat_lon, bins=[lat_edges, lon_edges], weights=flat_values)
        val_count, _, _ = np.histogram2d(flat_lat, flat_lon, bins=[lat_edges, lon_edges])

        with np.errstate(invalid="ignore", divide="ignore"):
            grid = val_sum / val_count

        grid = np.flipud(grid)
        grid = fill_nan_with_neighbors(grid)
        return grid

    def _target_grid_shape(self, dataset_id: str) -> Tuple[int, int]:
        cfg = self._dataset_config(dataset_id)
        spacing_km = float(getattr(cfg, "target_grid_spacing_km", 0.0) or 0.0)
        if spacing_km <= 0.0:
            return int(cfg.target_grid_width), int(cfg.target_grid_height)

        with self._grid_bounds_guard:
            cached = self._computed_target_shapes.get(dataset_id)
            if cached is not None:
                return cached
            bounds = dict(self._grid_bounds.get(dataset_id, SWISS_BOUNDS))

        min_lat = float(bounds["min_lat"])
        max_lat = float(bounds["max_lat"])
        min_lon = float(bounds["min_lon"])
        max_lon = float(bounds["max_lon"])
        lat_span = max(1e-9, max_lat - min_lat)
        lon_span = max(1e-9, max_lon - min_lon)
        mid_lat_rad = math.radians((min_lat + max_lat) * 0.5)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * max(0.2, math.cos(mid_lat_rad))
        width_km = lon_span * km_per_deg_lon
        height_km = lat_span * km_per_deg_lat
        width = int(max(64, min(4096, round(width_km / spacing_km))))
        height = int(max(64, min(4096, round(height_km / spacing_km))))
        shape = (width, height)
        with self._grid_bounds_guard:
            self._computed_target_shapes[dataset_id] = shape
        return shape

    def _lock_grid_bounds(self, dataset_id: str, lat: np.ndarray, lon: np.ndarray) -> None:
        if lat.size == 0 or lon.size == 0:
            return
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        if not (np.isfinite(lat_min) and np.isfinite(lat_max) and np.isfinite(lon_min) and np.isfinite(lon_max)):
            return
        with self._grid_bounds_guard:
            if self._grid_bounds_locked.get(dataset_id, False):
                return
            self._grid_bounds[dataset_id] = {
                "min_lat": lat_min,
                "max_lat": lat_max,
                "min_lon": lon_min,
                "max_lon": lon_max,
            }
            self._grid_bounds_locked[dataset_id] = True
            self._computed_target_shapes.pop(dataset_id, None)

    @staticmethod
    def _extract_lat_lon(data_array, value_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coord_pairs = [
            ("latitude", "longitude"),
            ("lat", "lon"),
            ("Latitude", "Longitude"),
        ]

        for lat_name, lon_name in coord_pairs:
            if lat_name in data_array.coords and lon_name in data_array.coords:
                lat = np.asarray(data_array.coords[lat_name])
                lon = np.asarray(data_array.coords[lon_name])
                if lat.shape == value_shape and lon.shape == value_shape:
                    return lat, lon
                if lat.size == np.prod(value_shape) and lon.size == np.prod(value_shape):
                    return lat.reshape(value_shape), lon.reshape(value_shape)

        raise RuntimeError(
            "Failed to extract lat/lon coordinates from forecast data. "
            "Check meteodata-lab output format for this variable."
        )


    def _catalog_cache_path(self, dataset_id: str) -> Path:
        return self._catalog_cache_dir / f"catalog_{dataset_id}.json"

    def _load_or_refresh_catalog(self, cfg: DatasetMeta, force: bool) -> Dict[str, object]:
        cached = self._load_catalog_cache(cfg)
        if not force and cached is not None:
            return cached

        discovered = self._discover_catalog(cfg)
        if discovered["init_times"]:
            if cached is not None:
                merged = self._merge_catalogs(cfg, cached, discovered)
                self._save_catalog_cache(cfg, merged)
                return merged
            self._save_catalog_cache(cfg, discovered)
            return discovered

        if cached is not None:
            return cached

        return self._fallback_catalog(cfg)

    def _load_catalog_cache(self, cfg: DatasetMeta) -> Dict[str, object] | None:
        path = self._catalog_cache_path(cfg.dataset_id)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        fetched_at = payload.get("fetched_at")
        if not fetched_at:
            return None

        try:
            fetched_dt = datetime.fromisoformat(fetched_at)
        except ValueError:
            return None
        if fetched_dt.tzinfo is None:
            fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
        else:
            fetched_dt = fetched_dt.astimezone(timezone.utc)

        if datetime.now(timezone.utc) - fetched_dt > timedelta(minutes=20):
            return None

        init_times = payload.get("init_times", [])
        lead_hours = payload.get("lead_hours", [])
        init_to_leads = payload.get("init_to_leads", {})
        if not init_times:
            return None

        if not init_to_leads:
            init_to_leads = {init: list(lead_hours) for init in init_times}
        if not lead_hours:
            lead_union = set()
            for leads in init_to_leads.values():
                lead_union.update(leads)
            lead_hours = sorted(lead_union)

        return {"init_times": init_times, "lead_hours": lead_hours, "init_to_leads": init_to_leads}

    def _save_catalog_cache(self, cfg: DatasetMeta, catalog: Dict[str, object]) -> None:
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "init_times": catalog["init_times"],
            "lead_hours": catalog["lead_hours"],
            "init_to_leads": catalog["init_to_leads"],
        }
        self._catalog_cache_path(cfg.dataset_id).write_text(json.dumps(payload))

    def _discover_catalog(self, cfg: DatasetMeta) -> Dict[str, object]:
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(hours=24)

        base_body = {
            "collections": [cfg.collection_id],
            "forecast:variable": self._catalog_reference_ogd_variable,
            "forecast:perturbed": False,
            "forecast:reference_datetime": f"{start_utc.isoformat().replace('+00:00', 'Z')}/..",
            "limit": 3000,
        }

        try:
            features = self._search_stac_features(STAC_SEARCH_URL, base_body)
        except Exception:
            return {"init_times": [], "lead_hours": [], "init_to_leads": {}}

        init_to_leads: Dict[str, set[int]] = {}
        for feature in features:
            props = feature.get("properties", {})
            ref_iso = props.get("forecast:reference_datetime")
            horizon = props.get("forecast:horizon")
            if not ref_iso:
                continue

            init_dt = datetime.fromisoformat(ref_iso.replace("Z", "+00:00"))
            if init_dt < start_utc:
                continue
            init = init_dt.strftime("%Y%m%d%H")

            init_to_leads.setdefault(init, set())
            if not horizon:
                continue
            lead_h = parse_iso_duration_hours(horizon)
            if lead_h is None:
                continue

            init_to_leads[init].add(lead_h)

        if not init_to_leads:
            return {"init_times": [], "lead_hours": [], "init_to_leads": {}}

        init_times = sorted(init_to_leads.keys(), reverse=True)
        normalized_map = {init: sorted(init_to_leads[init]) for init in init_times}
        all_leads = set()
        for leads in normalized_map.values():
            all_leads.update(leads)
        lead_hours = sorted(all_leads)

        return {"init_times": init_times, "lead_hours": lead_hours, "init_to_leads": normalized_map}

    def _merge_catalogs(self, cfg: DatasetMeta, cached: Dict[str, object], discovered: Dict[str, object]) -> Dict[str, object]:
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=26)

        cached_map: Dict[str, List[int]] = dict(cached.get("init_to_leads", {}))
        discovered_map: Dict[str, List[int]] = dict(discovered.get("init_to_leads", {}))
        merged_map = dict(cached_map)
        merged_map.update(discovered_map)

        # keep only recent runs within ~24h window (slightly padded)
        pruned_map: Dict[str, List[int]] = {}
        for init_str, leads in merged_map.items():
            try:
                init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if init_dt < cutoff:
                continue
            pruned_map[init_str] = sorted(set(int(x) for x in leads))

        if not pruned_map:
            return discovered

        init_times = sorted(pruned_map.keys(), reverse=True)
        all_leads = sorted({lead for leads in pruned_map.values() for lead in leads})

        # Guard against severe regressions from transient partial STAC responses.
        expected_cycles = max(1, int(24 / cfg.fallback_cycle_hours))
        if len(discovered.get("init_times", [])) < max(2, expected_cycles // 2) and len(cached.get("init_times", [])) >= expected_cycles - 1:
            return cached

        return {"init_times": init_times, "lead_hours": all_leads, "init_to_leads": pruned_map}

    @staticmethod
    def _search_stac_features(url: str, body: Dict[str, object]) -> List[Dict[str, object]]:
        features: List[Dict[str, object]] = []
        next_url = url
        base_body: Dict[str, object] = dict(body)
        next_body: Dict[str, object] = dict(base_body)
        visited: set[Tuple[str, str]] = set()
        max_pages = 50
        page_count = 0

        while True:
            signature = (next_url, json.dumps(next_body, sort_keys=True))
            if signature in visited:
                break
            visited.add(signature)

            response = requests.post(next_url, json=next_body, timeout=12)
            response.raise_for_status()
            payload = response.json()
            features.extend(payload.get("features", []))
            page_count += 1
            if page_count >= max_pages:
                break

            next_link = None
            for link in payload.get("links", []):
                if link.get("rel") == "next":
                    next_link = link
                    break

            if not next_link:
                break

            next_url = next_link.get("href", next_url)
            link_body = next_link.get("body", {})
            if next_link.get("merge", True):
                merged = dict(base_body)
                merged.update(link_body)
                next_body = merged
            else:
                next_body = dict(link_body)

        return features

    def _fallback_catalog(self, cfg: DatasetMeta) -> Dict[str, object]:
        now_utc = datetime.now(timezone.utc)
        latest_cycle = now_utc.replace(
            hour=(now_utc.hour // cfg.fallback_cycle_hours) * cfg.fallback_cycle_hours,
            minute=0,
            second=0,
            microsecond=0,
        )
        # Use one cycle back to avoid advertising a run that is still being published.
        latest_cycle = latest_cycle - timedelta(hours=cfg.fallback_cycle_hours)
        cycle_count = max(1, int(24 / cfg.fallback_cycle_hours))
        init_times = [
            (latest_cycle - timedelta(hours=cfg.fallback_cycle_hours * i)).strftime("%Y%m%d%H")
            for i in range(cycle_count)
        ]
        lead_hours = list(cfg.fallback_lead_hours)
        init_to_leads = {init: list(lead_hours) for init in init_times}
        return {"init_times": init_times, "lead_hours": lead_hours, "init_to_leads": init_to_leads}

    def _cleanup_field_cache(self, force: bool = False) -> None:
        with self._cleanup_guard:
            now = datetime.now(timezone.utc)
            if not force and (now - self._last_cleanup_at).total_seconds() < FIELD_CACHE_CLEANUP_INTERVAL_SECONDS:
                return
            self._last_cleanup_at = now

            keep_inits_by_dataset: Dict[str, set[str]] = {}
            for dataset_id in self._dataset_configs:
                catalog = self._catalogs.get(dataset_id, {})
                keep_inits_by_dataset[dataset_id] = set(catalog.get("init_times", []))

            cutoff = now - timedelta(hours=FIELD_CACHE_RETENTION_HOURS)
            self._prune_memory_cache(keep_inits_by_dataset, cutoff)

            for path in self._field_cache_dir.glob("*.npz"):
                parsed = self._parse_field_cache_filename(path.name)
                if parsed is None:
                    continue
                version, dataset_id, init_str = parsed

                if version != FIELD_CACHE_VERSION:
                    safe_unlink(path)
                    continue

                if dataset_id not in self._dataset_configs:
                    safe_unlink(path)
                    continue

                if keep_inits_by_dataset.get(dataset_id) and init_str not in keep_inits_by_dataset[dataset_id]:
                    safe_unlink(path)
                    continue

                try:
                    init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    safe_unlink(path)
                    continue
                if init_dt < cutoff:
                    safe_unlink(path)
            for path in self._vector_cache_dir.glob("*.npz"):
                parsed = self._parse_field_cache_filename(path.name)
                if parsed is None:
                    continue
                version, dataset_id, init_str = parsed
                if version != FIELD_CACHE_VERSION:
                    safe_unlink(path)
                    continue
                if dataset_id not in self._dataset_configs:
                    safe_unlink(path)
                    continue
                if keep_inits_by_dataset.get(dataset_id) and init_str not in keep_inits_by_dataset[dataset_id]:
                    safe_unlink(path)
                    continue
                try:
                    init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    safe_unlink(path)
                    continue
                if init_dt < cutoff:
                    safe_unlink(path)
        self._cleanup_grib_asset_cache(force=force)

    def _prune_memory_cache(self, keep_inits_by_dataset: Dict[str, set[str]], cutoff: datetime) -> None:
        drop_keys: List[Tuple[str, ...]] = []
        active_keys = list(self._field_cache.keys()) + [k for k in self._wind_vector_cache.keys() if k not in self._field_cache]
        for key in active_keys:
            if len(key) >= 6:
                dataset_id, _type_id, _variable_id, _time_operator, init_str, _lead = key[:6]
            elif len(key) >= 5:
                dataset_id, _type_id, _variable_id, init_str, _lead = key[:5]
            elif len(key) == 4:
                # Legacy in-memory key format
                dataset_id, _variable_id, init_str, _lead = key
            else:
                drop_keys.append(key)
                continue
            if dataset_id not in self._dataset_configs:
                drop_keys.append(key)
                continue
            if keep_inits_by_dataset.get(dataset_id) and init_str not in keep_inits_by_dataset[dataset_id]:
                drop_keys.append(key)
                continue
            try:
                init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                drop_keys.append(key)
                continue
            if init_dt < cutoff:
                drop_keys.append(key)

        for key in drop_keys:
            self._field_cache.pop(key, None)
            self._wind_vector_cache.pop(key, None)
        self._prune_stale_key_locks()

    def _prune_stale_key_locks(self) -> None:
        with self._key_locks_guard:
            active_size = len(self._field_cache) + len(self._wind_vector_cache)
            if len(self._key_locks) <= max(128, active_size * 2):
                return
            active_keys = set(self._field_cache.keys()) | set(self._wind_vector_cache.keys())
            stale = [key for key in self._key_locks.keys() if key not in active_keys]
            for key in stale:
                self._key_locks.pop(key, None)

    def _enforce_memory_cache_limit(self) -> None:
        """Evict oldest entries when the in-memory field/wind-vector caches exceed the limit."""
        excess = len(self._field_cache) - FIELD_CACHE_MAX_ENTRIES
        if excess > 0:
            for k in list(self._field_cache)[:excess]:
                del self._field_cache[k]
        excess = len(self._wind_vector_cache) - FIELD_CACHE_MAX_ENTRIES
        if excess > 0:
            for k in list(self._wind_vector_cache)[:excess]:
                del self._wind_vector_cache[k]

    def _prune_grib_asset_key_locks(self) -> None:
        """Prune stale entries from _grib_asset_key_locks when the dict grows too large."""
        # Acquire futures guard first to maintain consistent lock ordering and avoid
        # potential deadlocks with threads that hold _grib_download_futures_guard.
        with self._grib_download_futures_guard:
            active_keys = set(self._grib_download_futures.keys())
        with self._grib_asset_key_locks_guard:
            if len(self._grib_asset_key_locks) <= GRIB_ASSET_KEY_LOCKS_MAX_ENTRIES:
                return
            stale = [k for k in self._grib_asset_key_locks if k not in active_keys]
            for k in stale:
                self._grib_asset_key_locks.pop(k, None)

    def _parse_field_cache_filename(self, filename: str) -> Tuple[str, str, str] | None:
        if not filename.endswith(".npz"):
            return None
        stem = filename[:-4]

        version = "legacy"
        rest = stem
        m = re.match(r"^(v\d+)_(.+)$", stem)
        if m:
            version = m.group(1)
            rest = m.group(2)

        dataset_id = None
        for candidate in self._dataset_configs.keys():
            prefix = f"{candidate}_"
            if rest.startswith(prefix):
                dataset_id = candidate
                rest = rest[len(prefix) :]
                break
        if dataset_id is None:
            return None

        m2 = re.match(r"^.+_(\d{10})_(\d{3})$", rest)
        if not m2:
            return None
        init_str = m2.group(1)
        return version, dataset_id, init_str

