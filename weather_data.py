from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import threading
import sys
import time
from urllib.parse import urlparse
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

SWISS_BOUNDS = {
    "min_lat": 45.75,
    "max_lat": 47.9,
    "min_lon": 5.9,
    "max_lon": 10.7,
}

STAC_SEARCH_URL = "https://data.geo.admin.ch/api/stac/v1/search"
CATALOG_REFRESH_SECONDS = 300
FIELD_CACHE_VERSION = "v4"
MS_TO_KMH = 3.6
FIELD_CACHE_RETENTION_HOURS = 30
FIELD_CACHE_CLEANUP_INTERVAL_SECONDS = 300
SUPPORTED_FORECAST_TYPES = {"control", "mean", "median", "p10", "p90"}
HOT_PREWARM_INTERVAL_SECONDS = 300
HOT_PREWARM_VARIABLES = tuple(
    v.strip()
    for v in os.getenv("HOT_PREWARM_VARIABLES", "t_2m,tot_prec").split(",")
    if v.strip()
)
HOT_PREWARM_TYPES = ("control",)
HOT_PREWARM_LEADS = (0, 6, 12)
HOT_PREWARM_ENABLED = os.getenv("HOT_PREWARM_ENABLED", "1").strip() == "1"
OGD_FETCH_RETRIES = 3
OGD_FETCH_BASE_BACKOFF_SECONDS = 0.4
BACKGROUND_FETCH_WORKERS = int(os.getenv("BACKGROUND_FETCH_WORKERS", "2"))
LOGGER = logging.getLogger("icon_forecast.weather_data")


class OGDIngestionError(RuntimeError):
    """Base class for OGD ingestion failures."""


class OGDRequestError(OGDIngestionError):
    """Raised when data retrieval from OGD fails."""


class OGDDecodeError(OGDIngestionError):
    """Raised when a fetched OGD payload cannot be decoded."""


@dataclass(frozen=True)
class VariableMeta:
    variable_id: str
    display_name: str
    unit: str
    min_value: float
    max_value: float
    ogd_variable: str | None = None
    ogd_components: Tuple[str, ...] = ()
    lead_time_display_offset_hours: int = 0


@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str
    display_name: str
    collection_id: str
    ogd_collection: str
    expected_members_total: int
    fallback_cycle_hours: int
    fallback_lead_hours: List[int]


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
            "snow": VariableMeta(
                variable_id="snow",
                display_name="Snowfall amount",
                unit="mm",
                min_value=0.0,
                max_value=400.0,
                ogd_variable="SNOW",
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
            ),
            "icon-ch2-eps-control": DatasetMeta(
                dataset_id="icon-ch2-eps-control",
                display_name="ICON-CH2-EPS",
                collection_id="ch.meteoschweiz.ogd-forecasting-icon-ch2",
                ogd_collection="ogd-forecasting-icon-ch2",
                expected_members_total=21,
                fallback_cycle_hours=6,
                fallback_lead_hours=list(range(0, 121)),
            ),
        }

        self._grid_width = 540
        self._grid_height = 380

        self._cache_dir = Path("cache")
        self._field_cache_dir = self._cache_dir / "fields"
        self._vector_cache_dir = self._cache_dir / "vectors"
        self._catalog_cache_dir = self._cache_dir / "catalogs"
        self._field_cache_dir.mkdir(parents=True, exist_ok=True)
        self._vector_cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_cache_dir.mkdir(parents=True, exist_ok=True)

        self._field_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._wind_vector_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self._field_debug_info: Dict[Tuple[str, ...], Dict[str, object]] = {}
        self._key_locks: Dict[Tuple[str, ...], threading.Lock] = {}
        self._key_locks_guard = threading.Lock()
        self._catalog_guard = threading.Lock()
        self._refresh_state_guard = threading.Lock()
        self._cleanup_guard = threading.Lock()
        self._prewarm_guard = threading.Lock()
        self._display_offset_guard = threading.Lock()
        self._background_fetch_guard = threading.Lock()
        self._background_fetch_inflight: set[Tuple[str, ...]] = set()
        self._background_fetch_executor = ThreadPoolExecutor(
            max_workers=max(1, BACKGROUND_FETCH_WORKERS),
            thread_name_prefix="field-fetch",
        )

        self._catalogs: Dict[str, Dict[str, object]] = {}
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
        self._background_fetch_executor.shutdown(wait=False, cancel_futures=False)
        LOGGER.info("Stopped background workers")

    def queue_field_fetch(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
    ) -> bool:
        self._validate_request(dataset_id, variable_id, init_str, lead_hour, type_id)
        key = (dataset_id, type_id, variable_id, init_str, lead_hour)
        if key in self._field_cache:
            return False
        if self._field_cache_path(dataset_id, type_id, variable_id, init_str, lead_hour).exists():
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
    ) -> bool:
        self._validate_request(dataset_id, "wind_speed_10m", init_str, lead_hour, type_id)
        key = (dataset_id, type_id, "wind_vectors", init_str, lead_hour)
        if key in self._wind_vector_cache:
            return False
        if self._wind_vector_cache_path(dataset_id, type_id, init_str, lead_hour).exists():
            return False

        with self._background_fetch_guard:
            if key in self._background_fetch_inflight:
                return False
            self._background_fetch_inflight.add(key)

        self._background_fetch_executor.submit(self._background_fetch_job, key)
        LOGGER.debug("Queued wind vector fetch key=%s", key)
        return True

    def _background_fetch_job(self, key: Tuple[str, ...]) -> None:
        dataset_id, type_id, variable_id, init_str, lead_hour = key
        try:
            if str(variable_id) == "wind_vectors":
                self.get_wind_vectors(dataset_id, init_str, int(lead_hour), type_id=str(type_id))
            else:
                self.get_field(dataset_id, variable_id, init_str, int(lead_hour), type_id=str(type_id))
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
            if ds.dataset_id == "icon-ch2-eps-control":
                prewarm_variables = ("t_2m", "tot_prec")
            else:
                prewarm_variables = HOT_PREWARM_VARIABLES
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
                    for lead in HOT_PREWARM_LEADS:
                        if self._background_fetch_inflight:
                            LOGGER.debug(
                                "Aborting hot prewarm lead loop: interactive fetches in-flight=%d",
                                len(self._background_fetch_inflight),
                            )
                            return
                        if lead not in leads_available:
                            continue
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

    def get_field(self, dataset_id: str, variable_id: str, init_str: str, lead_hour: int, type_id: str = "control") -> np.ndarray:
        self._validate_request(dataset_id, variable_id, init_str, lead_hour, type_id)
        key = (dataset_id, type_id, variable_id, init_str, lead_hour)
        if key in self._field_cache:
            return self._field_cache[key]

        key_lock = self._get_key_lock(key)
        with key_lock:
            if key in self._field_cache:
                return self._field_cache[key]

            disk_path = self._field_cache_path(dataset_id, type_id, variable_id, init_str, lead_hour)
            if disk_path.exists():
                loaded = self._load_cached_field_file(disk_path)
                if loaded is not None:
                    self._field_cache[key] = loaded
                    debug_info = self._load_field_debug_info(disk_path)
                    if debug_info is not None:
                        self._field_debug_info[key] = debug_info
                    return loaded

            field, debug_info = self._fetch_and_regrid(dataset_id, variable_id, init_str, lead_hour, type_id=type_id)
            self._save_cached_field_file(disk_path, field)
            self._field_cache[key] = field
            if debug_info:
                self._field_debug_info[key] = debug_info
                self._save_field_debug_info(disk_path, debug_info)
            return field

    def get_cached_field_debug_info(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
    ) -> Dict[str, object] | None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        key = (dataset_id, type_id, variable_id, init_str, lead_hour)
        info = self._field_debug_info.get(key)
        if info is not None:
            return dict(info)
        disk_path = self._field_cache_path(dataset_id, type_id, variable_id, init_str, lead_hour)
        if not disk_path.exists():
            return None
        info = self._load_field_debug_info(disk_path)
        if info is None:
            info = self._build_field_debug_info_from_request(
                dataset_id=dataset_id,
                type_id=type_id,
                variable_id=variable_id,
                init_str=init_str,
                lead_hour=lead_hour,
            )
            if info is None:
                return None
            self._save_field_debug_info(disk_path, info)
        self._field_debug_info[key] = info
        return dict(info)

    def _build_field_debug_info_from_request(
        self,
        dataset_id: str,
        type_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
    ) -> Dict[str, object] | None:
        try:
            from meteodatalab import ogd_api
        except Exception:
            return None

        cfg = self._dataset_config(dataset_id)
        variable = self.variable_meta(variable_id)
        reference_iso = self._init_to_iso(init_str)

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
    ) -> float:
        field = self.get_field(dataset_id, variable_id, init_str, lead_hour, type_id=type_id)

        lon_frac = (lon - SWISS_BOUNDS["min_lon"]) / (SWISS_BOUNDS["max_lon"] - SWISS_BOUNDS["min_lon"])
        lat_frac = (SWISS_BOUNDS["max_lat"] - lat) / (SWISS_BOUNDS["max_lat"] - SWISS_BOUNDS["min_lat"])

        x = int(np.clip(round(lon_frac * (self._grid_width - 1)), 0, self._grid_width - 1))
        y = int(np.clip(round(lat_frac * (self._grid_height - 1)), 0, self._grid_height - 1))
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
    ) -> float | None:
        field = self.get_cached_field(dataset_id, variable_id, init_str, lead_hour, type_id=type_id)
        if field is None:
            return None

        lon_frac = (lon - SWISS_BOUNDS["min_lon"]) / (SWISS_BOUNDS["max_lon"] - SWISS_BOUNDS["min_lon"])
        lat_frac = (SWISS_BOUNDS["max_lat"] - lat) / (SWISS_BOUNDS["max_lat"] - SWISS_BOUNDS["min_lat"])

        x = int(np.clip(round(lon_frac * (self._grid_width - 1)), 0, self._grid_width - 1))
        y = int(np.clip(round(lat_frac * (self._grid_height - 1)), 0, self._grid_height - 1))
        return float(field[y, x])

    def get_cached_field(
        self,
        dataset_id: str,
        variable_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
    ) -> np.ndarray | None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"]:
            return None
        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            return None
        key = (dataset_id, type_id, variable_id, init_str, lead_hour)
        field = self._field_cache.get(key)
        if field is not None:
            return field

        disk_path = self._field_cache_path(dataset_id, type_id, variable_id, init_str, lead_hour)
        if not disk_path.exists():
            return None
        loaded = self._load_cached_field_file(disk_path)
        if loaded is None:
            return None
        self._field_cache[key] = loaded
        return loaded

    def get_cached_wind_vectors(
        self,
        dataset_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        self._dataset_config(dataset_id)
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")
        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"]:
            return None
        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            return None
        key = (dataset_id, type_id, "wind_vectors", init_str, lead_hour)
        cached = self._wind_vector_cache.get(key)
        if cached is not None:
            return cached
        disk_path = self._wind_vector_cache_path(dataset_id, type_id, init_str, lead_hour)
        if not disk_path.exists():
            return None
        loaded = self._load_cached_wind_vector_file(disk_path)
        if loaded is None:
            return None
        self._wind_vector_cache[key] = loaded
        return loaded

    def get_wind_vectors(
        self,
        dataset_id: str,
        init_str: str,
        lead_hour: int,
        type_id: str = "control",
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_request(dataset_id, "wind_speed_10m", init_str, lead_hour, type_id)
        key = (dataset_id, type_id, "wind_vectors", init_str, lead_hour)
        cached = self._wind_vector_cache.get(key)
        if cached is not None:
            return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            cached = self._wind_vector_cache.get(key)
            if cached is not None:
                return cached

            disk_path = self._wind_vector_cache_path(dataset_id, type_id, init_str, lead_hour)
            if disk_path.exists():
                loaded = self._load_cached_wind_vector_file(disk_path)
                if loaded is not None:
                    self._wind_vector_cache[key] = loaded
                    return loaded

            vectors = self._fetch_and_regrid_wind_vectors(dataset_id, init_str, lead_hour, type_id=type_id)
            self._save_cached_wind_vector_file(disk_path, vectors[0], vectors[1])
            self._wind_vector_cache[key] = vectors
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

    def _validate_request(self, dataset_id: str, variable_id: str, init_str: str, lead_hour: int, type_id: str) -> None:
        self._dataset_config(dataset_id)
        if variable_id not in self._variables:
            raise ValueError(f"Unknown variable_id: {variable_id}")
        if type_id not in SUPPORTED_FORECAST_TYPES:
            raise ValueError(f"Unknown type_id: {type_id}")

        catalog = self._catalog_for(dataset_id)
        if init_str not in catalog["init_times"] or lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            self.refresh_catalog(dataset_id=dataset_id, force=True)
            catalog = self._catalog_for(dataset_id)

        if init_str not in catalog["init_times"]:
            raise ValueError(f"Unknown init time: {init_str}")

        if lead_hour not in self.lead_hours_for_init(dataset_id, init_str):
            raise ValueError(f"Unknown lead hour {lead_hour} for init {init_str}")

    def _field_cache_path(self, dataset_id: str, type_id: str, variable_id: str, init_str: str, lead_hour: int) -> Path:
        unit_tag = "u2" if variable_id in {"wind_speed_10m", "vmax_10m"} else "u1"
        return self._field_cache_dir / (
            f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_{variable_id}_{unit_tag}_{init_str}_{lead_hour:03d}.npz"
        )

    def _wind_vector_cache_path(self, dataset_id: str, type_id: str, init_str: str, lead_hour: int) -> Path:
        return self._vector_cache_dir / (
            f"{FIELD_CACHE_VERSION}_{dataset_id}_{type_id}_wind_vectors_u2_{init_str}_{lead_hour:03d}.npz"
        )

    @staticmethod
    def _field_debug_path(field_cache_path: Path) -> Path:
        return field_cache_path.with_suffix(".json")

    def _load_field_debug_info(self, field_cache_path: Path) -> Dict[str, object] | None:
        path = self._field_debug_path(field_cache_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_field_debug_info(self, field_cache_path: Path, debug_info: Dict[str, object]) -> None:
        path = self._field_debug_path(field_cache_path)
        try:
            path.write_text(json.dumps(debug_info))
        except OSError:
            LOGGER.warning("Failed to write field debug info path=%s", path)
            pass

    @staticmethod
    def _load_cached_field_file(path: Path) -> np.ndarray | None:
        try:
            return np.load(path)["field"]
        except (EOFError, OSError, zipfile.BadZipFile, KeyError, ValueError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            LOGGER.warning("Dropped corrupt/incomplete field cache file path=%s", path)
            return None

    @staticmethod
    def _save_cached_field_file(path: Path, field: np.ndarray) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as tmp_file:
            np.savez_compressed(tmp_file, field=field)
        os.replace(tmp_path, path)
        LOGGER.debug("Saved field cache path=%s bytes=%s", path, path.stat().st_size if path.exists() else -1)

    @staticmethod
    def _load_cached_wind_vector_file(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
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

    @staticmethod
    def _save_cached_wind_vector_file(path: Path, u_field: np.ndarray, v_field: np.ndarray) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as tmp_file:
            np.savez_compressed(tmp_file, u=u_field, v=v_field)
        os.replace(tmp_path, path)
        LOGGER.debug("Saved wind vector cache path=%s bytes=%s", path, path.stat().st_size if path.exists() else -1)

    def _fetch_and_regrid(
        self, dataset_id: str, variable_id: str, init_str: str, lead_hour: int, type_id: str = "control"
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        cfg = self._dataset_config(dataset_id)
        variable = self.variable_meta(variable_id)
        self._ensure_eccodes_definition_path()
        try:
            from meteodatalab import ogd_api
        except ImportError as exc:
            raise RuntimeError(
                "meteodata-lab is required for OGD ingestion. Install dependencies from requirements.txt"
            ) from exc

        reference_iso = self._init_to_iso(init_str)
        if type_id == "control":
            return self._fetch_control_field(ogd_api, cfg.ogd_collection, variable_id, variable, reference_iso, lead_hour)
        return self._fetch_ensemble_stat_field(
            ogd_api,
            cfg.dataset_id,
            cfg.ogd_collection,
            cfg.expected_members_total,
            variable_id,
            variable,
            reference_iso,
            lead_hour,
            type_id,
        )

    def _fetch_and_regrid_wind_vectors(
        self, dataset_id: str, init_str: str, lead_hour: int, type_id: str = "control"
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self._dataset_config(dataset_id)
        self._ensure_eccodes_definition_path()
        try:
            from meteodatalab import ogd_api
        except ImportError as exc:
            raise RuntimeError(
                "meteodata-lab is required for OGD ingestion. Install dependencies from requirements.txt"
            ) from exc

        reference_iso = self._init_to_iso(init_str)
        if type_id == "control":
            u_field, u_units, _u_offset, _u_info = self._fetch_direct_regridded(
                ogd_api, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_field, v_units, _v_offset, _v_info = self._fetch_direct_regridded(
                ogd_api, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
            )
            u_field = self._normalize_variable_units(u_field, "wind_speed_10m", units_hint=u_units).astype(np.float32)
            v_field = self._normalize_variable_units(v_field, "wind_speed_10m", units_hint=v_units).astype(np.float32)
            return u_field, v_field

        u_ctrl, u_ctrl_units, _u_ctrl_offset, _u_ctrl_info = self._fetch_direct_regridded(
            ogd_api, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
        )
        v_ctrl, v_ctrl_units, _v_ctrl_offset, _v_ctrl_info = self._fetch_direct_regridded(
            ogd_api, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
        )
        u_ens, u_ens_units, _u_ens_offset, _u_ens_info = self._fetch_direct_member_stack(
            ogd_api, cfg.ogd_collection, "U_10M", reference_iso, lead_hour
        )
        v_ens, v_ens_units, _v_ens_offset, _v_ens_info = self._fetch_direct_member_stack(
            ogd_api, cfg.ogd_collection, "V_10M", reference_iso, lead_hour
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
        return self._reduce_members(u_members, type_id), self._reduce_members(v_members, type_id)

    def _fetch_control_field(
        self, ogd_api, ogd_collection: str, variable_id: str, variable: VariableMeta, reference_iso: str, lead_hour: int
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if variable.ogd_components:
            if tuple(variable.ogd_components) == ("U_10M", "V_10M"):
                u_field, u_units, _u_offset, u_info = self._fetch_direct_regridded(
                    ogd_api, ogd_collection, "U_10M", reference_iso, lead_hour
                )
                v_field, v_units, _v_offset, v_info = self._fetch_direct_regridded(
                    ogd_api, ogd_collection, "V_10M", reference_iso, lead_hour
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
            ogd_api, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        self._record_variable_lead_display_offset_hours(variable_id, display_offset)
        field = self._normalize_variable_units(field, variable_id, units_hint=units)
        return field.astype(np.float32), {
            "source_files": info.get("source_files", []),
            "source_variables": [variable.ogd_variable],
            "mode": "control",
            "display_offset_hours": display_offset,
        }

    def _fetch_ensemble_stat_field(
        self,
        ogd_api,
        dataset_id: str,
        ogd_collection: str,
        expected_members_total: int,
        variable_id: str,
        variable: VariableMeta,
        reference_iso: str,
        lead_hour: int,
        type_id: str,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if variable.ogd_components:
            if tuple(variable.ogd_components) != ("U_10M", "V_10M"):
                raise RuntimeError(f"Unsupported derived variable: {variable_id}")
            u_ctrl, u_ctrl_units, _u_ctrl_offset, u_ctrl_info = self._fetch_direct_regridded(
                ogd_api, ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_ctrl, v_ctrl_units, _v_ctrl_offset, v_ctrl_info = self._fetch_direct_regridded(
                ogd_api, ogd_collection, "V_10M", reference_iso, lead_hour
            )
            u_ens, u_ens_units, _u_ens_offset, u_ens_info = self._fetch_direct_member_stack(
                ogd_api, ogd_collection, "U_10M", reference_iso, lead_hour
            )
            v_ens, v_ens_units, _v_ens_offset, v_ens_info = self._fetch_direct_member_stack(
                ogd_api, ogd_collection, "V_10M", reference_iso, lead_hour
            )

            u_ctrl = self._normalize_variable_units(u_ctrl, "wind_speed_10m", units_hint=u_ctrl_units).astype(np.float32)
            v_ctrl = self._normalize_variable_units(v_ctrl, "wind_speed_10m", units_hint=v_ctrl_units).astype(np.float32)
            u_ens = self._normalize_variable_units(u_ens, "wind_speed_10m", units_hint=u_ens_units).astype(np.float32)
            v_ens = self._normalize_variable_units(v_ens, "wind_speed_10m", units_hint=v_ens_units).astype(np.float32)

            ctrl_speed = np.sqrt(u_ctrl * u_ctrl + v_ctrl * v_ctrl).astype(np.float32)
            ens_speed = np.sqrt(u_ens * u_ens + v_ens * v_ens).astype(np.float32)
            members = np.concatenate([ctrl_speed[np.newaxis, ...], ens_speed], axis=0)
            self._check_ensemble_member_count(dataset_id, expected_members_total, members.shape[0], variable_id, init=reference_iso, lead_hour=lead_hour)
            return self._reduce_members(members, type_id), {
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
            ogd_api, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        ens, ens_units, _ens_display_offset, ens_info = self._fetch_direct_member_stack(
            ogd_api, ogd_collection, variable.ogd_variable, reference_iso, lead_hour
        )
        self._record_variable_lead_display_offset_hours(variable_id, ctrl_display_offset)
        ctrl = self._normalize_variable_units(ctrl, variable_id, units_hint=ctrl_units).astype(np.float32)
        ens = self._normalize_variable_units(ens, variable_id, units_hint=ens_units).astype(np.float32)
        members = np.concatenate([ctrl[np.newaxis, ...], ens], axis=0)
        self._check_ensemble_member_count(dataset_id, expected_members_total, members.shape[0], variable_id, init=reference_iso, lead_hour=lead_hour)
        return self._reduce_members(members, type_id), {
            "source_files": sorted(set(ctrl_info.get("source_files", []) + ens_info.get("source_files", []))),
            "source_variables": [variable.ogd_variable],
            "mode": type_id,
            "display_offset_hours": ctrl_display_offset,
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
        ogd_collection: str,
        ogd_variable: str,
        reference_iso: str,
        lead_hour: int,
        perturbed: bool = False,
    ) -> Tuple[np.ndarray, str, int, Dict[str, object]]:
        request = ogd_api.Request(
            collection=ogd_collection,
            variable=ogd_variable,
            reference_datetime=reference_iso,
            perturbed=perturbed,
            horizon=timedelta(hours=int(lead_hour)),
        )
        last_exc: Exception | None = None
        for attempt in range(1, OGD_FETCH_RETRIES + 1):
            try:
                try:
                    data_array = ogd_api.get_from_ogd(request)
                except KeyError as exc:
                    data_array = self._load_with_decode_fallbacks(ogd_api, request, exc)
                field = self._regrid_data_array(data_array).astype(np.float32)
                units = str(data_array.attrs.get("units", ""))
                display_offset = self._extract_display_lead_offset_hours(data_array, int(lead_hour))
                return (
                    field,
                    units,
                    display_offset,
                    {
                        "source_files": self._asset_filenames_for_request(ogd_api, request),
                        "display_offset_hours": display_offset,
                    },
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= OGD_FETCH_RETRIES:
                    break
                time.sleep(OGD_FETCH_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise OGDRequestError(
            f"OGD fetch failed for variable={ogd_variable} ref={reference_iso} lead={lead_hour} "
            f"after {OGD_FETCH_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _fetch_direct_member_stack(
        self,
        ogd_api,
        ogd_collection: str,
        ogd_variable: str,
        reference_iso: str,
        lead_hour: int,
    ) -> Tuple[np.ndarray, str, int, Dict[str, object]]:
        request = ogd_api.Request(
            collection=ogd_collection,
            variable=ogd_variable,
            reference_datetime=reference_iso,
            perturbed=True,
            horizon=timedelta(hours=int(lead_hour)),
        )
        last_exc: Exception | None = None
        for attempt in range(1, OGD_FETCH_RETRIES + 1):
            try:
                try:
                    data_array = ogd_api.get_from_ogd(request)
                except KeyError as exc:
                    data_array = self._load_with_decode_fallbacks(ogd_api, request, exc)
                members = self._regrid_member_stack(data_array).astype(np.float32)
                units = str(data_array.attrs.get("units", ""))
                display_offset = self._extract_display_lead_offset_hours(data_array, int(lead_hour))
                return (
                    members,
                    units,
                    display_offset,
                    {
                        "source_files": self._asset_filenames_for_request(ogd_api, request),
                        "display_offset_hours": display_offset,
                    },
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= OGD_FETCH_RETRIES:
                    break
                time.sleep(OGD_FETCH_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise OGDRequestError(
            f"OGD ensemble fetch failed for variable={ogd_variable} ref={reference_iso} lead={lead_hour} "
            f"after {OGD_FETCH_RETRIES} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _reduce_members(members: np.ndarray, type_id: str) -> np.ndarray:
        if members.ndim != 3:
            raise RuntimeError(f"Expected member stack with shape (member,y,x), got {members.shape}")
        if type_id == "mean":
            return np.nanmean(members, axis=0).astype(np.float32)
        if type_id == "median":
            return np.nanmedian(members, axis=0).astype(np.float32)
        if type_id == "p10":
            return np.nanpercentile(members, 10, axis=0).astype(np.float32)
        if type_id == "p90":
            return np.nanpercentile(members, 90, axis=0).astype(np.float32)
        raise ValueError(f"Unsupported ensemble statistic: {type_id}")

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

        if variable_id in {"tot_prec", "w_snow", "snow"}:
            if units_compact in {"m", "meter", "metre", "mwe", "mofwaterequivalent"}:
                return result * 1000.0
            return result

        if variable_id in {"clct", "clcl"}:
            if units_compact in {"1", "fraction"}:
                return result * 100.0
            finite = result[np.isfinite(result)]
            if finite.size and float(np.nanmax(finite)) <= 1.2:
                return result * 100.0
            return result

        if variable_id == "dursun":
            if units_compact in {"s", "sec", "second", "seconds"}:
                return result / 60.0
            if units_compact in {"h", "hr", "hour", "hours"}:
                return result * 60.0
            return result

        return result

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

    def _load_with_decode_fallbacks(self, ogd_api, request, original_error: Exception):
        urls = ogd_api.get_asset_urls(request)
        source = ogd_api.data_source.URLDataSource(urls=urls)
        geo_coords = getattr(ogd_api, "_geo_coords", None)

        candidates = self._decode_param_candidates(request.variable)
        for param in candidates:
            try:
                result = ogd_api.grib_decoder.load(source, {"param": param}, geo_coords=geo_coords)
            except Exception:
                continue
            if result:
                return self._pick_best_array(result, request.variable)

        try:
            result = ogd_api.grib_decoder.load(source, {}, geo_coords=geo_coords)
        except Exception as exc:
            raise OGDDecodeError(f"Failed to decode OGD asset for variable {request.variable}: {exc}") from exc

        if not result:
            raise OGDDecodeError(f"Decoded OGD asset is empty for variable {request.variable}") from original_error

        return self._pick_best_array(result, request.variable)

    @staticmethod
    def _decode_param_candidates(stac_variable: str) -> List[str]:
        upper = stac_variable.upper()
        mapping = {
            "T_2M": ["T_2M", "2t", "t_2m", "2T"],
            "TD_2M": ["TD_2M", "2d", "td_2m", "2D"],
            "U_10M": ["U_10M", "10u", "u10", "u_10m"],
            "V_10M": ["V_10M", "10v", "v10", "v_10m"],
            "VMAX_10M": ["VMAX_10M", "10fg", "vmax_10m", "gust"],
            "CLCT": ["CLCT", "tcc", "clct"],
            "CLCL": ["CLCL", "lcc", "clcl"],
            "CEILING": ["CEILING", "ceiling"],
            "TOT_PREC": ["TOT_PREC", "tp", "tot_prec"],
            "W_SNOW": ["W_SNOW", "sd", "w_snow"],
            "SNOW": ["SNOW", "snow", "sf"],
            "SNOWLMT": ["SNOWLMT", "snowlmt", "snowlmt_h"],
            "HZEROCL": ["HZEROCL", "hzerocl", "h0cl"],
            "DURSUN": ["DURSUN", "dursun", "sunshine_duration"],
        }
        return mapping.get(upper, [stac_variable, stac_variable.lower()])

    @staticmethod
    def _pick_best_array(result_map: Dict[str, object], requested_variable: str):
        if requested_variable in result_map:
            return result_map[requested_variable]

        aliases = {
            "T_2M": ["2t", "t_2m", "2T"],
            "TD_2M": ["2d", "td_2m", "2D"],
            "U_10M": ["10u", "u10", "u_10m"],
            "V_10M": ["10v", "v10", "v_10m"],
            "VMAX_10M": ["10fg", "vmax_10m", "gust"],
            "CLCT": ["tcc", "clct"],
            "CLCL": ["lcc", "clcl"],
            "CEILING": ["ceiling"],
            "TOT_PREC": ["tp", "tot_prec"],
            "W_SNOW": ["sd", "w_snow"],
            "SNOW": ["snow", "sf"],
            "SNOWLMT": ["snowlmt", "snowlmt_h"],
            "HZEROCL": ["hzerocl", "h0cl"],
            "DURSUN": ["dursun", "sunshine_duration"],
        }
        for alias in aliases.get(requested_variable.upper(), []):
            if alias in result_map:
                return result_map[alias]

        available = ", ".join(sorted(str(k) for k in result_map.keys()))
        raise RuntimeError(
            f"Requested variable {requested_variable} not found in decoded GRIB payload. "
            f"Available keys: {available}"
        )

    @staticmethod
    def _ensure_eccodes_definition_path() -> None:
        if os.environ.get("ECCODES_DEFINITION_PATH"):
            return

        cwd_defs = (Path.cwd() / ".venv/share/eccodes-cosmo-resources/definitions").resolve()
        prefix_defs = (Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions").resolve()

        if cwd_defs.exists():
            os.environ["ECCODES_DEFINITION_PATH"] = str(cwd_defs)
            return
        if prefix_defs.exists():
            os.environ["ECCODES_DEFINITION_PATH"] = str(prefix_defs)
            return

        raise RuntimeError(
            "Missing ecCodes COSMO definitions. Expected one of:\n"
            f" - {cwd_defs}\n"
            f" - {prefix_defs}\n"
            "Install dependencies in the project virtualenv so "
            ".venv/share/eccodes-cosmo-resources/definitions is present."
        )

    def _get_key_lock(self, key: Tuple[str, ...]) -> threading.Lock:
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _regrid_data_array(self, data_array) -> np.ndarray:
        values = np.asarray(data_array).squeeze()
        lat, lon = self._extract_lat_lon(data_array, values.shape)
        return self._regrid_values(values, lat, lon)

    def _regrid_member_stack(self, data_array) -> np.ndarray:
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
        member_axis = self._member_axis(dims_t, values.ndim)
        if member_axis is None:
            raise RuntimeError(
                "Failed to identify ensemble member dimension. "
                f"dims={tuple(getattr(data_array, 'dims', ()))}, shape={values.shape}"
            )

        moved = np.moveaxis(values, member_axis, 0)
        spatial_shape = moved.shape[1:]
        lat, lon = self._extract_lat_lon(data_array, spatial_shape)
        regridded = [self._regrid_values(moved[idx], lat, lon) for idx in range(moved.shape[0])]
        return np.stack(regridded, axis=0).astype(np.float32)

    @staticmethod
    def _member_axis(dims: Tuple[str, ...], ndim: int) -> int | None:
        candidates = {"eps", "number", "member", "realization", "ensemble_member", "perturbationNumber"}
        for idx, dim in enumerate(dims):
            if str(dim) in candidates:
                return idx
        # Be strict to avoid reducing over the wrong axis when metadata changes.
        return None

    def _regrid_values(self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        values = np.asarray(values)

        flat_values = values.reshape(-1).astype(np.float64)
        flat_lat = lat.reshape(-1).astype(np.float64)
        flat_lon = lon.reshape(-1).astype(np.float64)

        finite = np.isfinite(flat_values) & np.isfinite(flat_lat) & np.isfinite(flat_lon)
        flat_values = flat_values[finite]
        flat_lat = flat_lat[finite]
        flat_lon = flat_lon[finite]

        lat_edges = np.linspace(SWISS_BOUNDS["min_lat"], SWISS_BOUNDS["max_lat"], self._grid_height + 1)
        lon_edges = np.linspace(SWISS_BOUNDS["min_lon"], SWISS_BOUNDS["max_lon"], self._grid_width + 1)

        val_sum, _, _ = np.histogram2d(flat_lat, flat_lon, bins=[lat_edges, lon_edges], weights=flat_values)
        val_count, _, _ = np.histogram2d(flat_lat, flat_lon, bins=[lat_edges, lon_edges])

        with np.errstate(invalid="ignore", divide="ignore"):
            grid = val_sum / val_count

        grid = np.flipud(grid)
        grid = self._fill_nan_with_neighbors(grid)
        return grid

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

    @staticmethod
    def _fill_nan_with_neighbors(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        global_mean = float(np.nanmean(result)) if np.isfinite(np.nanmean(result)) else 0.0

        for _ in range(8):
            nan_mask = ~np.isfinite(result)
            if not np.any(nan_mask):
                break

            neighbor_sum = np.zeros_like(result, dtype=np.float64)
            neighbor_count = np.zeros_like(result, dtype=np.int16)

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if dy >= 0:
                    src_y = slice(0, result.shape[0] - dy)
                    dst_y = slice(dy, result.shape[0])
                else:
                    src_y = slice(-dy, result.shape[0])
                    dst_y = slice(0, result.shape[0] + dy)
                if dx >= 0:
                    src_x = slice(0, result.shape[1] - dx)
                    dst_x = slice(dx, result.shape[1])
                else:
                    src_x = slice(-dx, result.shape[1])
                    dst_x = slice(0, result.shape[1] + dx)

                neighbor = result[src_y, src_x]
                finite = np.isfinite(neighbor)
                if not np.any(finite):
                    continue
                dst_sum = neighbor_sum[dst_y, dst_x]
                dst_count = neighbor_count[dst_y, dst_x]
                dst_sum[finite] += neighbor[finite]
                dst_count[finite] += 1

            fillable = nan_mask & (neighbor_count > 0)
            result[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]

        result[~np.isfinite(result)] = global_mean
        return result.astype(np.float32)

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
            lead_h = self._parse_iso_duration_hours(horizon)
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
                    self._safe_unlink(path)
                    continue

                if dataset_id not in self._dataset_configs:
                    self._safe_unlink(path)
                    continue

                if keep_inits_by_dataset.get(dataset_id) and init_str not in keep_inits_by_dataset[dataset_id]:
                    self._safe_unlink(path)
                    continue

                try:
                    init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    self._safe_unlink(path)
                    continue
                if init_dt < cutoff:
                    self._safe_unlink(path)
            for path in self._vector_cache_dir.glob("*.npz"):
                parsed = self._parse_field_cache_filename(path.name)
                if parsed is None:
                    continue
                version, dataset_id, init_str = parsed
                if version != FIELD_CACHE_VERSION:
                    self._safe_unlink(path)
                    continue
                if dataset_id not in self._dataset_configs:
                    self._safe_unlink(path)
                    continue
                if keep_inits_by_dataset.get(dataset_id) and init_str not in keep_inits_by_dataset[dataset_id]:
                    self._safe_unlink(path)
                    continue
                try:
                    init_dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    self._safe_unlink(path)
                    continue
                if init_dt < cutoff:
                    self._safe_unlink(path)

    def _prune_memory_cache(self, keep_inits_by_dataset: Dict[str, set[str]], cutoff: datetime) -> None:
        drop_keys: List[Tuple[str, ...]] = []
        active_keys = list(self._field_cache.keys()) + [k for k in self._wind_vector_cache.keys() if k not in self._field_cache]
        for key in active_keys:
            if len(key) >= 5:
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

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _parse_iso_duration_hours(value: str) -> int | None:
        match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
        if not match:
            return None
        days = int(match.group(1) or 0)
        hours = int(match.group(2) or 0)
        minutes = int(match.group(3) or 0)
        seconds = int(match.group(4) or 0)
        total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
        return int(round(total_hours))

    @staticmethod
    def _init_to_iso(init_str: str) -> str:
        dt = datetime.strptime(init_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
