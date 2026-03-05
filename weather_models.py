"""Data models, constants, and exception classes for the ICON Forecast Explorer.

This module is intentionally free of heavy dependencies so it can be imported
anywhere without side-effects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

SWISS_BOUNDS = {
    "min_lat": 45.75,
    "max_lat": 47.9,
    "min_lon": 5.9,
    "max_lon": 10.7,
}

# ---------------------------------------------------------------------------
# OGD / STAC catalog
# ---------------------------------------------------------------------------

STAC_SEARCH_URL = "https://data.geo.admin.ch/api/stac/v1/search"
CATALOG_REFRESH_SECONDS = 300

# ---------------------------------------------------------------------------
# Field cache
# ---------------------------------------------------------------------------

FIELD_CACHE_VERSION = "v7"
FIELD_CACHE_RETENTION_HOURS = 30
FIELD_CACHE_CLEANUP_INTERVAL_SECONDS = 300
FIELD_CACHE_MAX_ENTRIES = int(os.getenv("FIELD_CACHE_MAX_ENTRIES", "2048"))
FIELD_FAILURE_TTL_SECONDS = int(os.getenv("FIELD_FAILURE_TTL_SECONDS", "180"))

# ---------------------------------------------------------------------------
# GRIB asset cache
# ---------------------------------------------------------------------------

GRIB_ASSET_KEY_LOCKS_MAX_ENTRIES = 256
GRIB_ASSET_CACHE_ENABLED = os.getenv("GRIB_ASSET_CACHE_ENABLED", "1").strip() == "1"
GRIB_ASSET_CACHE_TTL_HOURS = int(os.getenv("GRIB_ASSET_CACHE_TTL_HOURS", "168"))
GRIB_ASSET_CACHE_MAX_BYTES = int(os.getenv("GRIB_ASSET_CACHE_MAX_BYTES", str(24 * 1024 * 1024 * 1024)))
GRIB_DOWNLOAD_WORKERS = int(os.getenv("GRIB_DOWNLOAD_WORKERS", "24"))

# ---------------------------------------------------------------------------
# Meteogram warmup
# ---------------------------------------------------------------------------

METEOGRAM_WARM_WORKERS = int(os.getenv("METEOGRAM_WARM_WORKERS", "8"))
METEOGRAM_WARM_JOB_TTL_SECONDS = int(os.getenv("METEOGRAM_WARM_JOB_TTL_SECONDS", "3600"))
METEOGRAM_WARM_PREFETCH_ASSETS = os.getenv("METEOGRAM_WARM_PREFETCH_ASSETS", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Forecast types and time operators
# ---------------------------------------------------------------------------

SUPPORTED_FORECAST_TYPES = {"control", "mean", "median", "p10", "p90", "min", "max"}

TIME_OPERATORS = [
    {"time_operator": "none", "display_name": "None"},
    {"time_operator": "avg_3h", "display_name": "Avg 3h"},
    {"time_operator": "avg_6h", "display_name": "Avg 6h"},
    {"time_operator": "avg_12h", "display_name": "Avg 12h"},
    {"time_operator": "avg_24h", "display_name": "Avg 24h"},
    {"time_operator": "acc_3h", "display_name": "Acc 3h"},
    {"time_operator": "acc_6h", "display_name": "Acc 6h"},
    {"time_operator": "acc_12h", "display_name": "Acc 12h"},
    {"time_operator": "acc_24h", "display_name": "Acc 24h"},
    {"time_operator": "min_3h", "display_name": "Min 3h"},
    {"time_operator": "min_6h", "display_name": "Min 6h"},
    {"time_operator": "min_12h", "display_name": "Min 12h"},
    {"time_operator": "min_24h", "display_name": "Min 24h"},
    {"time_operator": "max_3h", "display_name": "Max 3h"},
    {"time_operator": "max_6h", "display_name": "Max 6h"},
    {"time_operator": "max_12h", "display_name": "Max 12h"},
    {"time_operator": "max_24h", "display_name": "Max 24h"},
]
SUPPORTED_TIME_OPERATORS = {item["time_operator"] for item in TIME_OPERATORS}

# ---------------------------------------------------------------------------
# Hot pre-warm
# ---------------------------------------------------------------------------

HOT_PREWARM_INTERVAL_SECONDS = 300
HOT_PREWARM_VARIABLES = tuple(
    v.strip()
    for v in os.getenv("HOT_PREWARM_VARIABLES", "clct,tot_prec,vmax_10m,t_2m").split(",")
    if v.strip()
)
HOT_PREWARM_TYPES = tuple(
    t.strip()
    for t in os.getenv("HOT_PREWARM_TYPES", "control,p10,p90").split(",")
    if t.strip() in SUPPORTED_FORECAST_TYPES
) or ("control",)
HOT_PREWARM_LEADS = tuple(
    int(v.strip())
    for v in os.getenv("HOT_PREWARM_LEADS", "0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48").split(",")
    if v.strip().isdigit()
) or (0, 6, 12)
HOT_PREWARM_ALL_LEADS = os.getenv("HOT_PREWARM_ALL_LEADS", "1").strip() == "1"
HOT_PREWARM_ENABLED = os.getenv("HOT_PREWARM_ENABLED", "1").strip() == "1"

# ---------------------------------------------------------------------------
# OGD fetch
# ---------------------------------------------------------------------------

OGD_FETCH_RETRIES = 3
OGD_FETCH_BASE_BACKOFF_SECONDS = 0.4
OGD_HORIZON_FALLBACK_STEPS = (0, 1, 2, 3)
BACKGROUND_FETCH_WORKERS = int(os.getenv("BACKGROUND_FETCH_WORKERS", "2"))

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

MS_TO_KMH = 3.6

# ---------------------------------------------------------------------------
# De-aggregation fallback sets
# ---------------------------------------------------------------------------

DEAGGREGATE_FALLBACK_ACCUM_VARIABLE_IDS = {
    "dursun",
    "rain_gsp",
    "snow_gsp",
    "tot_prec",
}
DEAGGREGATE_FALLBACK_AVG_VARIABLE_IDS = {
    "alhfl_s",
    "ashfl_s",
    "asob_s",
    "aswdifd_s",
    "aswdir_s",
    "athb_s",
}

# ---------------------------------------------------------------------------
# OGD parameter metadata
# ---------------------------------------------------------------------------

OGD_PARAMETER_INFO: Dict[str, Dict[str, str]] = {
    "T_2M": {"long_name": "2 m temperature", "standard_unit": "K"},
    "TD_2M": {"long_name": "2 m dew point temperature", "standard_unit": "K"},
    "PS": {"long_name": "Surface pressure", "standard_unit": "Pa"},
    "U_10M": {"long_name": "U-Component of Wind", "standard_unit": "m/s"},
    "V_10M": {"long_name": "V-Component of Wind", "standard_unit": "m/s"},
    "VMAX_10M": {"long_name": "10 m wind gust", "standard_unit": "m/s"},
    "TOT_PREC": {"long_name": "Total precipitation", "standard_unit": "kg m-2 s-1"},
    "RAIN_GSP": {"long_name": "Large-scale rain", "standard_unit": "kg m-2 s-1"},
    "CLCT": {"long_name": "Total cloud cover", "standard_unit": "%"},
    "CLCL": {"long_name": "Low cloud cover", "standard_unit": "%"},
    "CLCM": {"long_name": "Mid cloud cover", "standard_unit": "%"},
    "CLCH": {"long_name": "High cloud cover", "standard_unit": "%"},
    "CEILING": {"long_name": "Cloud ceiling", "standard_unit": "m"},
    "HZEROCL": {"long_name": "Freezing level height", "standard_unit": "m"},
    "W_SNOW": {"long_name": "Snow water equivalent", "standard_unit": "kg m-2"},
    "SNOW_GSP": {"long_name": "Large-scale snowfall water equivalent", "standard_unit": "kg m-2 s-1"},
    "SNOWLMT": {"long_name": "Snowfall limit height", "standard_unit": "m"},
    "DURSUN": {"long_name": "Sunshine duration", "standard_unit": "s"},
    "ASOB_S": {"long_name": "Net shortwave radiation flux at surface", "standard_unit": "W m-2"},
    "ATHB_S": {"long_name": "Net longwave radiation flux at surface", "standard_unit": "W m-2"},
    "ASWDIR_S": {"long_name": "Direct shortwave radiation flux at surface", "standard_unit": "W m-2"},
    "ASWDIFD_S": {"long_name": "Diffuse shortwave radiation flux at surface", "standard_unit": "W m-2"},
    "ASHFL_S": {"long_name": "Sensible heat flux at surface", "standard_unit": "W m-2"},
    "ALHFL_S": {"long_name": "Latent heat flux at surface", "standard_unit": "W m-2"},
    "CAPE_ML": {"long_name": "CAPE (mixed layer)", "standard_unit": "J/kg"},
    "CIN_ML": {"long_name": "CIN (mixed layer)", "standard_unit": "J/kg"},
}

# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class OGDIngestionError(RuntimeError):
    """Base class for OGD ingestion failures."""


class OGDRequestError(OGDIngestionError):
    """Raised when data retrieval from OGD fails."""


class OGDDecodeError(OGDIngestionError):
    """Raised when a fetched OGD payload cannot be decoded."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    target_grid_width: int = 540
    target_grid_height: int = 380
    target_grid_spacing_km: float = 0.0
