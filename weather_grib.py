"""Standalone GRIB/OGD decoding and numeric helpers for the ICON Forecast Explorer.

All functions in this module are stateless (no ``ForecastStore`` dependency) and
operate only on their explicit arguments.  They are extracted here so they can be
unit-tested and reused without importing the full ``weather_data`` module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from weather_models import OGD_HORIZON_FALLBACK_STEPS


# ---------------------------------------------------------------------------
# GRIB parameter candidate tables
# ---------------------------------------------------------------------------


def decode_param_candidates(stac_variable: str) -> List[str]:
    """Return a list of GRIB short-name candidates for *stac_variable*."""
    upper = stac_variable.upper()
    mapping = {
        "T_2M": ["T_2M", "2t", "t_2m", "2T"],
        "TD_2M": ["TD_2M", "2d", "td_2m", "2D"],
        "PRES_SFC": ["PRES_SFC", "PS", "ps", "sp", "pres_sfc", "sfc_pressure"],
        "PS": ["PS", "ps", "sp", "PRES_SFC", "pres_sfc", "sfc_pressure"],
        "U_10M": ["U_10M", "10u", "u10", "u_10m"],
        "V_10M": ["V_10M", "10v", "v10", "v_10m"],
        "VMAX_10M": ["VMAX_10M", "10fg", "vmax_10m", "gust"],
        "CLCT": ["CLCT", "tcc", "clct"],
        "CLCL": ["CLCL", "lcc", "clcl"],
        "CLCM": ["CLCM", "mcc", "clcm"],
        "CLCH": ["CLCH", "hcc", "clch"],
        "CEILING": ["CEILING", "ceiling"],
        "TOT_PREC": ["TOT_PREC", "tp", "tot_prec"],
        "RAIN_GSP": ["RAIN_GSP", "rain_gsp", "lsrain", "rain"],
        "W_SNOW": ["W_SNOW", "sd", "w_snow"],
        "SNOW_GSP": ["SNOW_GSP", "snow_gsp", "lssnow"],
        "SNOWLMT": ["SNOWLMT", "snowlmt", "snowlmt_h"],
        "HZEROCL": ["HZEROCL", "hzerocl", "h0cl"],
        "DURSUN": ["DURSUN", "dursun", "sunshine_duration"],
        "ASOB_S": ["ASOB_S", "asob_s"],
        "ATHB_S": ["ATHB_S", "athb_s"],
        "ASWDIR_S": ["ASWDIR_S", "aswdir_s"],
        "ASWDIFD_S": ["ASWDIFD_S", "aswdifd_s"],
        "ASHFL_S": ["ASHFL_S", "ashfl_s"],
        "ALHFL_S": ["ALHFL_S", "alhfl_s"],
        "CAPE_ML": ["CAPE_ML", "cape_ml", "cape"],
        "CIN_ML": ["CIN_ML", "cin_ml", "cin"],
        "HSURF": ["HSURF", "hsurf", "orog", "z"],
    }
    return mapping.get(upper, [stac_variable, stac_variable.lower()])


def pick_best_array(result_map: Dict[str, object], requested_variable: str):
    """Return the best-matching array from *result_map* for *requested_variable*.

    Raises :class:`RuntimeError` when no match is found.
    """
    if requested_variable in result_map:
        return result_map[requested_variable]

    aliases = {
        "T_2M": ["2t", "t_2m", "2T"],
        "TD_2M": ["2d", "td_2m", "2D"],
        "PRES_SFC": ["PS", "ps", "sp", "pres_sfc", "sfc_pressure"],
        "PS": ["ps", "sp", "PRES_SFC", "pres_sfc", "sfc_pressure"],
        "U_10M": ["10u", "u10", "u_10m"],
        "V_10M": ["10v", "v10", "v_10m"],
        "VMAX_10M": ["10fg", "vmax_10m", "gust"],
        "CLCT": ["tcc", "clct"],
        "CLCL": ["lcc", "clcl"],
        "CLCM": ["mcc", "clcm"],
        "CLCH": ["hcc", "clch"],
        "CEILING": ["ceiling"],
        "TOT_PREC": ["tp", "tot_prec"],
        "RAIN_GSP": ["rain_gsp", "lsrain", "rain"],
        "W_SNOW": ["sd", "w_snow"],
        "SNOW_GSP": ["snow_gsp", "lssnow"],
        "SNOWLMT": ["snowlmt", "snowlmt_h"],
        "HZEROCL": ["hzerocl", "h0cl"],
        "DURSUN": ["dursun", "sunshine_duration"],
        "ASOB_S": ["asob_s"],
        "ATHB_S": ["athb_s"],
        "ASWDIR_S": ["aswdir_s"],
        "ASWDIFD_S": ["aswdifd_s"],
        "ASHFL_S": ["ashfl_s"],
        "ALHFL_S": ["alhfl_s"],
        "CAPE_ML": ["cape_ml", "cape"],
        "CIN_ML": ["cin_ml", "cin"],
        "HSURF": ["hsurf", "orog", "z"],
    }
    for alias in aliases.get(requested_variable.upper(), []):
        if alias in result_map:
            return result_map[alias]

    available = ", ".join(sorted(str(k) for k in result_map.keys()))
    raise RuntimeError(
        f"Requested variable {requested_variable} not found in decoded GRIB payload. "
        f"Available keys: {available}"
    )


# ---------------------------------------------------------------------------
# OGD variable candidate lists
# ---------------------------------------------------------------------------


def ogd_variable_candidates(ogd_variable: str) -> List[str]:
    """Return a deduplicated list of OGD variable name candidates for *ogd_variable*."""
    primary = str(ogd_variable or "").strip()
    if not primary:
        return []
    upper = primary.upper()
    aliases = {
        "PS": ["PS", "PRES_SFC", "pres_sfc", "sp", "ps"],
        "PRES_SFC": ["PRES_SFC", "PS", "pres_sfc", "sp", "ps"],
        "TOT_PREC": ["TOT_PREC", "tot_prec", "TP", "tp"],
        "T_2M": ["T_2M", "t_2m", "2t", "2T"],
        "TD_2M": ["TD_2M", "td_2m", "2d", "2D"],
        "U_10M": ["U_10M", "u_10m", "10u", "u10"],
        "V_10M": ["V_10M", "v_10m", "10v", "v10"],
        "VMAX_10M": ["VMAX_10M", "vmax_10m", "10fg"],
        "W_SNOW": ["W_SNOW", "w_snow", "sd"],
        "SNOW": ["SNOW", "snow", "sf"],
    }
    candidates = aliases.get(upper, [primary, upper, primary.lower()])
    deduped: List[str] = []
    seen: set[str] = set()
    for token in candidates:
        value = str(token).strip()
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


# ---------------------------------------------------------------------------
# Lead-hour horizon candidates
# ---------------------------------------------------------------------------


def horizon_candidates(requested_lead: int) -> List[int]:
    """Return a list of horizon (lead-hour) fallback values for *requested_lead*."""
    candidates: List[int] = []
    for step in OGD_HORIZON_FALLBACK_STEPS:
        lead = int(requested_lead + int(step))
        if lead < 0:
            continue
        if lead not in candidates:
            candidates.append(lead)
    return candidates


# ---------------------------------------------------------------------------
# Ensemble member reduction
# ---------------------------------------------------------------------------


def reduce_members(members: np.ndarray, type_id: str) -> np.ndarray:
    """Reduce a ``(member, y, x)`` stack to a single ``(y, x)`` array.

    Parameters
    ----------
    members:
        Three-dimensional float array with shape ``(member, y, x)``.
    type_id:
        One of ``"mean"``, ``"median"``, ``"p10"``, ``"p90"``, ``"min"``,
        ``"max"``.

    Raises
    ------
    RuntimeError
        If *members* does not have three dimensions.
    ValueError
        If *type_id* is not a supported statistic.
    """
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
    if type_id == "min":
        return np.nanmin(members, axis=0).astype(np.float32)
    if type_id == "max":
        return np.nanmax(members, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported ensemble statistic: {type_id}")


# ---------------------------------------------------------------------------
# Ensemble member axis detection
# ---------------------------------------------------------------------------


def member_axis(dims: Tuple[str, ...], ndim: int) -> int | None:
    """Return the axis index for the member/ensemble dimension, or ``None``."""
    candidates = {"eps", "number", "member", "realization", "ensemble_member", "perturbationNumber"}
    for idx, dim in enumerate(dims):
        if str(dim) in candidates:
            return idx
    # Be strict to avoid reducing over the wrong axis when metadata changes.
    return None


# ---------------------------------------------------------------------------
# Grid NaN interpolation
# ---------------------------------------------------------------------------


def fill_nan_with_neighbors(grid: np.ndarray) -> np.ndarray:
    """Fill ``NaN`` cells in *grid* by averaging finite neighbours (up to 8 passes)."""
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


# ---------------------------------------------------------------------------
# ecCodes definition path
# ---------------------------------------------------------------------------


def ensure_eccodes_definition_path() -> None:
    """Set ``ECCODES_DEFINITION_PATH`` to the COSMO definitions directory if not already set."""
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


# ---------------------------------------------------------------------------
# De-aggregation helper
# ---------------------------------------------------------------------------


def field_end_step(info: Dict[str, object], fallback_lead_hour: int) -> float:
    """Return the GRIB ``end_step`` value from *info*, falling back to *fallback_lead_hour*."""
    raw = info.get("end_step")
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return float(fallback_lead_hour)


def deaggregate_from_reference(
    current: np.ndarray, previous: np.ndarray, kind: str, end_step: float, previous_end_step: float
) -> np.ndarray:
    """De-aggregate *current* from a reference-time accumulation/average.

    Parameters
    ----------
    current:
        Field at ``end_step``.
    previous:
        Field at ``previous_end_step``.
    kind:
        ``"avg"`` for averages, anything else for simple differences.
    end_step:
        Step of *current* in hours.
    previous_end_step:
        Step of *previous* in hours.
    """
    window = max(1.0, float(end_step) - float(previous_end_step))
    if kind == "avg":
        return ((current * float(end_step)) - (previous * float(previous_end_step))) / window
    return current - previous
