"""Input validation helpers for ICON Forecast Explorer API endpoints."""

from __future__ import annotations

from typing import List

from fastapi import HTTPException

from weather_data import SUPPORTED_TIME_OPERATORS

FORECAST_TYPES = [
    {"type_id": "control", "display_name": "Control"},
    {"type_id": "mean", "display_name": "Mean"},
    {"type_id": "median", "display_name": "Median"},
    {"type_id": "p10", "display_name": "10% Percentile"},
    {"type_id": "p90", "display_name": "90% Percentile"},
]
FORECAST_TYPE_IDS = {item["type_id"] for item in FORECAST_TYPES}
SERIES_FORECAST_TYPE_IDS = FORECAST_TYPE_IDS | {"min", "max"}


def _validate_time_operator(value: str) -> str:
    if not isinstance(value, str):
        default = getattr(value, "default", None)
        if isinstance(default, str):
            value = default
    time_operator = str(value or "none").strip() or "none"
    if time_operator not in SUPPORTED_TIME_OPERATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown time_operator: {time_operator}",
        )
    return time_operator


def _parse_requested_types(types: str) -> List[str]:
    req_types = [t.strip() for t in str(types).split(",") if t.strip()]
    if not req_types:
        req_types = ["control"]
    unknown_types = [t for t in req_types if t not in SERIES_FORECAST_TYPE_IDS]
    if unknown_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type_id values: {', '.join(unknown_types)}",
        )
    return req_types


def _parse_requested_variables(variables: str, store=None) -> List[str]:
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
