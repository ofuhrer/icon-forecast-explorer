# AGENTS.md

Project continuity notes for coding agents.

## Scope

Applies to the whole repository.

## Architecture

- Backend:
  - `app.py`: FastAPI app, endpoint layer, logging setup, colormap loading, tile rendering.
  - `weather_data.py`: catalog refresh/discovery, OGD fetches, field/value/wind-vector caches, unit normalization, time-operator and de-aggregation logic.
- Frontend:
  - `static/index.html`: layout and static asset version query strings.
  - `static/main.js`: app state, control wiring, map layer logic, hover/value calls, loading overlay, wind vectors, meteogram orchestration.
  - `static/styles.css`: control panel + map overlays (summary, legend, centered loading status).
- Tests:
  - `tests/test_weather_data.py`
  - `tests/test_api_endpoints.py`
  - `tests/test_api_series.py`

## Current Product Behavior (Important)

- Variables are sorted alphabetically in the UI (`sortedVariables()` in `static/main.js`).
- Legend is in-map (lower-left), no legend title.
- Sidebar "Valid Time" box is removed; valid time is shown in map summary (`Valid time: ...`).
- Map summary lines are:
  1. `[Variable name] ([GRIB_NAME], [UNIT])`
  2. `[Model] [Forecast] +[Lead]h`
  3. `Valid time: ...` (red)
  4. `[Statistic], [Time Operator]`
- Loading overlay is centered on map and delayed to avoid flicker:
  - normal mode delay
  - longer delay while animating
- Wind vectors appear only for `wind_speed_10m`; they are removed/hidden for other variables.
- While animation is running, expensive non-essential requests are throttled/suppressed (hover value and debug probing).

## Time Operators

- Supported operators are defined in `weather_data.py` (`TIME_OPERATORS`, `SUPPORTED_TIME_OPERATORS`):
  - `none`
  - `avg_3h`, `avg_6h`, `avg_12h`, `avg_24h`
  - `acc_3h`, `acc_6h`, `acc_12h`, `acc_24h`
  - `min_3h`, `min_6h`, `min_12h`, `min_24h`
  - `max_3h`, `max_6h`, `max_12h`, `max_24h`
- API and cache keys include `time_operator`.
- UI lead slider is quantized by selected operator period (`static/js/time_operator.js` + lead filtering in `static/main.js`).

## Unit Normalization and De-Aggregation (Critical)

- Unit conversion is centralized in `weather_data.py` (`_normalize_variable_units`).
- De-aggregation from reference-time products relies primarily on GRIB metadata:
  - aggregation kind + from-reference flags extracted from GRIB attrs.
  - fallback allowlists exist for known products if metadata is insufficient.
- `dursun` is displayed in minutes in UI; conversion from seconds must remain correct.
- Any change touching units/de-aggregation must be validated with both map values and meteogram values.

## OGD / Catalog Mapping Notes

- Some variables may be unavailable in OGD depending on run/catalog state.
- Current behavior expects unavailable assets to surface clearly (`/api/field-debug`, loading/unavailable messaging) without UI deadlock.
- Keep error handling non-blocking and avoid infinite “loading” states when asset truly missing.

## Runtime Configuration

- `LOG_LEVEL` is preferred.
- `ICON_LOG_LEVEL` is legacy fallback only.
- `ICON_LOG_FILE` default: `logs/icon_forecast.log`.
- Also used:
  - `CORS_ALLOW_ORIGINS`
  - `ICON_EXPLORER_ALLOW_ALL_CORS`
  - `HOT_PREWARM_ENABLED`
  - `HOT_PREWARM_VARIABLES`
  - `BACKGROUND_FETCH_WORKERS`

## Repository Layout / Files to Keep Clean

- Colormap manifest is top-level `colormaps.json`.
- Screenshot is top-level `screenshot.png`.
- `scripts/` and `docs/` folders were intentionally removed.
- Runtime data should not be committed:
  - `cache/`
  - `logs/`
  - `*.log`
- If a log file becomes tracked again, untrack with:
  - `git rm --cached logs/icon_forecast.log`

## Run / Test Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Debug:

```bash
LOG_LEVEL=DEBUG uvicorn app:app --reload
```

Tests:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

Quick syntax checks:

```bash
python -m py_compile app.py weather_data.py
node --check static/main.js
```

## API Stability Requirements

Keep behavior stable for:

- `/api/metadata`
- `/api/tiles/{dataset_id}/{variable_id}/{init}/{lead}/{z}/{x}/{y}.png`
- `/api/value`
- `/api/series`
- `/api/prefetch`
- `/api/field-debug`
- `/api/wind-vectors`

## Frontend Change Checklist

When editing UI/JS/CSS:

1. If static files changed, update version query strings in `static/index.html`.
2. Verify:
   - variable switch behavior
   - wind vectors appear/disappear correctly
   - animation still smooth (no spinner flicker)
   - tooltip values and meteogram values remain consistent
3. Hard refresh browser to bypass stale static cache.

## Backend Change Checklist

When editing fetch/cache/unit logic:

1. Validate normal path + cache-miss path.
2. Validate corrupt cache file handling still drops bad files.
3. Validate unit conversion for representative variables, especially:
   - wind (`wind_speed_10m`, `vmax_10m`)
   - pressure (`pres_sfc`)
   - sunshine (`dursun`)
   - accum/average products (`tot_prec`, radiative/flux products).
4. Validate time operators (`avg/acc/min/max`) at early and later leads.

## Editing Guidance

- Make focused, minimal changes.
- Preserve query parameter compatibility.
- Keep logging informative but avoid noisy temporary debug output in final state.
- Update `README.md` whenever user-visible behavior changes.
