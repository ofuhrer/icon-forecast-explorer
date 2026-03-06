# AGENTS.md

Project continuity notes for coding agents.

## Scope

Applies to the whole repository.

## Architecture

- Backend:
  - `app.py`: **Thin FastAPI endpoint wiring only.** Logging setup, CORS, lifespan, and one
    handler function per API route. Imports pure helpers from the modules below; no business
    logic lives here. Run with `uvicorn app:app`.
  - `tile_rendering.py`: Pure colormap and tile-rendering functions — `render_tile_rgba()`,
    `apply_colormap()`, `_colormap_for_variable()`, `_legend_for_variable()`,
    `_load_project_colormaps()`, and the `TILE_SIZE` constant. No FastAPI dependency.
  - `grid_utils.py`: Pure coordinate-transform utilities (`_lon_to_x`, `_lat_to_y`,
    `_x_to_lon`, `_y_to_lat`, `_coord_key`) plus store-aware helpers (`_dataset_grid_shape`,
    `_effective_grid_bounds`) that accept an optional `store` argument.
  - `series_cache.py`: Thread-safe in-memory series-response cache — data structures
    (`_SERIES_CACHE`, `_SERIES_KEY_LOCKS`), constants (`SERIES_CACHE_TTL_SECONDS`,
    `SERIES_CACHE_MAX_ENTRIES`, `SERIES_LATLON_BUCKET_DEG`), and all `_series_*` helpers.
  - `input_validation.py`: FastAPI-aware input validators — `_validate_time_operator()`,
    `_parse_requested_types()`, `_parse_requested_variables()`; forecast-type constants
    (`FORECAST_TYPES`, `FORECAST_TYPE_IDS`, `SERIES_FORECAST_TYPE_IDS`).
  - `weather_models.py`: constants, `OGD_PARAMETER_INFO`, exception classes (`OGDIngestionError`
    etc.), and frozen dataclasses (`VariableMeta`, `DatasetMeta`).
  - `weather_cache.py`: standalone (no-`self`) file I/O helpers for field/wind-vector cache
    files and debug sidecars (`load_cached_field_file`, `save_cached_field_file`,
    `load_field_debug_info`, `save_field_debug_info`, `safe_unlink`,
    `parse_iso_duration_hours`, `init_to_iso`, …).
  - `weather_grib.py`: standalone GRIB/OGD decoding helpers (`decode_param_candidates`,
    `pick_best_array`, `ogd_variable_candidates`, `horizon_candidates`, `reduce_members`,
    `member_axis`, `fill_nan_with_neighbors`, `ensure_eccodes_definition_path`,
    `field_end_step`, `deaggregate_from_reference`).
  - `weather_data.py`: `ForecastStore` class — catalog refresh/discovery, OGD fetches,
    field/value/wind-vector caches, unit normalisation, time-operator and de-aggregation
    logic.  Imports and re-exports symbols from the three sub-modules above so existing
    callers need no changes.
- Frontend:
  - `static/index.html`: layout and static asset version query strings.
  - `static/main.js`: app state, control wiring, map layer logic, hover/value calls, loading
    overlay, wind vectors, meteogram orchestration.  Organised with `// ─── Section: XXX ───`
    headers.
  - `static/styles.css`: control panel + map overlays (summary, legend, centred loading
    status).
  - `static/js/escape.js`: `escapeHtml(str)` utility.
  - `static/js/search.js`: pure SwissTopo search helpers — `firstSwissTopoResult()`,
    `normalizeSwissTopoLabel()`.
  - `static/js/url_state.js`: URL state parsing — `parseUrlState()`.
  - `static/js/wind_vectors.js`: wind-arrow GeoJSON builder —
    `buildWindVectorFeatures(vectors, map)`.
  - `static/js/full_meteogram.js`: full-meteogram constants and pure utility functions
    (abort/sleep helpers, chart scale/tick formatters, warmup key).
  - `static/js/animation.js`: placeholder module (animation logic remains in `main.js`
    Animation section; candidate for future factory-pattern extraction).
  - `static/js/map_layer.js`: placeholder module (tile-layer logic remains in `main.js` Map
    Layer section; candidate for future factory-pattern extraction).
  - `static/js/formatting.js`, `static/js/time_operator.js`, `static/js/ui_text.js`,
    `static/js/meteogram.js`, `static/js/api.js`: pre-existing focused modules.
- Tests:
  - `tests/test_weather_data.py`
  - `tests/test_api_endpoints.py`
  - `tests/test_api_series.py`

## Module Dependency Map

```
app.py
 ├── tile_rendering.py   (no further project deps)
 ├── grid_utils.py
 │    └── weather_data.py  (SWISS_BOUNDS only)
 ├── series_cache.py
 │    ├── grid_utils.py
 │    └── weather_data.py  (via grid_utils; fastapi HTTPException)
 ├── input_validation.py
 │    └── weather_data.py  (SUPPORTED_TIME_OPERATORS only; fastapi HTTPException)
 └── weather_data.py
      ├── weather_models.py
      ├── weather_cache.py
      └── weather_grib.py
```

## What Goes Where

| Task | File to edit |
|------|-------------|
| Add / modify an API endpoint | `app.py` |
| Change tile rendering or colormap logic | `tile_rendering.py` |
| Change coordinate transforms or grid-bounds helpers | `grid_utils.py` |
| Change series-response caching | `series_cache.py` |
| Change input validation or forecast-type constants | `input_validation.py` |
| Change data fetching, field/value caching, or unit conversion | `weather_data.py` |
| Change data model constants or dataclasses | `weather_models.py` |
| Change GRIB decoding helpers | `weather_grib.py` |
| Change field/wind-vector file I/O cache helpers | `weather_cache.py` |
| Change frontend map/UI behaviour | `static/main.js` |
| Change focused frontend module | appropriate `static/js/*.js` file |

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
pip install -r requirements-dev.txt   # ruff linter/formatter
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
python -m py_compile app.py weather_data.py weather_models.py weather_cache.py weather_grib.py tile_rendering.py grid_utils.py series_cache.py input_validation.py
node --check static/main.js
```

Lint and format (must pass before committing):

```bash
ruff check app.py weather_data.py weather_models.py weather_cache.py weather_grib.py tile_rendering.py grid_utils.py series_cache.py input_validation.py
ruff format --check app.py weather_models.py weather_cache.py weather_grib.py tile_rendering.py grid_utils.py series_cache.py input_validation.py
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

## CI Checks

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Syntax check** — `python -m py_compile` on all Python source files.
2. **Lint** — `ruff check` on all Python source files.  Configuration lives in
   `pyproject.toml` (line length 120, Python 3.11).
3. **Format** — `ruff format --check` on all Python source files *except*
   `weather_data.py` (deferred to a dedicated cleanup PR).
4. **JavaScript syntax** — `node --check static/main.js`.
5. **Tests** — full unittest suite via `PYTHONPATH=. python -m unittest discover`.

**Agents must ensure all five checks pass before considering a backend task done.**
Running `ruff check` and `ruff format --check` locally (see commands above) is the
fastest way to verify this.

## Editing Guidance

- Make focused, minimal changes.
- Preserve query parameter compatibility.
- Keep logging informative but avoid noisy temporary debug output in final state.
- Update `README.md` whenever user-visible behavior changes.
