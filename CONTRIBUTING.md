# Contributing

Guidelines for contributors and AI coding agents working on this repository.

## File-size Limits

Keeping files focused makes them easier for both humans and agents to reason about:

| Language | Recommended maximum |
|----------|---------------------|
| Python   | ~500 lines          |
| JavaScript | ~1 000 lines      |

If a file exceeds these limits, consider extracting a focused helper module before
adding more code.

## Module Naming Conventions

- Python backend modules use **snake_case**.
- New backend helpers belong in one of the existing focused modules (see *What Goes
  Where* in `AGENTS.md`) or in a clearly named new module that follows the pattern:
  `<concern>.py` (e.g. `tile_rendering.py`, `series_cache.py`).
- Do **not** add business logic directly to `app.py`.  It is the thin endpoint wiring
  layer only.
- Frontend JS modules live under `static/js/` and export named functions.

## Before You Commit

### 1 — Run the tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

All 38 (or more) tests must pass.

### 2 — Lint

```bash
ruff check app.py weather_data.py weather_models.py weather_cache.py weather_grib.py \
           tile_rendering.py grid_utils.py series_cache.py input_validation.py
```

### 3 — Format check

```bash
ruff format --check app.py weather_models.py weather_cache.py weather_grib.py \
                    tile_rendering.py grid_utils.py series_cache.py input_validation.py
```

Apply auto-formatting with `ruff format <file>` if needed.

### 4 — JavaScript syntax

```bash
node --check static/main.js
```

### 5 — Python syntax check (quick sanity)

```bash
python -m py_compile app.py weather_data.py weather_models.py weather_cache.py \
                     weather_grib.py tile_rendering.py grid_utils.py \
                     series_cache.py input_validation.py
```

## Dev Dependencies

Install once:

```bash
pip install -r requirements-dev.txt
```

This installs `ruff` for linting and formatting.

## API Compatibility

Do **not** change the shape of any existing API response.  All endpoints listed in
`AGENTS.md` under *API Stability Requirements* must remain backward-compatible.

## CI

The GitHub Actions workflow runs all five checks above automatically on every push
and pull request.  A failing CI must be fixed before merging.
