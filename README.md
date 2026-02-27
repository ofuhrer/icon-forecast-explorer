# ICON Forecast Explorer

Interactive web app for visualizing MeteoSwiss' ICON forecast.

![ICON Forecast Explorer screenshot](docs/screenshot.png)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Runtime Configuration

- `LOG_LEVEL` (`INFO` default, use `DEBUG` for troubleshooting)
- `ICON_LOG_FILE` (`logs/icon_forecast.log` default)
- `CORS_ALLOW_ORIGINS` (comma-separated allowlist)
- `ICON_EXPLORER_ALLOW_ALL_CORS=1` (allow all origins)
- `HOT_PREWARM_ENABLED` (`1` default)
- `HOT_PREWARM_VARIABLES` (`t_2m,tot_prec` default)
- `BACKGROUND_FETCH_WORKERS` (`2` default)

## Time Operators

Time operators are defined centrally in `weather_data.py` and exposed via `/api/metadata`.

- `None`
- `Avg 3h`, `Avg 6h`, `Avg 12h`, `Avg 24h`
- `Acc 3h`, `Acc 6h`, `Acc 12h`, `Acc 24h`

Semantics:
- operators use a trailing window ending at selected lead
- if early leads are missing in that window, available leads are used
- UI lead selector is quantized by operator period using displayed lead time

## Cache Behavior

- Field and vector caches are written as compressed `.npz`
- Cache keys include `model/type/variable/time_operator/init/lead`
- Corrupt partial cache files are detected and dropped automatically

## Debugging Checklist

1. Start with debug logs:
```bash
LOG_LEVEL=DEBUG uvicorn app:app --reload
```
2. Check browser network calls:
- `/api/metadata`
- `/api/tiles/...`
- `/api/value`
3. If UI appears stale, hard refresh (`Cmd+Shift+R`) to bypass static asset cache.

## Core Endpoints

- `GET /api/metadata`
- `GET /api/tiles/{dataset_id}/{variable_id}/{init}/{lead}/{z}/{x}/{y}.png?type_id=...`
- `GET /api/value?...`
- `GET /api/series?...`
