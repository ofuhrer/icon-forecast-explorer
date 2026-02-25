# ICON Forecast Explorer

Interactive web app for Swiss ICON forecasts (map + meteogram).

![ICON Forecast Explorer screenshot](docs/screenshot.png)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Core Endpoints

- `GET /api/metadata`
- `GET /api/tiles/{dataset_id}/{variable_id}/{init}/{lead}/{z}/{x}/{y}.png?type_id=...`
- `GET /api/value?...`
- `GET /api/series?...`
