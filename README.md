# ICON Forecast Explorer (v1)

Interactive weather map prototype for Switzerland with:
- grayscale swisstopo base map
- zoom/pan map interaction
- model/type/variable/init/lead controls
- lead-time animation (play/pause with speed selection)
- lead display and separate valid-time display in Swiss local time (`Europe/Zurich`)
- mouse-over value readout (regridded values)

## Implemented data pipeline (ICON-CH1/CH2-EPS)

The backend now includes a real ingestion/caching pipeline in [weather_data.py](/Users/fuhrer/Desktop/geospatial/weather_data.py):
- discovers available init/lead combinations from the STAC API
- requests control (`perturbed=false`) and ensemble members (`perturbed=true`) with `meteodata-lab` `ogd_api`
- regrids downloaded ICON data to a regular display grid over Switzerland
- caches regridded fields in memory and on disk

Cache layout:
- catalog cache: `cache/catalogs/*.json`
- field cache: `cache/fields/*.npz`

## Current scope
- models:
  - `ICON-CH1-EPS` (`icon-ch1-eps-control`)
  - `ICON-CH2-EPS` (`icon-ch2-eps-control`)
- variables:
  - `t_2m` (2 m temperature)
  - `td_2m` (2 m dew point)
  - `wind_speed_10m` (derived from `U_10M` and `V_10M`)
  - `vmax_10m` (10 m wind gust)
  - `tot_prec` (total precipitation)
  - `snow` (snowfall amount)
  - `clct` (total cloud cover)
  - `clcl` (low cloud cover)
  - `ceiling` (cloud ceiling)
  - `hzerocl` (freezing level)
  - `snowlmt` (snowfall limit)
  - `dursun` (sunshine duration)
  - `w_snow` (snow water equivalent)

The API is designed to add additional variables and products later.

Forecast types currently available:
- `control`
- `mean`
- `median`
- `p10` (10th percentile)
- `p90` (90th percentile)

## Colormaps

Required colormaps are stored inside the project in a modern JSON format:
- `colormaps/manifest.json`

The backend reads this manifest at startup and maps variables to colormaps by `variables` entries.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

- `GET /api/metadata`
- `GET /api/tiles/{dataset_id}/{variable_id}/{init}/{lead}/{z}/{x}/{y}.png?type_id=control|mean|median|p10|p90`
- `GET /api/value?dataset_id=...&type_id=control|mean|median|p10|p90&variable_id=...&init=...&lead=...&lat=...&lon=...`
- `GET /api/prefetch?dataset_id=...&type_id=...&variable_id=...&init=...&lead=...`

Metadata dataset entries include:
- `init_to_leads`: currently discovered available lead times per run
- `expected_init_to_leads`: expected operational lead-time coverage per run (used for complete/incomplete labeling)

## Notes

- First request for a new `(dataset, variable, init, lead)` may be slower due to download + regridding.
- Repeated requests are served from memory/disk cache.
- Startup launches a background prewarm loop for hot combinations (latest complete run; common variables/types/leads).
- Cache cleanup policy:
  - field cache files for older cache versions are removed automatically
  - field cache files for runs no longer present in the current catalog are removed automatically
  - field cache files older than 30 hours are removed automatically
  - in-memory field cache is pruned with the same policy
- If STAC discovery is temporarily unavailable, backend falls back to a recent-cycle schedule until discovery works again.
- ecCodes definitions are expected at `.venv/share/eccodes-cosmo-resources/definitions` (same pattern as MeteoSwiss demo notebooks).
