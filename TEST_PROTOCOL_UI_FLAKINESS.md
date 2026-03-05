# UI Flakiness Debug Protocol

This protocol is for reproducing and diagnosing:
- stale map tiles after control changes
- unnecessary reloads of previously loaded content
- stuck `Loading...` or `Unavailable` status text

## 1) Start Server with Debug Logs

```bash
LOG_LEVEL=DEBUG uvicorn app:app --reload
```

## 2) Open UI with Frontend Trace Enabled

Use:

```text
http://127.0.0.1:8000/?trace_ui=1
```

This enables browser console traces (`[ui-trace] ...`) and keeps backend traces in logs.

## 3) What to Capture

For each failing run, capture:

1. Exact user sequence and timestamps (local time).
2. Browser console lines containing `[ui-trace]`.
3. Backend lines containing:
   - `API trace path=... vt=...`
   - `Tile cache miss; sync-fetch ...`
   - `Tile request runtime error: ...`

The `vt` token is the correlation key between UI actions and backend calls.

## 4) Strict Test Sequences

Run each sequence 5 times.

### Sequence A: Variable Toggle Under Load

1. Load app and wait until map is fully ready.
2. Switch variable quickly:
   - current default -> `10 m wind speed` -> `surface pressure` -> default
3. Do **not** pan/zoom while switching.
4. Check:
   - map image matches selected variable
   - wind vectors only visible for wind variable
   - `Loading...` disappears once ready
   - no stale `Unavailable` message if tiles are present

### Sequence B: Lead Scrub Stress

1. Pick one variable with broad lead coverage (e.g. `vmax_10m`).
2. Drag lead slider rapidly from min to max and back in ~2 seconds.
3. Release slider and wait.
4. Check:
   - final displayed lead corresponds to current slider label
   - map settles without stale previous lead image
   - status text returns to empty (not stuck)

### Sequence C: Model + Variable Switch Combined

1. Start from ICON-CH1-EPS at lead +0.
2. Switch model to ICON-CH2-EPS.
3. Immediately switch variable twice.
4. Wait for ready state.
5. Check:
   - selected model/variable pair is actually displayed
   - previous model imagery is not retained
   - no permanent `Loading...` overlay

## 5) Pass/Fail Criteria

Pass if all are true for all runs:
- correct final map content for current control state
- no stale vectors/layers from previous selection
- no lingering `Loading...` or `Unavailable` after readiness
- no backend errors for matching `vt` except expected transient 503 during fetch

Fail if any run violates above; include `vt` and timestamps in the report.

## 6) Quick Triage Rules

- If backend `vt` does not match current UI `vt`: stale-response race.
- If backend has `status=200` for tiles but UI remains `Loading...`: UI state transition bug.
- If repeated `Tile cache miss` with same `vt` long after first render: bad cache key/invalidation issue.
- If `Unavailable` appears with successful tile loads for same `vt`: field-debug/error precedence bug.
