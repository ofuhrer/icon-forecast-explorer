#!/usr/bin/env python3
from __future__ import annotations

from datetime import timedelta
import json

from meteodatalab import ogd_api

from weather_data import ForecastStore


def _to_iso(init: str) -> str:
    return f"{init[0:4]}-{init[4:6]}-{init[6:8]}T{init[8:10]}:00:00Z"


def _asset_count(collection: str, variable: str, reference_iso: str, lead: int, perturbed: bool = False) -> tuple[int, str]:
    req = ogd_api.Request(
        collection=collection,
        variable=variable,
        reference_datetime=reference_iso,
        perturbed=perturbed,
        horizon=timedelta(hours=int(lead)),
    )
    try:
        urls = ogd_api.get_asset_urls(req)
        return len(urls), ""
    except Exception as exc:  # pragma: no cover - diagnostics script
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    store = ForecastStore()
    store.refresh_catalog(force=True, blocking=True)

    rows = []
    for ds in store.dataset_metas:
        init = store.latest_complete_init(ds.dataset_id)
        if not init:
            rows.append({"dataset": ds.dataset_id, "status": "no-init"})
            continue
        leads = store.lead_hours_for_init(ds.dataset_id, init)
        lead = 0 if 0 in leads else (min(leads) if leads else 0)
        reference_iso = _to_iso(init)

        for var in store.variable_metas:
            if var.ogd_components:
                details = []
                ok = True
                for comp in var.ogd_components:
                    n, err = _asset_count(ds.ogd_collection, comp, reference_iso, lead, perturbed=False)
                    details.append({"component": comp, "assets": n, "error": err})
                    ok = ok and n > 0
                rows.append(
                    {
                        "dataset": ds.dataset_id,
                        "init": init,
                        "lead": lead,
                        "variable_id": var.variable_id,
                        "mapped_to": list(var.ogd_components),
                        "status": "ok" if ok else "missing",
                        "detail": details,
                    }
                )
                continue

            mapped = str(var.ogd_variable or "")
            candidates = store._ogd_variable_candidates(mapped) if mapped else []
            hits = []
            for candidate in candidates:
                n, err = _asset_count(ds.ogd_collection, candidate, reference_iso, lead, perturbed=False)
                if n > 0:
                    hits.append({"candidate": candidate, "assets": n})
                elif err:
                    hits.append({"candidate": candidate, "assets": 0, "error": err})

            matched = next((entry for entry in hits if entry.get("assets", 0) > 0), None)
            rows.append(
                {
                    "dataset": ds.dataset_id,
                    "init": init,
                    "lead": lead,
                    "variable_id": var.variable_id,
                    "mapped_to": mapped,
                    "status": "ok" if matched else "missing",
                    "resolved_candidate": matched["candidate"] if matched else None,
                    "candidates": hits,
                }
            )

    missing = [r for r in rows if r.get("status") == "missing"]
    print(f"total={len(rows)} missing={len(missing)}")
    for row in missing:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
