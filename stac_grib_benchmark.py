#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

STAC_BASE = "https://data.geo.admin.ch/api/stac/v1"
DATASET_TO_COLLECTION = {
    "icon-ch1-eps-control": "ch.meteoschweiz.ogd-forecasting-icon-ch1",
    "icon-ch2-eps-control": "ch.meteoschweiz.ogd-forecasting-icon-ch2",
}
ITEM_ID_RE = re.compile(r"icon-ch[12]-eps-(\d{10,12})-(\d+)-", re.IGNORECASE)


@dataclass
class AssetRecord:
    item_id: str
    asset_key: str
    href: str
    init: Optional[str]
    lead: Optional[int]


@dataclass
class DownloadResult:
    url: str
    ok: bool
    status_code: int
    bytes_read: int
    seconds: float
    error: str = ""


_thread_local = threading.local()


def _parse_item_id(item_id: str) -> Tuple[Optional[str], Optional[int]]:
    m = ITEM_ID_RE.search(str(item_id or ""))
    if not m:
        return None, None
    init = m.group(1)
    try:
        lead = int(m.group(2))
    except ValueError:
        lead = None
    return init, lead


def _init_matches(init_filter: str, asset_init: Optional[str]) -> bool:
    """
    Match init robustly across 10-digit (YYYYMMDDHH) and 12-digit (YYYYMMDDHHMM)
    forms used in different catalog/item identifiers.
    """
    if not init_filter:
        return True
    if not asset_init:
        return False
    wanted = str(init_filter).strip()
    got = str(asset_init).strip()
    if wanted == got:
        return True
    if len(wanted) == 10 and len(got) == 12 and got.startswith(wanted):
        return True
    if len(wanted) == 12 and len(got) == 10 and wanted.startswith(got):
        return True
    return False


def _iter_collection_items(collection_id: str, timeout_s: float) -> Iterable[Dict]:
    url: Optional[str] = f"{STAC_BASE}/collections/{collection_id}/items?limit=500"
    session = requests.Session()
    while url:
        resp = session.get(url, timeout=(10, timeout_s))
        resp.raise_for_status()
        payload = resp.json()
        for feature in payload.get("features", []):
            yield feature
        next_url = None
        for link in payload.get("links", []):
            if str(link.get("rel", "")).lower() == "next":
                href = str(link.get("href", "")).strip()
                if href:
                    next_url = href
                break
        url = next_url


def _collect_assets(
    collection_id: str,
    timeout_s: float,
    init_filter: Optional[str],
    lead_min: Optional[int],
    lead_max: Optional[int],
    contains: Optional[str],
) -> List[AssetRecord]:
    contains_norm = (contains or "").strip().lower()
    records: List[AssetRecord] = []
    for feature in _iter_collection_items(collection_id, timeout_s):
        item_id = str(feature.get("id", ""))
        init, lead = _parse_item_id(item_id)
        if init_filter and not _init_matches(init_filter, init):
            continue
        if lead_min is not None and lead is not None and lead < lead_min:
            continue
        if lead_max is not None and lead is not None and lead > lead_max:
            continue
        assets = feature.get("assets", {}) or {}
        for asset_key, asset in assets.items():
            href = str((asset or {}).get("href", "")).strip()
            if not href or ".grib2" not in href.lower():
                continue
            if contains_norm:
                hay = f"{item_id} {asset_key} {href}".lower()
                if contains_norm not in hay:
                    continue
            records.append(
                AssetRecord(
                    item_id=item_id,
                    asset_key=str(asset_key),
                    href=href,
                    init=init,
                    lead=lead,
                )
            )
    return records


def _latest_init(records: List[AssetRecord]) -> Optional[str]:
    inits = sorted({r.init for r in records if r.init})
    return inits[-1] if inits else None


def _dedupe_by_path(records: List[AssetRecord]) -> List[AssetRecord]:
    # Signed URLs can differ by query string while pointing to the same file.
    seen: Dict[str, AssetRecord] = {}
    for rec in records:
        path = urlparse(rec.href).path
        if path not in seen:
            seen[path] = rec
    out = list(seen.values())
    out.sort(key=lambda r: (r.init or "", r.lead if r.lead is not None else -1, r.item_id, r.asset_key))
    return out


def _session_for_worker() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def _download_one(
    url: str,
    timeout_s: float,
    chunk_size: int,
    output_dir: Optional[Path],
) -> DownloadResult:
    sess = _session_for_worker()
    bytes_read = 0
    t0 = time.perf_counter()
    target_file = None
    try:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(urlparse(url).path).name
            target_file = output_dir / filename
        with sess.get(url, stream=True, timeout=(10, timeout_s)) as resp:
            status = int(resp.status_code)
            resp.raise_for_status()
            if target_file is None:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        bytes_read += len(chunk)
            else:
                with target_file.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            bytes_read += len(chunk)
                            fh.write(chunk)
        dt = max(1e-6, time.perf_counter() - t0)
        return DownloadResult(url=url, ok=True, status_code=200, bytes_read=bytes_read, seconds=dt)
    except Exception as exc:
        dt = max(1e-6, time.perf_counter() - t0)
        return DownloadResult(url=url, ok=False, status_code=0, bytes_read=bytes_read, seconds=dt, error=str(exc))


def _format_rate(bytes_count: int, seconds: float) -> str:
    bps = (bytes_count / max(1e-9, seconds))
    mbps = bps / (1024 * 1024)
    return f"{mbps:8.2f} MiB/s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark raw STAC GRIB asset download throughput (sequential by default)."
    )
    parser.add_argument(
        "--dataset-id",
        default="icon-ch2-eps-control",
        choices=sorted(DATASET_TO_COLLECTION.keys()),
        help="Dataset identifier mapped to STAC collection.",
    )
    parser.add_argument(
        "--collection-id",
        default="",
        help="Optional explicit STAC collection id. Overrides --dataset-id mapping.",
    )
    parser.add_argument(
        "--init",
        default="",
        help="Forecast init YYYYMMDDHH. If omitted, latest init found in catalog is used.",
    )
    parser.add_argument("--lead-min", type=int, default=None)
    parser.add_argument("--lead-max", type=int, default=None)
    parser.add_argument(
        "--contains",
        default="",
        help="Only include assets where this substring is found in item-id/asset-key/href (e.g. 'tot_prec').",
    )
    parser.add_argument("--max-assets", type=int, default=0, help="Limit number of assets downloaded (0 = all).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel download workers (1 = sequential).")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=180.0,
        help="Read timeout per request in seconds.",
    )
    parser.add_argument("--chunk-kib", type=int, default=1024, help="Streaming chunk size in KiB.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory to save downloaded files. If omitted, data is streamed and discarded.",
    )
    args = parser.parse_args()

    collection_id = args.collection_id.strip() or DATASET_TO_COLLECTION[args.dataset_id]
    init_filter = args.init.strip() or None
    contains = args.contains.strip() or None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir.strip() else None

    print(f"Collection: {collection_id}")
    print("Discovering STAC assets...")
    records = _collect_assets(
        collection_id=collection_id,
        timeout_s=float(args.timeout_s),
        init_filter=init_filter,
        lead_min=args.lead_min,
        lead_max=args.lead_max,
        contains=contains,
    )
    if not records:
        print("No matching GRIB assets found.")
        return 2

    if init_filter is None:
        latest = _latest_init(records)
        if latest:
            records = [r for r in records if r.init == latest]
            print(f"Auto-selected latest init: {latest} ({len(records)} raw assets before de-dup)")

    records = _dedupe_by_path(records)
    if args.max_assets and args.max_assets > 0:
        records = records[: int(args.max_assets)]

    if not records:
        print("No assets left after de-dup / max-assets filter.")
        return 2

    print(f"Assets to download: {len(records)}")
    if output_dir is None:
        print("Mode: stream+discard (network benchmark, minimal disk impact)")
    else:
        print(f"Mode: save files to {output_dir}")
    print(f"Workers: {max(1, int(args.workers))}")

    t_start = time.perf_counter()
    total_bytes = 0
    ok_count = 0
    fail_count = 0
    chunk_size = max(1, int(args.chunk_kib)) * 1024

    def submit_one(executor, rec: AssetRecord):
        return executor.submit(
            _download_one,
            rec.href,
            float(args.timeout_s),
            chunk_size,
            output_dir,
        )

    if max(1, int(args.workers)) == 1:
        for idx, rec in enumerate(records, start=1):
            result = _download_one(rec.href, float(args.timeout_s), chunk_size, output_dir)
            total_bytes += result.bytes_read
            if result.ok:
                ok_count += 1
            else:
                fail_count += 1
            elapsed_total = max(1e-6, time.perf_counter() - t_start)
            label = Path(urlparse(rec.href).path).name
            if result.ok:
                print(
                    f"[{idx:4d}/{len(records)}] OK   {label}  "
                    f"{result.bytes_read / (1024*1024):8.2f} MiB  "
                    f"{_format_rate(result.bytes_read, result.seconds)}  "
                    f"total={_format_rate(total_bytes, elapsed_total)}"
                )
            else:
                print(f"[{idx:4d}/{len(records)}] FAIL {label}  err={result.error}")
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers)), thread_name_prefix="stac-bench") as ex:
            fut_map = {submit_one(ex, rec): rec for rec in records}
            done = 0
            for fut in cf.as_completed(fut_map):
                done += 1
                rec = fut_map[fut]
                result = fut.result()
                total_bytes += result.bytes_read
                if result.ok:
                    ok_count += 1
                else:
                    fail_count += 1
                elapsed_total = max(1e-6, time.perf_counter() - t_start)
                label = Path(urlparse(rec.href).path).name
                if result.ok:
                    print(
                        f"[{done:4d}/{len(records)}] OK   {label}  "
                        f"{result.bytes_read / (1024*1024):8.2f} MiB  "
                        f"{_format_rate(result.bytes_read, result.seconds)}  "
                        f"total={_format_rate(total_bytes, elapsed_total)}"
                    )
                else:
                    print(f"[{done:4d}/{len(records)}] FAIL {label}  err={result.error}")

    elapsed = max(1e-6, time.perf_counter() - t_start)
    print("")
    print("Summary")
    print(f"- ok files      : {ok_count}")
    print(f"- failed files  : {fail_count}")
    print(f"- total bytes   : {total_bytes / (1024*1024*1024):.3f} GiB")
    print(f"- total time    : {elapsed:.1f} s")
    print(f"- avg throughput: {_format_rate(total_bytes, elapsed)}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
