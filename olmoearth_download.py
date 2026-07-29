#!/usr/bin/env python3
"""
Download OlmoEarth prediction outputs (e.g. embedding COGs) to local disk.

Workflow:
  1. POST /api/v1/predictions/search        -> find your prediction(s)
  2. POST /api/v1/prediction-results/search -> get signed download URLs
  3. GET  <signed url>                      -> stream to disk (resumable)

Auth: https://docs.olmoearth.allenai.org/authentication
Set your key once:  export OLMOEARTH_API_KEY=...

Usage:
  # 1. see what predictions exist
  python olmoearth_download.py list --project-id <PROJECT_UUID>

  # 2. inspect what files a prediction has (no download)
  python olmoearth_download.py files --prediction-id <PREDICTION_UUID>

  # 3. download them (--match takes a subset, so you can do it in batches)
  python olmoearth_download.py get --prediction-id <PREDICTION_UUID> --out ./embeddings
  python olmoearth_download.py get --prediction-id <ID> --match '_2024_0[1-3]_' --out ./q1

  # 3b. or don't download the whole COG at all -- pull just an AOI window
  #     (and optionally just some embedding dimensions) straight off the server
  python olmoearth_download.py crop --prediction-id <ID> \
      --bbox 85.30 27.65 85.40 27.75 --bands 1-64 --out ./aoi
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

BASE = "https://olmoearth.allenai.org/api/v1"
CHUNK = 8 * 1024 * 1024  # 8 MiB


# --------------------------------------------------------------------------
# API plumbing
# --------------------------------------------------------------------------
def make_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )
    return s


def post(session, path, payload, verbose=False):
    url = f"{BASE}{path}"
    if verbose:
        print(f"POST {url}\n{json.dumps(payload, indent=2)}", file=sys.stderr)
    r = session.post(url, json=payload, timeout=180)
    if not r.ok:
        # Print the server's complaint -- it usually names the bad field.
        print(f"HTTP {r.status_code} from {url}", file=sys.stderr)
        print(r.text[:4000], file=sys.stderr)
        r.raise_for_status()
    return r.json()


def records_of(payload):
    """OlmoEarth search endpoints wrap results; be liberal about the key name."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("records", "results", "items", "data", "predictions"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    return []


def search_all(session, path, body, verbose=False, max_pages=100):
    """Follow simple cursor/page-token pagination if the API returns one."""
    out, body, pages = [], dict(body), 0
    while pages < max_pages:
        payload = post(session, path, body, verbose)
        out.extend(records_of(payload))
        pages += 1
        token = None
        if isinstance(payload, dict):
            for key in ("next_page_token", "page_token", "next_cursor", "cursor"):
                if payload.get(key):
                    token, token_key = payload[key], key
                    break
        if not token:
            break
        body[token_key] = token
    return out


# --------------------------------------------------------------------------
# Pulling signed URLs out of the response
# --------------------------------------------------------------------------
URL_RE = re.compile(r"^https?://")


def harvest_files(obj, acc=None, name_hint=None):
    """
    Walk the prediction-results JSON and collect (filename, url) pairs.
    Robust to the exact nesting of `output_files`, which is why we recurse
    instead of hard-coding a path.
    """
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        url = None
        for k in ("url", "signed_url", "download_url", "href", "presigned_url"):
            v = obj.get(k)
            if isinstance(v, str) and URL_RE.match(v):
                url = v
                break
        if url:
            fname = None
            for k in ("name", "filename", "file_name", "path", "key"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    fname = Path(v).name
                    break
            acc.append((fname or filename_from_url(url), url, obj.get("size") or obj.get("size_bytes")))
        for v in obj.values():
            harvest_files(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            harvest_files(v, acc)
    elif isinstance(obj, str) and URL_RE.match(obj) and looks_like_a_file(obj):
        acc.append((filename_from_url(obj), obj, None))
    return acc


def looks_like_a_file(url):
    path = urlparse(url).path
    return bool(Path(path).suffix)


def filename_from_url(url):
    name = Path(unquote(urlparse(url).path)).name
    return name or "download.bin"


def dedupe(files):
    seen, out = set(), []
    for name, url, size in files:
        key = (name, urlparse(url).path)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, url, size))
    return out


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
def human(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download(url, dest: Path, retries=5):
    """Stream to disk with HTTP Range resume. Safe to re-run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            # NOTE: no Authorization header -- the URL is already signed, and
            # sending both can make S3/GCS reject the request.
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if r.status_code == 416:  # already complete
                    break
                if have and r.status_code != 206:
                    have, mode = 0, "wb"  # server ignored Range; start over
                else:
                    mode = "ab" if have else "wb"
                r.raise_for_status()

                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                done, t0 = have, time.time()
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        rate = done / max(time.time() - t0, 1e-6)
                        print(
                            f"\r  {dest.name}: {human(done)}"
                            + (f" / {human(total)}" if total else "")
                            + f"  ({human(rate)}/s)   ",
                            end="",
                            flush=True,
                        )
            print()
            break
        except (requests.RequestException, OSError) as e:
            print(f"\n  retry {attempt}/{retries} after error: {e}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))

    part.replace(dest)
    return dest


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_list(session, args):
    body = {}
    if args.project_id:
        body["project_id"] = args.project_id
    if args.model_id:
        body["model_id"] = args.model_id
    recs = search_all(session, "/predictions/search", body, args.verbose)
    if not recs:
        print("No predictions returned. Try `--verbose`, or drop the filters.")
        return
    for r in recs:
        print(
            f"{r.get('id')}  status={r.get('status', '?')}  "
            f"created={r.get('created_time', r.get('created_at', '?'))}  "
            f"name={r.get('name', '')}"
        )
    if args.raw:
        print(json.dumps(recs, indent=2))


def fetch_result_files(session, args):
    body = {"include": ["output_files"]}
    if args.prediction_id:
        body["prediction_ids"] = args.prediction_id
    if args.project_id:
        body["project_id"] = args.project_id
    payload = post(session, "/prediction-results/search", body, args.verbose)
    if args.raw:
        print(json.dumps(payload, indent=2))
    files = dedupe(harvest_files(payload))
    if getattr(args, "match", None):
        pat = re.compile(args.match)
        files = [f for f in files if pat.search(f[0])]
    return files, payload


def cmd_files(session, args):
    files, payload = fetch_result_files(session, args)
    if not files:
        print("No file URLs found. Re-run with --raw to inspect the response shape.")
        print(json.dumps(payload, indent=2)[:3000])
        return
    total = sum(s for _, _, s in files if isinstance(s, int))
    for name, _, size in files:
        print(f"{human(size):>10}  {name}")
    print(f"\n{len(files)} file(s), {human(total) if total else 'unknown'} total")


def cmd_get(session, args):
    files, payload = fetch_result_files(session, args)
    if not files:
        print("No file URLs found. Re-run with --raw to inspect the response shape.")
        print(json.dumps(payload, indent=2)[:3000])
        return
    out = Path(args.out)
    print(f"{len(files)} file(s) -> {out.resolve()}")
    for i, (name, url, size) in enumerate(files, 1):
        dest = out / name
        if dest.exists() and (not isinstance(size, int) or dest.stat().st_size == size):
            print(f"[{i}/{len(files)}] {name}: already present, skipping")
            continue
        print(f"[{i}/{len(files)}] {name} ({human(size)})")
        download(url, dest)
    print("Done.")


def parse_bands(spec):
    """'1-64,100,200-204' -> [1..64, 100, 200..204]"""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_crop(session, args):
    """Windowed read straight off the signed URL -- transfers only the tiles
    (and bands) that intersect the AOI, instead of the whole COG."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError:
        sys.exit("crop needs rasterio:  pip install rasterio")

    files, payload = fetch_result_files(session, args)
    if not files:
        print("No file URLs found. Re-run with --raw to inspect the response shape.")
        return

    if args.geojson:
        import json as _json
        gj = _json.loads(Path(args.geojson).read_text())
        geoms = [f["geometry"] for f in gj["features"]] if gj.get("type") == "FeatureCollection" else [gj.get("geometry", gj)]
        xs, ys = [], []
        def walk(c):
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for sub in c:
                    walk(sub)
        for g in geoms:
            walk(g["coordinates"])
        bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        bbox = tuple(args.bbox)

    bands = parse_bands(args.bands)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: do not set CPL_VSIL_CURL_ALLOWED_EXTENSIONS -- signed URLs carry a
    # query string, so the extension sniff fails and GDAL refuses to open them.
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="536870912",
    )

    with env:
        for i, (name, url, _) in enumerate(files, 1):
            dest = out / f"crop_{name}"
            try:
                with rasterio.open(f"/vsicurl/{url}") as src:
                    left, bottom, right, top = transform_bounds(
                        args.bbox_crs, src.crs, *bbox, densify_pts=21
                    )
                    win = from_bounds(left, bottom, right, top, src.transform)
                    win = win.round_offsets().round_lengths()
                    win = win.intersection(
                        rasterio.windows.Window(0, 0, src.width, src.height)
                    )
                    if win.width <= 0 or win.height <= 0:
                        print(f"[{i}/{len(files)}] {name}: no overlap with AOI, skipping")
                        continue
                    idx = bands or list(range(1, src.count + 1))
                    data = src.read(idx, window=win)
                    prof = src.profile.copy()
                    prof.update(
                        driver="GTiff",
                        height=data.shape[1],
                        width=data.shape[2],
                        count=len(idx),
                        transform=src.window_transform(win),
                        tiled=True,
                        blockxsize=512,
                        blockysize=512,
                        compress="deflate",
                    )
                    with rasterio.open(dest, "w", **prof) as dst:
                        dst.write(data)
                print(
                    f"[{i}/{len(files)}] {dest.name}: "
                    f"{data.shape[2]}x{data.shape[1]} px, {len(idx)} band(s), "
                    f"{human(dest.stat().st_size)}"
                )
            except Exception as e:
                print(f"[{i}/{len(files)}] {name}: FAILED ({e})", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", default=os.environ.get("OLMOEARTH_API_KEY"))
    p.add_argument("--verbose", action="store_true", help="echo requests")
    p.add_argument("--raw", action="store_true", help="dump raw JSON responses")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list predictions")
    sp.add_argument("--project-id")
    sp.add_argument("--model-id")
    sp.set_defaults(func=cmd_list)

    for name, fn in (("files", cmd_files), ("get", cmd_get), ("crop", cmd_crop)):
        sp = sub.add_parser(name, help=f"{name} for a prediction")
        sp.add_argument("--prediction-id", nargs="+", help="one or more prediction UUIDs")
        sp.add_argument("--project-id")
        sp.add_argument("--match", help="regex; only act on filenames that match")
        if name in ("get", "crop"):
            sp.add_argument("--out", default="./olmoearth_results")
        if name == "crop":
            sp.add_argument("--bbox", nargs=4, type=float,
                            metavar=("MINX", "MINY", "MAXX", "MAXY"))
            sp.add_argument("--bbox-crs", default="EPSG:4326")
            sp.add_argument("--geojson", help="AOI file; its bounding box is used")
            sp.add_argument("--bands", help="e.g. 1-64,100,200-204 (1-indexed)")
        sp.set_defaults(func=fn)

    args = p.parse_args()
    if not args.api_key:
        sys.exit("Set OLMOEARTH_API_KEY or pass --api-key")
    args.func(make_session(args.api_key), args)


if __name__ == "__main__":
    main()