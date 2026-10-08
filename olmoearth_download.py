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


def post(session, path, payload, verbose=False, timeout=300, raise_on_error=True):
    url = f"{BASE}{path}"
    if verbose:
        print(f"POST {url}\n{json.dumps(payload, indent=2)}", file=sys.stderr)
    r = session.post(url, json=payload, timeout=timeout)
    if not r.ok:
        # Print the server's complaint -- it usually names the bad field.
        print(f"HTTP {r.status_code} from {url}", file=sys.stderr)
        print(r.text[:4000], file=sys.stderr)
        if raise_on_error:
            r.raise_for_status()
        return None
    return r.json()


def get(session, path, params=None, verbose=False, timeout=300, raise_on_error=True):
    url = f"{BASE}{path}"
    if verbose:
        print(f"GET {url} {params or ''}", file=sys.stderr)
    r = session.get(url, params=params, timeout=timeout)
    if not r.ok:
        print(f"HTTP {r.status_code} from {url}", file=sys.stderr)
        print(r.text[:4000], file=sys.stderr)
        if raise_on_error:
            r.raise_for_status()
        return None
    return r.json()


def kw(value):
    """Wrap a value as a KeywordFilter.

    Every *_id field on the search endpoints is a KeywordFilter object
    ({eq, neq, inc, ninc, exists}), NOT a bare id. Passing the raw uuid gets:
      422 ... Input should be a valid dictionary or object [location=body.project_id]
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        vals = list(value)
        return {"eq": vals[0]} if len(vals) == 1 else {"inc": vals}
    return {"eq": value}


def resolve_result_ids(session, args):
    """prediction id(s) / project id -> prediction_result id(s).

    /prediction-results/search filters on the RESULT id; it has no prediction_ids
    field. The join is PredictionRead.result.id, which only GET /predictions/{id}
    returns, so resolve it here rather than guessing a filter that does not exist.
    """
    rids, pred_ids = [], list(getattr(args, "prediction_id", None) or [])

    if getattr(args, "project_id", None) and not pred_ids:
        body = {"project_id": kw(args.project_id), "limit": 500}
        for rec in records_of(post(session, "/predictions/search", body,
                                   args.verbose, args.timeout)):
            # The search records are PredictionRead too, so the result may already
            # be here -- no need for a per-prediction GET when it is.
            res = rec.get("result") or {}
            if res.get("id"):
                rids.append(res["id"])
            elif rec.get("id"):
                pred_ids.append(rec["id"])
        print(f"  project: {len(rids)} result(s) inline, {len(pred_ids)} to look up")

    for pid in pred_ids:
        payload = get(session, f"/predictions/{pid}", verbose=args.verbose,
                      timeout=getattr(args, "timeout", 300), raise_on_error=False)
        if not payload:
            continue
        # Every endpoint wraps in ApiOutputModel: {"records": [...], "meta", "errors"}.
        recs = records_of(payload)
        rec = recs[0] if recs else payload
        res = rec.get("result") or {}
        if res.get("id"):
            rids.append(res["id"])
        else:
            print(f"  prediction {pid}: no result id "
                  f"(status={rec.get('status')}) -- re-run with --raw to inspect",
                  file=sys.stderr)
    return rids


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


def search_all(session, path, body, verbose=False, max_pages=100, timeout=300):
    """Follow simple cursor/page-token pagination if the API returns one."""
    out, body, pages = [], dict(body), 0
    while pages < max_pages:
        payload = post(session, path, body, verbose, timeout)
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
        # file_path is what PredictionResultFileResponse actually carries.
        for k in ("url", "signed_url", "download_url", "href", "presigned_url",
                  "file_path"):
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
            # PredictionResultFileResponse carries the size under
            # properties.filesize_bytes, not a top-level "size".
            size = obj.get("size") or obj.get("size_bytes")
            if size is None and isinstance(obj.get("properties"), dict):
                size = obj["properties"].get("filesize_bytes")
            acc.append((fname or filename_from_url(url), url, size))
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
        body["project_id"] = kw(args.project_id)
    if args.model_id:
        body["model_id"] = kw(args.model_id)
    if getattr(args, "extra", None):
        body.update(json.loads(args.extra))
    recs = search_all(session, "/predictions/search", body, args.verbose,
                      timeout=args.timeout)
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
    result_ids = list(getattr(args, "result_id", None) or [])
    if not result_ids:
        result_ids = resolve_result_ids(session, args)
    if result_ids:
        body["id"] = kw(result_ids)
    else:
        print("  no prediction_result ids resolved -- searching unfiltered",
              file=sys.stderr)
    if getattr(args, "extra", None):
        body.update(json.loads(args.extra))
    payload = post(session, "/prediction-results/search", body, args.verbose,
                   getattr(args, "timeout", 300))
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


NAME_FIELD_CANDIDATES = (
    "healpix", "healpix_id", "hp_id", "hpx", "ipix", "pix", "pixel",
    "tile", "tile_id", "tileid", "cell", "cell_id", "name", "id",
)


def guess_name_field(columns):
    lower = {c.lower(): c for c in columns}
    for cand in NAME_FIELD_CANDIDATES:
        if cand in lower:
            return lower[cand]
    return None


def sanitize(value):
    s = str(value).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s) or "unnamed"


def bbox_overlaps(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


class Sources:
    """Open remote COGs, grouped by CRS, with signed-URL re-minting."""

    def __init__(self, session, args, rasterio):
        self.session, self.args, self.rasterio = session, args, rasterio
        self.datasets, self.groups, self.opened_at = [], {}, 0.0

    def open(self):
        from collections import defaultdict

        self.close()
        files, _ = fetch_result_files(self.session, self.args)
        self.datasets = []
        for name, url, _ in files:
            try:
                self.datasets.append(self.rasterio.open(f"/vsicurl/{url}"))
            except Exception as e:
                print(f"  cannot open {name}: {e}", file=sys.stderr)
        groups = defaultdict(list)
        for ds in self.datasets:
            groups[str(ds.crs)].append(ds)
        self.groups = dict(groups)
        self.opened_at = time.time()
        print(
            f"  {len(self.datasets)} source COG(s) across "
            f"{len(self.groups)} CRS: {', '.join(self.groups)}"
        )

    def maybe_refresh(self, minutes, retries=5):
        """Re-mint signed URLs, retrying the API call.

        This runs mid-download after tens of minutes of heavy traffic, which is
        exactly when the server is most likely to drop a connection. An unprotected
        request here kills a multi-hour run at, say, tile 136 of 217 -- so retry with
        backoff rather than letting a RemoteDisconnected propagate.
        """
        if not (minutes and time.time() - self.opened_at > minutes * 60):
            return False
        print("  signed URLs going stale, re-minting...")
        for attempt in range(1, retries + 1):
            try:
                self.open()
                return True
            except Exception as e:
                if attempt == retries:
                    print(f"  re-mint failed after {retries} attempts: {e}",
                          file=sys.stderr)
                    raise
                wait = min(60, 2 ** attempt)
                print(f"  re-mint attempt {attempt}/{retries} failed ({type(e).__name__}"
                      f"); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        return False

    def close(self):
        for ds in self.datasets:
            try:
                ds.close()
            except Exception:
                pass
        self.datasets = []


def find_gedi(gedi_dir: Path, tile_value):
    """healpix_<id>_gedi{.tif,.tiff,} -- same extension order as load_gedi()."""
    tid = sanitize(tile_value)
    for ext in (".tif", ".tiff", ""):
        p = gedi_dir / f"healpix_{tid}_gedi{ext}"
        if p.exists():
            return p
    return None


def tile_grid(hits, bounds, res=None, snap=True):
    """Target grid for one tile: (transform, width, height, (rx, ry)).

    Derived from the source resolution rather than from a merge, so the output
    extent is known BEFORE any pixels are read -- that is what lets the tile be
    written window by window.

    snap=True aligns the output origin to the SOURCE pixel grid. Without it the
    origin is the polygon's own corner, which sits at an arbitrary phase relative to
    the source pixels; merge() then resamples (nearest by default) and the whole tile
    is displaced by up to half a pixel -- 20 m at 40 m resolution. That is a rigid
    shift, invisible in any self-consistency check, and it biases every sample the
    same way. Snapping makes each output pixel land exactly on a source pixel, so the
    read is a copy and no resampling happens at all.

    The tile is expanded outward to the enclosing grid lines, never cropped, so the
    polygon stays fully covered.
    """
    import math
    from rasterio.transform import from_origin

    if res is None:
        res = hits[0].res
    rx, ry = abs(res[0]), abs(res[1])
    minx, miny, maxx, maxy = bounds

    if snap:
        ox, oy = hits[0].transform.c, hits[0].transform.f
        # All sources should share one grid; if they do not, snapping to the first is
        # still better than not snapping, but the mismatch is worth knowing about.
        phases = {(round(((h.transform.c - ox) / rx) % 1.0, 6),
                   round(((h.transform.f - oy) / ry) % 1.0, 6)) for h in hits}
        if len(phases) > 1:
            print(f"      NOTE: {len(phases)} different pixel-grid phases among "
                  f"{len(hits)} sources; snapping to the first", file=sys.stderr)
        minx = ox + math.floor((minx - ox) / rx) * rx
        maxy = oy - math.floor((oy - maxy) / ry) * ry
        # -1e-9 absorbs float noise so an exactly-aligned edge does not add a column.
        width = max(1, math.ceil((maxx - minx) / rx - 1e-9))
        height = max(1, math.ceil((maxy - miny) / ry - 1e-9))
    else:
        width = max(1, int(round((maxx - minx) / rx)))
        height = max(1, int(round((maxy - miny) / ry)))

    return from_origin(minx, maxy, rx, ry), width, height, (rx, ry)


def iter_windows(width, height, block):
    """Yield (row_off, col_off, w, h) blocks covering the grid."""
    for row in range(0, height, block):
        h = min(block, height - row)
        for col in range(0, width, block):
            yield row, col, min(block, width - col), h


def write_tile_chunked(rasterio, rio_merge, hits, bounds, dest, bands, nodata,
                       block=1024, retries=3, geom=None, clip=False, tags=None,
                       snap=True):
    """Write one tile to `dest` in `block`-sized windows.

    The original code called merge() once over the whole tile, which allocates the
    entire (bands, H, W) array in RAM and pulls every byte in a single long-lived
    transfer -- that is why whole-tile downloads kept failing. Here each window is an
    independent short read that is retried on its own and flushed to disk immediately,
    so peak memory is one window and a transient error costs one window, not the tile.

    Returns (width, height, count) or None if nothing overlapped.
    """
    from rasterio.windows import Window
    from rasterio.transform import array_bounds

    transform, width, height, res = tile_grid(hits, bounds, snap=snap)
    count = len(bands) if bands else hits[0].count

    prof = hits[0].profile.copy()
    prof.update(driver="GTiff", height=height, width=width, count=count,
                transform=transform, crs=hits[0].crs, nodata=nodata,
                tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", predictor=2)

    do_clip = clip and geom is not None and nodata is not None
    if do_clip:
        from rasterio.features import geometry_mask
        geom_json = geom.__geo_interface__

    wrote_any = False
    with rasterio.open(dest, "w", **prof) as dst:
        for row, col, w, h in iter_windows(width, height, block):
            win = Window(col, row, w, h)
            wb = array_bounds(h, w, rasterio.windows.transform(win, transform))
            # array_bounds returns (minx, miny, maxx, maxy) for this window.
            for attempt in range(1, retries + 1):
                try:
                    arr, _ = rio_merge(hits, bounds=wb, res=res,
                                       nodata=nodata, indexes=bands)
                    break
                except Exception:
                    if attempt == retries:
                        raise
                    time.sleep(2 ** attempt)
            if arr.size == 0:
                continue
            # merge() can be off by a pixel from floating-point bounds maths.
            arr = arr[:, :h, :w]
            if arr.shape[1] != h or arr.shape[2] != w:
                import numpy as np
                pad = np.full((arr.shape[0], h, w),
                              nodata if nodata is not None else 0, dtype=arr.dtype)
                pad[:, :arr.shape[1], :arr.shape[2]] = arr
                arr = pad
            if do_clip:
                # Masked here rather than in a second pass: a dataset opened "w"
                # cannot be read back, so there is nothing to post-process.
                outside = geometry_mask(
                    [geom_json], out_shape=(h, w),
                    transform=rasterio.windows.transform(win, transform),
                    invert=False)
                if outside.any():
                    arr[:, outside] = nodata

            dst.write(arr, window=win)
            wrote_any = True

        if tags:
            dst.update_tags(**tags)

    return (width, height, count) if wrote_any else None


def dense_npz(rasterio, tif_path, dest):
    """Whole grid + georeferencing, compressed. Same order of magnitude as the tif."""
    import numpy as np
    with rasterio.open(tif_path) as src:
        arr = src.read()
        np.savez_compressed(dest, emb=arr,
                            transform=np.array(src.transform.to_gdal(), dtype="float64"),
                            crs=str(src.crs),
                            bounds=np.array(src.bounds, dtype="float64"),
                            nodata=np.array([src.nodata if src.nodata is not None else np.nan]))
    return dest.stat().st_size


def sparse_npz(rasterio, tif_paths, gedi_path, dest, block_rows=512):
    """(X, y) at valid GEDI pixels -- the presto/tessera/prithvi convention.

    Mask and target match preprocess_olmo_to_gedi_tiles.process_tile(): GEDI != nodata,
    > 0, < 12000 (values are cm), y in metres.

    Sampling does NOT match it, deliberately. That function scales indices
    proportionally (rows * emb_h / gedi_h), which assumes the two grids are related
    by a per-axis linear stretch. They are not: GEDI here is EPSG:4326 (square in
    degrees) while the embeddings are UTM (square in metres), and lon/lat -> UTM is a
    projection, not a stretch. Over one healpix tile at ~44 N the meridian convergence
    alone shifts the tile's west edge by ~560 m between its south and north sides --
    ~14 pixels at 40 m, worst at the corners -- with y still perfectly correct, so the
    damage is invisible in any check that only looks at y.

    So: take each valid GEDI pixel's CENTRE through its own affine, reproject into the
    embedding CRS, and look up the pixel there. Correct for any combination of
    resolution, extent and CRS.

    Reads the embedding raster in row blocks rather than whole, so a multi-GB tile
    does not have to fit in RAM just to pull a few thousand samples out of it.
    """
    import numpy as np
    from rasterio.windows import Window
    from rasterio.transform import xy as transform_xy, rowcol
    from rasterio.warp import transform as warp_transform

    with rasterio.open(gedi_path) as g:
        gedi = g.read(1)
        gnodata = g.nodata if g.nodata is not None else -9999
        gtransform, gcrs = g.transform, g.crs

    valid = (gedi != gnodata) & (gedi > 0) & (gedi < 12000)
    rows, cols = np.where(valid)
    if len(rows) == 0:
        return None, 0

    y_all = (gedi[rows, cols] / 100.0).astype(np.float32)   # cm -> m

    if not isinstance(tif_paths, (list, tuple)):
        tif_paths = [tif_paths]

    # A tile straddling a UTM zone boundary is cut once per CRS, and each cut only
    # holds data where that zone's COGs reach; the rest is nodata. So sample EVERY
    # variant and keep, per GEDI pixel, the first one that yields real data. Writing
    # one file per CRS instead would either collide on the filename or split a tile's
    # samples across two files that a training run would have to reconcile.
    X = None
    filled = np.zeros(len(y_all), dtype=bool)

    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            emb_h, emb_w, count = src.height, src.width, src.count
            src_nodata = src.nodata
            if X is None:
                X = np.zeros((len(y_all), count), dtype=np.float32)
            elif X.shape[1] != count:
                print(f"      band count differs between CRS variants "
                      f"({X.shape[1]} vs {count}); skipping {Path(tif_path).name}")
                continue

            # GEDI pixel centres -> embedding CRS -> embedding row/col.
            xs, ys = transform_xy(gtransform, rows.tolist(), cols.tolist())
            if gcrs and src.crs and gcrs != src.crs:
                xs, ys = warp_transform(gcrs, src.crs, list(xs), list(ys))
            erows, ecols = rowcol(src.transform, xs, ys)
            erows = np.asarray(erows, dtype=np.int64)
            ecols = np.asarray(ecols, dtype=np.int64)

            # Candidates: inside this cut's grid and not already filled by an
            # earlier variant. Points outside are dropped rather than clipped --
            # clipping would silently pile samples onto the border row.
            cand = ((erows >= 0) & (erows < emb_h) &
                    (ecols >= 0) & (ecols < emb_w) & ~filled)
            if not cand.any():
                continue

            idx = np.where(cand)[0]
            order = idx[np.argsort(erows[idx], kind="stable")]
            for start in range(0, emb_h, block_rows):
                stop = min(start + block_rows, emb_h)
                sel = order[(erows[order] >= start) & (erows[order] < stop)]
                if sel.size == 0:
                    continue
                blk = src.read(window=Window(0, start, emb_w, stop - start))
                vals = blk[:, erows[sel] - start, ecols[sel]].T.astype(np.float32)
                # A pixel outside this zone's coverage reads back as nodata; those
                # samples stay unfilled so a later variant can supply them.
                if src_nodata is not None:
                    good = ~np.all(vals == np.float32(src_nodata), axis=1)
                else:
                    good = np.ones(len(vals), dtype=bool)
                X[sel[good], :] = vals[good]
                filled[sel[good]] = True

    if X is None:
        return None, 0

    dropped = int((~filled).sum())
    if dropped:
        print(f"      {dropped}/{len(filled)} GEDI pixels had no embedding coverage "
              f"-- dropped")
    X, y = X[filled], y_all[filled]
    if len(y) == 0:
        return None, 0

    np.savez_compressed(dest, X=X, y=y)
    return dest.stat().st_size, len(y)


def cmd_tiles(session, args):
    """Cut one output per polygon in a tiling shapefile, named after the tile."""
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.merge import merge as rio_merge
    except ImportError as e:
        sys.exit(f"tiles needs rasterio + geopandas:  pip install rasterio geopandas  ({e})")

    import csv

    tiles = gpd.read_file(args.shapefile)
    if tiles.crs is None:
        sys.exit("Shapefile has no CRS. Set one (e.g. with --tile-crs) before proceeding.")
    if args.tile_crs:
        tiles = tiles.set_crs(args.tile_crs, allow_override=True)

    field = args.name_field or guess_name_field(tiles.columns)
    if not field:
        sys.exit(f"Could not guess the tile-name column. Columns: {list(tiles.columns)}\n"
                 f"Pass --name-field.")
    print(f"Naming tiles from column '{field}' ({len(tiles)} features)")

    names = [sanitize(v) for v in tiles[field]]
    if len(set(names)) != len(names):
        print("  WARNING: tile names are not unique -- outputs will collide.", file=sys.stderr)

    if args.only:
        want = {sanitize(v) for v in args.only}
        keep = [sanitize(v) in want for v in tiles[field]]
        tiles = tiles[keep]
        if tiles.empty:
            sys.exit(f"--only matched no tiles. Wanted: {sorted(want)}")
        print(f"  --only: {len(tiles)} of {len(names)} tile(s) selected")

    bands = parse_bands(args.bands)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz_out = Path(args.npz_dir) if args.npz_dir else out
    npz_out.mkdir(parents=True, exist_ok=True)
    if args.npz in ("sparse", "both") and not args.gedi_dir:
        sys.exit("--npz sparse/both needs --gedi-dir (the healpix_<id>_gedi.tif rasters)")
    manifest_rows = []
    done = 0
    # tile_name -> [(tif path, tile value)], one entry per CRS cut of that tile.
    from collections import defaultdict
    pending = defaultdict(list)

    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="536870912",
    )

    src = Sources(session, args, rasterio)
    with env:
        src.open()
        if not src.datasets:
            sys.exit("No readable source COGs.")
        crs_keys = list(src.groups)
        multi_crs = len(crs_keys) > 1

        for crs_key in crs_keys:
            projected = tiles.to_crs(crs_key)
            print(f"\n[{crs_key}]")
            for i, (_, row) in enumerate(projected.iterrows(), 1):
                tile_name = sanitize(row[field])
                suffix = f"__{crs_key.replace(':', '')}" if multi_crs else ""
                dest = out / args.name_template.format(tile=tile_name + suffix)

                if dest.exists() and not args.overwrite:
                    # Still register it: the sparse pass runs at the end over
                    # `pending`, so a tile skipped because it was downloaded by an
                    # EARLIER run must be included or the resume would produce npz
                    # only for tiles fetched in this run.
                    pending[tile_name].append((dest, row[field]))
                    continue

                if src.maybe_refresh(args.refresh_min):
                    pass
                group = src.groups.get(crs_key, [])

                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                bounds = geom.bounds
                hits = [d for d in group if bbox_overlaps(d.bounds, bounds)]
                if not hits:
                    continue

                try:
                    nodata = args.nodata
                    if nodata is None:
                        nodata = next((d.nodata for d in hits if d.nodata is not None), None)

                    shape = write_tile_chunked(
                        rasterio, rio_merge, hits, bounds, dest, bands, nodata,
                        block=args.window_size, retries=args.window_retries,
                        geom=geom if args.clip else None, clip=args.clip,
                        snap=not args.no_snap,
                        tags={"healpix_tile": str(row[field]),
                              "source_field": field,
                              "n_source_cogs": str(len(hits))},
                    )
                    if shape is None:
                        continue
                    width_px, height_px, count = shape

                    extra = ""
                    npz_bytes = n_samples = 0
                    if args.npz in ("dense", "both"):
                        # Dense stays per-CRS: two cuts in different projections
                        # cannot be one array, so the name keeps the suffix the
                        # GeoTIFF has. (Sparse is merged across CRS below.)
                        dpath = npz_out / f"{tile_name}{suffix}_olmo_dense.npz"
                        npz_bytes = dense_npz(rasterio, dest, dpath)
                        extra += f", dense npz {human(npz_bytes)}"

                    # Sparse is deferred until every CRS group has been cut, so a
                    # tile straddling a zone boundary yields ONE npz drawing on all
                    # of its cuts rather than one per CRS overwriting the other.
                    pending[tile_name].append((dest, row[field]))

                    manifest_rows.append(
                        {
                            "tile": str(row[field]),
                            "file": dest.name,
                            "crs": crs_key,
                            "width": width_px,
                            "height": height_px,
                            "bands": count,
                            "n_sources": len(hits),
                            "bytes": dest.stat().st_size if dest.exists() else 0,
                            "npz_bytes": npz_bytes,
                            "n_samples": n_samples,
                        }
                    )
                    print(
                        f"  [{i}/{len(projected)}] {dest.name}: "
                        f"{width_px}x{height_px} px, {count} band(s), "
                        f"{len(hits)} source(s)"
                        f"{', ' + human(dest.stat().st_size) if dest.exists() else ''}{extra}"
                    )
                except Exception as e:
                    print(f"  [{i}/{len(projected)}] {tile_name}: FAILED ({e})", file=sys.stderr)

                done += 1
                if args.limit and done >= args.limit:
                    print(f"  --limit {args.limit} reached, stopping.")
                    break
            if args.limit and done >= args.limit:
                break

        # ---- sparse npz, once per tile, across every CRS cut of that tile ----
        if args.npz in ("sparse", "both") and pending:
            print(f"\n[sparse npz] {len(pending)} tile(s)")
            for tile_name, entries in sorted(pending.items()):
                tif_paths = [p for p, _ in entries if p.exists()]
                tile_value = entries[0][1]
                if not tif_paths:
                    continue
                gpath = find_gedi(Path(args.gedi_dir), tile_value)
                if gpath is None:
                    print(f"  {tile_name}: NO GEDI raster, skipped")
                    continue
                spath = npz_out / f"healpix_{sanitize(tile_value)}_olmo.npz"
                if spath.exists() and not args.overwrite:
                    print(f"  {spath.name}: exists, kept (use --overwrite to replace)")
                    continue
                try:
                    res = sparse_npz(rasterio, tif_paths, gpath, spath)
                except Exception as e:
                    print(f"  {tile_name}: FAILED ({e})", file=sys.stderr)
                    continue
                if res[0] is None:
                    print(f"  {tile_name}: no valid GEDI pixels with coverage")
                else:
                    nbytes, n = res
                    src_note = f" from {len(tif_paths)} CRS cut(s)" if len(tif_paths) > 1 else ""
                    print(f"  {spath.name}: {n} samples, {human(nbytes)}{src_note}")

        if args.no_keep_tif and args.npz != "none":
            for entries in pending.values():
                for p, _ in entries:
                    p.unlink(missing_ok=True)

        src.close()

    if manifest_rows:
        mpath = out / "manifest.csv"
        with open(mpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]))
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"\n{len(manifest_rows)} tile(s) written. Manifest: {mpath}")
        missing = len(tiles) - len({r['tile'] for r in manifest_rows})
        if missing > 0:
            print(f"{missing} tile(s) produced no output (no overlap, or failed).")
    else:
        print("\nNothing written -- check that the tiling and the COGs actually overlap.")


def cmd_rebuild(session, args):
    """Regenerate the .npz files from GeoTIFFs already on disk -- no API, no download.

    Use after a change to how the npz are built (e.g. the cross-CRS sparse merge, or
    the nodata drop) so a fix does not cost another full download. Tiles cut in more
    than one CRS are grouped back together by name: `29846__EPSG32610.tif` and
    `29846__EPSG32611.tif` both belong to tile 29846 and jointly produce ONE sparse npz.
    """
    try:
        import rasterio
    except ImportError as e:
        sys.exit(f"rebuild needs rasterio:  pip install rasterio  ({e})")

    from collections import defaultdict

    tif_dir = Path(args.tif_dir)
    if not tif_dir.is_dir():
        sys.exit(f"--tif-dir is not a directory: {tif_dir}")
    npz_out = Path(args.npz_dir) if args.npz_dir else tif_dir
    npz_out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in sorted(tif_dir.glob("*.tif")):
        groups[p.stem.split("__")[0]].append(p)

    if args.only:
        want = {sanitize(v) for v in args.only}
        groups = {k: v for k, v in groups.items() if k in want}

    print(f"tif dir : {tif_dir}")
    print(f"tiles   : {len(groups)}  ({sum(len(v) for v in groups.values())} tif(s))")
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"multi-CRS tiles: {len(multi)}\n")

    n_ok = n_fail = 0
    for tile_name, paths in sorted(groups.items()):
        note = f" [{len(paths)} CRS cuts]" if len(paths) > 1 else ""
        if args.npz in ("dense", "both"):
            for p in paths:
                dpath = npz_out / f"{p.stem}_olmo_dense.npz"
                if dpath.exists() and not args.overwrite:
                    print(f"  {dpath.name}: exists, skipped (use --overwrite)")
                    continue
                try:
                    nbytes = dense_npz(rasterio, p, dpath)
                    print(f"  {dpath.name}: {human(nbytes)}")
                    n_ok += 1
                except Exception as e:
                    print(f"  {p.name}: dense FAILED ({e})", file=sys.stderr)
                    n_fail += 1

        if args.npz in ("sparse", "both"):
            gpath = find_gedi(Path(args.gedi_dir), tile_name)
            if gpath is None:
                print(f"  {tile_name}: no GEDI raster, skipped")
                continue
            spath = npz_out / f"healpix_{tile_name}_olmo.npz"
            if spath.exists() and not args.overwrite:
                print(f"  {spath.name}: exists, skipped (use --overwrite)")
                continue
            try:
                res = sparse_npz(rasterio, paths, gpath, spath)
            except Exception as e:
                print(f"  {tile_name}: FAILED ({e})", file=sys.stderr)
                n_fail += 1
                continue
            if res[0] is None:
                print(f"  {tile_name}: no valid GEDI pixels with coverage{note}")
            else:
                nbytes, n = res
                print(f"  {spath.name}: {n} samples, {human(nbytes)}{note}")
                n_ok += 1

    print(f"\n{n_ok} tile(s) rebuilt, {n_fail} failed.")


def cmd_probe(session, args):
    """Fire candidate request bodies at the search endpoint to learn its schema.

    A slow/hanging search usually means the filter key was ignored and you asked
    for the entire workspace. This shows which keys the server actually honours.
    """
    pid = args.prediction_id[0] if args.prediction_id else None
    trials = [
        ("baseline, no include", {}),
        ("limit only", {"limit": 5}),
        ("page_size only", {"page_size": 5}),
        ("per_page only", {"per_page": 5}),
        ("size only", {"size": 5}),
    ]
    if pid:
        trials += [
            ("prediction_ids (list)", {"prediction_ids": [pid]}),
            ("prediction_id (scalar)", {"prediction_id": pid}),
            ("id (scalar)", {"id": pid}),
            ("prediction_ids + limit", {"prediction_ids": [pid], "limit": 5}),
            ("prediction_ids + limit + output_files",
             {"prediction_ids": [pid], "limit": 5, "include": ["output_files"]}),
        ]
    if args.project_id:
        trials.append(("project_id", {"project_id": args.project_id}))
    if args.extra:
        trials.append(("--extra", json.loads(args.extra)))

    print(f"Probing {BASE}{args.path}  (timeout {args.timeout}s each)\n")
    print(f"{'trial':<42} {'status':<9} {'secs':>6} {'records':>8}  keys")
    print("-" * 100)
    baseline_n = None
    for label, body in trials:
        t0 = time.time()
        try:
            payload = post(session, args.path, body, args.verbose,
                           args.timeout, raise_on_error=False)
            dt = time.time() - t0
            if payload is None:
                print(f"{label:<42} {'ERROR':<9} {dt:6.1f} {'-':>8}  (see stderr above)")
                continue
            recs = records_of(payload)
            n = len(recs)
            keys = sorted(recs[0].keys())[:8] if recs and isinstance(recs[0], dict) else []
            if label.startswith("baseline"):
                baseline_n = n
            flag = ""
            if baseline_n is not None and not label.startswith("baseline"):
                flag = "  <-- IGNORED?" if n == baseline_n and n > 0 else "  <-- honoured"
            print(f"{label:<42} {'ok':<9} {dt:6.1f} {n:>8}  {','.join(keys)}{flag}")
        except requests.exceptions.Timeout:
            print(f"{label:<42} {'TIMEOUT':<9} {args.timeout:6.1f} {'-':>8}")
        except Exception as e:
            print(f"{label:<42} {'FAIL':<9} {time.time()-t0:6.1f} {'-':>8}  {e}")

    print(
        "\nRead this as: a trial returning the same count as the baseline means the\n"
        "server ignored your filter. Anything that returns fewer, fast, is the key\n"
        "you want -- pass it via --extra."
    )


def main():
    # Global flags live on BOTH the main parser and every subparser, so that
    #   ... --verbose files --prediction-id X
    #   ... files --prediction-id X --verbose
    # both work. argparse.SUPPRESS is what makes that safe: without it the
    # subparser's default would overwrite a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", default=argparse.SUPPRESS)
    common.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="echo requests")
    common.add_argument("--raw", action="store_true", default=argparse.SUPPRESS,
                        help="dump raw JSON responses")
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="HTTP read timeout, seconds")
    common.add_argument("--extra", default=argparse.SUPPRESS,
                        help='JSON merged into the search body, e.g. \'{"limit":50}\'')

    p = argparse.ArgumentParser(description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(api_key=os.environ.get("OLMOEARTH_API_KEY"), verbose=False,
                   raw=False, timeout=300.0, extra=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", parents=[common], help="discover which filter/pagination keys the API honours")
    sp.add_argument("--prediction-id", nargs="+")
    sp.add_argument("--project-id")
    sp.add_argument("--path", default="/prediction-results/search")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("list", parents=[common], help="list predictions")
    sp.add_argument("--project-id")
    sp.add_argument("--model-id")
    sp.set_defaults(func=cmd_list)

    for name, fn in (("files", cmd_files), ("get", cmd_get), ("crop", cmd_crop)):
        sp = sub.add_parser(name, parents=[common], help=f"{name} for a prediction")
        sp.add_argument("--prediction-id", nargs="+", help="one or more prediction UUIDs")
        sp.add_argument("--project-id")
        sp.add_argument("--result-id", nargs="+",
                        help="prediction_result UUID(s), skipping the prediction lookup")
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

    sp = sub.add_parser("rebuild", parents=[common],
                        help="regenerate .npz from GeoTIFFs already on disk (no API)")
    sp.add_argument("--tif-dir", required=True, help="dir of <tile>[__EPSGxxxxx].tif")
    sp.add_argument("--gedi-dir", required=True)
    sp.add_argument("--npz-dir", help="where to write .npz (default: --tif-dir)")
    sp.add_argument("--npz", choices=("sparse", "dense", "both"), default="sparse")
    sp.add_argument("--only", nargs="+", metavar="TILE")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_rebuild)

    sp = sub.add_parser("tiles", parents=[common], help="cut one output per polygon in a tiling shapefile")
    sp.add_argument("--prediction-id", nargs="+", help="one or more prediction UUIDs")
    sp.add_argument("--project-id")
    sp.add_argument("--result-id", nargs="+",
                    help="prediction_result UUID(s), skipping the prediction lookup")
    sp.add_argument("--match", help="regex; only read source files that match")
    sp.add_argument("--shapefile", required=True, help=".shp/.gpkg/.geojson of tile polygons")
    sp.add_argument("--name-field", help="column holding the tile name (auto-detected if omitted)")
    sp.add_argument("--tile-crs", help="override the shapefile CRS, e.g. EPSG:4326")
    sp.add_argument("--name-template", default="{tile}.tif")
    sp.add_argument("--bands", help="e.g. 1-64,100 (1-indexed)")
    sp.add_argument("--clip", action="store_true",
                    help="mask to the polygon, not just its bounding box")
    sp.add_argument("--nodata", type=float, help="override nodata (OlmoEarth int8 uses -128)")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--refresh-min", type=float, default=30.0,
                    help="re-mint signed URLs every N minutes (0 to disable)")
    sp.add_argument("--out", default="./by_tile")
    # --- chunking: a whole healpix tile in one merge() is what kept failing ---
    sp.add_argument("--window-size", type=int, default=1024, metavar="PX",
                    help="read/write the tile in PX x PX windows (default 1024). "
                         "Smaller = less RAM per step and shorter HTTP reads.")
    sp.add_argument("--window-retries", type=int, default=3,
                    help="retries per window, with exponential backoff (default 3)")
    sp.add_argument("--no-snap", action="store_true",
                    help="do NOT align the output grid to the source pixel grid. "
                         "Leaving this off is recommended: snapping avoids a rigid "
                         "sub-pixel (<=20 m at 40 m) shift of the whole tile.")
    sp.add_argument("--limit", type=int, default=0, metavar="N",
                    help="stop after N tiles -- use this for a smoke test")
    sp.add_argument("--only", nargs="+", metavar="TILE",
                    help="only these tile names/ids, e.g. --only 29846 -- pairs well "
                         "with --limit for a targeted smoke test")
    # --- npz outputs ---
    sp.add_argument("--npz", choices=("none", "sparse", "dense", "both"), default="none",
                    help="also write .npz. 'sparse' = (X, y) at valid GEDI pixels, the "
                         "presto/tessera/prithvi convention (needs --gedi-dir); "
                         "'dense' = whole grid + georeferencing; 'both' = both.")
    sp.add_argument("--gedi-dir", help="dir of healpix_<id>_gedi.tif, for --npz sparse/both")
    sp.add_argument("--npz-dir", help="where to write .npz (default: --out)")
    sp.add_argument("--no-keep-tif", action="store_true",
                    help="delete the GeoTIFF once the .npz is written")
    sp.set_defaults(func=cmd_tiles)

    args = p.parse_args()
    if args.cmd == "rebuild":          # local-only, no API call
        args.func(None, args)
        return
    if not args.api_key:
        sys.exit("Set OLMOEARTH_API_KEY or pass --api-key")
    args.func(make_session(args.api_key), args)


if __name__ == "__main__":
    main()