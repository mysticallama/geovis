#!/usr/bin/env python3
"""
Water & Food Infrastructure Monitoring Pipeline — Yemen (Western AOI)
=====================================================================

A slimmed-down pipeline that removes NASA FIRMS thermal data and replaces it
with a richer OSM infrastructure query (water + food tags) and WFP logistics
cluster data, producing a Mapbox GL JS web application for interactive analysis.

Steps
-----
  Step 1 · Load AOI
  Step 2 · Query ACLED conflict events  (full AOI, explicit date range)
  Step 3 · Query OSM infrastructure     (water + food tags, full AOI)
  Step 4 · Download Planet imagery      (best scene per target date ±window)
  Step 5 · xView / YOLOv8 inference    (annotated tiles, CSV, OSM proximity)
  Step 6 · Download WFP logistics data  (HDX / LogIE → roads, airports, …)
  Step 7 · ACLED + WFP Mapbox web app  (interactive heatmap + vector overlay)

Usage
-----
    python run_pipeline.py                      # full pipeline
    python run_pipeline.py --skip-imagery       # skip Planet download (Steps 4–5)
    python run_pipeline.py --skip-yolo          # skip xView inference (Step 5)
    python run_pipeline.py --skip-wfp           # skip WFP download (Step 6)
    python run_pipeline.py --skip-fusion        # skip Mapbox generation (Step 7)
    python run_pipeline.py --debug-imagery      # Planet search diagnostic (no download)
    python run_pipeline.py --debug-yolo         # xView model load diagnostic
    python run_pipeline.py --debug-acled        # ACLED auth + query diagnostic
    python run_pipeline.py --debug-all          # run all diagnostics
    python run_pipeline.py --verbose            # DEBUG-level logging at every step
    python run_pipeline.py --osm-extended       # use extended OSM tag set

Environment variables (.env)
----------------------------
    PL_API_KEY        — Planet Labs API key         (required for Step 4)
    ACLED_EMAIL       — ACLED registered e-mail     (required for Step 2)
    ACLED_PASSWORD    — ACLED account password      (required for Step 2)
    MAPBOX_TOKEN      — Mapbox public token         (required for Step 7)
    HDX_API_KEY       — HDX API key (optional)      (optional for Step 6)
"""

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Root of the working_paper_version project (one level up from country_runs/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIGURATION — Edit these values for your analysis
# =============================================================================

# ── AOI ──────────────────────────────────────────────────────────────────────
# Bounding box [west, south, east, north]
# Western Yemen: Red Sea coast → Sanaa highlands (Hodeidah, Taiz, Ibb, Sanaa, Hajjah)
BBOX = [42.5, 12.5, 46.5, 16.5]   # Western Yemen

# ── Date range for ACLED ─────────────────────────────────────────────────────
ACLED_START_DATE = "2021-01-01"   # Inclusive start (YYYY-MM-DD)
ACLED_END_DATE   = "2026-02-24"   # Inclusive end   (YYYY-MM-DD)

# ── Planet imagery — before / after target dates ──────────────────────────────
BEFORE_DATE = "2025-10-01"   # Baseline imagery target date (YYYY-MM-DD)
AFTER_DATE  = "2025-10-14"   # Comparison imagery target date (YYYY-MM-DD)
PLANET_SEARCH_WINDOW_DAYS = 7    # ±days to search around each target date
PLANET_MAX_CLOUD_COVER    = 10.0  # percent (0–100)
PLANET_MIN_COVERAGE       = 0.90  # minimum AOI coverage fraction (0–1)
PLANET_TILE_SIZE          = 640   # 640×640 px tiles (xView standard)

# ── xView / YOLOv8 (Step 5) ──────────────────────────────────────────────────
YOLO_MODEL_PATH = "xview.pt"   # path to xView YOLOv8 weights
YOLO_CONFIDENCE = 0.25
YOLO_IOU        = 0.45
YOLO_DEVICE     = "auto"       # "auto", "cpu", "cuda", "mps"
# xView class IDs of interest (0-indexed YOLO IDs, remapped from sparse xView type_ids):
#   15 = truck_w_liquid          (type_id 29)
#   21 = tank_car                (type_id 37)
#   33 = engineering_vehicle     (type_id 53)
#   40 = haul_truck              (type_id 61)
#   50 = damaged_building        (type_id 76)
#   55 = storage_tank            (type_id 86)
#   56 = shipping_container_lot  (type_id 89)
#   57 = shipping_container      (type_id 91)
#   59 = tower                   (type_id 94)
YOLO_CLASSES_OF_INTEREST     = [15, 21, 33, 40, 50, 55, 56, 57, 59]
OSM_PROXIMITY_RADIUS_METERS  = 500  # radius for OSM feature cross-reference
SAVE_ANNOTATED_TILES         = True # draw detection boxes on tile PNGs

# ── WFP logistics (Step 6) ───────────────────────────────────────────────────
WFP_COUNTRY_ISO3 = "YEM"   # ISO 3166-1 alpha-3 country code
WFP_COUNTRY_NAME = "Yemen"  # Human-readable name for search

# ── Mapbox web app (Step 7) ──────────────────────────────────────────────────
MAP_TITLE   = f"Conflict & Logistics: {WFP_COUNTRY_NAME} Western ({ACLED_START_DATE} → {ACLED_END_DATE})"
MAP_ZOOM    = 7.0
# Map centre [longitude, latitude] — defaults to AOI centroid at runtime

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = _PROJECT_ROOT / "output_yemen"

# =============================================================================
# END CONFIGURATION
# =============================================================================

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def _set_verbose():
    for name in [
        "run_pipeline",
        "water_osm_pipeline.acled",
        "water_osm_pipeline.osm",
        "water_osm_pipeline.imagery",
        "water_osm_pipeline.yolov8",
        "water_osm_pipeline.wfp",
        "water_osm_pipeline.fusion",
        "water_osm_pipeline.aoi",
    ]:
        logging.getLogger(name).setLevel(logging.DEBUG)


@contextmanager
def _step(number: int, title: str):
    width = 62
    print()
    print("=" * width)
    print(f"[Step {number}] {title}")
    print("=" * width)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"  ↳ completed in {elapsed:.1f}s")


def _load_geojson(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _count(fc: dict) -> int:
    return len(fc.get("features", [])) if isinstance(fc, dict) else 0


def _save_geojson(geojson: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(geojson, fh, indent=2)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step1_load_aoi(verbose: bool = False) -> dict:
    from water_osm_pipeline.aoi import AOIHandler
    west, south, east, north = BBOX
    aoi = AOIHandler.from_bbox(west=west, south=south, east=east, north=north)
    print(f"  AOI bounds:  {AOIHandler.get_bounds(aoi)}")
    print(f"  Area:        ~{AOIHandler.area_sq_km(aoi):,.0f} km²")
    return aoi


def step2_acled(aoi: dict, verbose: bool = False) -> dict:
    from water_osm_pipeline.acled import ACLEDClient, DEFAULT_VIOLENCE_SUB_EVENTS

    print(f"  Date range:   {ACLED_START_DATE} → {ACLED_END_DATE}")
    print(f"  Query area:   full AOI (independent of imagery)")
    print(f"  Sub-events:   violence filter = {DEFAULT_VIOLENCE_SUB_EVENTS[:3]}…")

    client = ACLEDClient()
    df = client.query(
        geometry=aoi,
        start_date=ACLED_START_DATE,
        end_date=ACLED_END_DATE,
        filter_violence=True,
    )
    fc = client.to_geojson(df)
    n = _count(fc)
    print(f"  Conflict events found: {n}")

    if n > 0:
        out = OUTPUT_DIR / "conflict_events.geojson"
        _save_geojson(fc, out)
        print(f"  Saved → {out}")

    if verbose and n > 0:
        types: dict = {}
        for feat in fc.get("features", []):
            et = feat.get("properties", {}).get("event_type", "unknown")
            types[et] = types.get(et, 0) + 1
        for et, cnt in sorted(types.items(), key=lambda x: -x[1])[:6]:
            print(f"    {et}: {cnt}")
    return fc


def step3_osm(aoi: dict, extended: bool = False, verbose: bool = False) -> dict:
    from water_osm_pipeline.osm import query_infrastructure, save_geojson

    tag_mode = "extended" if extended else "standard"
    print(f"  Query area:   full AOI bounding box")
    print(f"  Tag set:      {tag_mode} (water + food infrastructure)")

    infra = query_infrastructure(aoi, extended=extended)
    n = _count(infra)
    print(f"  Infrastructure features found: {n}")

    if n > 0:
        out = OUTPUT_DIR / "infrastructure.geojson"
        save_geojson(infra, out)
        print(f"  Saved → {out}")

    if verbose and n > 0:
        cats: dict = {}
        for feat in infra.get("features", []):
            c = feat.get("properties", {}).get("infra_category", "other")
            cats[c] = cats.get(c, 0) + 1
        for c, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {c}: {cnt}")
    return infra


def step4_imagery(aoi: dict, verbose: bool = False) -> dict:
    from water_osm_pipeline.imagery import PlanetImageryClient, PlanetConfig, download_date_pair

    api_key = os.environ.get("PL_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        print("  SKIPPED: PL_API_KEY not set in .env")
        print("  Set PL_API_KEY to enable Planet imagery download.")
        return {}

    cfg = PlanetConfig(
        planet_api_key=api_key,
        max_cloud_cover=PLANET_MAX_CLOUD_COVER,
        min_coverage=PLANET_MIN_COVERAGE,
        tile_size=PLANET_TILE_SIZE,
        search_window_days=PLANET_SEARCH_WINDOW_DAYS,
    )

    print(f"  Before date:        {BEFORE_DATE}  (±{PLANET_SEARCH_WINDOW_DAYS} days)")
    print(f"  After date:         {AFTER_DATE}  (±{PLANET_SEARCH_WINDOW_DAYS} days)")
    print(f"  Max cloud cover:    {cfg.max_cloud_cover}%")
    print(f"  Min AOI coverage:   {cfg.min_coverage*100:.0f}%")
    print(f"  Tile size:          {cfg.tile_size} × {cfg.tile_size} px")

    imagery_dir = OUTPUT_DIR / "imagery"
    result = download_date_pair(
        aoi=aoi,
        before_date=BEFORE_DATE,
        after_date=AFTER_DATE,
        output_dir=imagery_dir,
        config=cfg,
    )

    n_images = len(result.get("images", []))
    n_tiles  = len(result.get("tiles", []))
    print(f"  Images downloaded:  {n_images}")
    print(f"  Tiles generated:    {n_tiles}")
    if result.get("tile_metadata_path"):
        print(f"  Tile metadata →     {result['tile_metadata_path']}")
    if verbose and result.get("images"):
        for p in result["images"]:
            print(f"    {p}")
    return result


def step5_yolo(
    imagery_result: dict,
    infrastructure: dict,
    skip: bool = False,
    verbose: bool = False,
) -> dict:
    if skip:
        print("  SKIPPED (--skip-yolo or --skip-imagery flag set)")
        return {}

    metadata_path = imagery_result.get("tile_metadata_path")
    if not metadata_path or not Path(metadata_path).exists():
        print("  SKIPPED: no tile metadata found from Step 4")
        print("  Run with a valid PL_API_KEY to produce tiles first.")
        return {}

    from water_osm_pipeline.yolov8 import (
        YOLOv8Config, detect_and_annotate,
        XVIEW_TRAINING_GSD_M, PLANET_INFERENCE_GSD_M,
    )

    cfg = YOLOv8Config(
        model_path=YOLO_MODEL_PATH,
        training_gsd_m=XVIEW_TRAINING_GSD_M,
        inference_gsd_m=PLANET_INFERENCE_GSD_M,
        confidence_threshold=YOLO_CONFIDENCE,
        iou_threshold=YOLO_IOU,
        device=YOLO_DEVICE,
        classes_of_interest=YOLO_CLASSES_OF_INTEREST,
        osm_proximity_radius_m=OSM_PROXIMITY_RADIUS_METERS,
        save_annotated_tiles=SAVE_ANNOTATED_TILES,
    )

    print(f"  Model:              {cfg.model_path}")
    print(f"  GSD correction:     {cfg.training_gsd_m} m → {cfg.inference_gsd_m} m ({cfg.scale_factor:.0f}×)")
    print(f"  Effective imgsz:    {cfg.effective_imgsz} px")
    print(f"  Confidence:         {cfg.confidence_threshold}")
    print(f"  OSM proximity:      {cfg.osm_proximity_radius_m:.0f} m radius")
    print(f"  Annotated tiles:    {cfg.save_annotated_tiles}")

    yolo_dir = OUTPUT_DIR / "yolo"
    result = detect_and_annotate(
        tile_metadata_path=metadata_path,
        output_dir=yolo_dir,
        osm_geojson=infrastructure if _count(infrastructure) > 0 else None,
        config=cfg,
    )

    comparison = result.get("comparison", [])
    by_status: dict = {}
    for r in comparison:
        s = r.get("change_status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    n_near = sum(1 for r in comparison if r.get("near_osm_feature"))
    print(f"  Object clusters:    {len(comparison)}")
    for s, cnt in sorted(by_status.items()):
        print(f"    {s}: {cnt}")
    print(f"  Near OSM features:  {n_near}")
    if result.get("geojson_path"):
        print(f"  GeoJSON →           {result['geojson_path']}")
    if result.get("csv_path"):
        print(f"  CSV →               {result['csv_path']}")
    if result.get("annotated_paths"):
        print(f"  Annotated tiles:    {len(result['annotated_paths'])} files")

    if verbose:
        for r in comparison[:5]:
            print(
                f"    [{r['change_status']}] {r['class_name']} "
                f"conf={r['confidence']:.3f}  "
                f"osm={r.get('nearest_osm_type')}@{r.get('nearest_osm_dist_m')} m"
            )
    return result


def step6_wfp(aoi: dict, skip: bool = False, verbose: bool = False) -> dict:
    if skip:
        print("  SKIPPED (--skip-wfp flag set)")
        return {}

    from water_osm_pipeline.wfp import WFPLogisticsClient, WFPConfig

    cfg = WFPConfig(
        country_iso3=WFP_COUNTRY_ISO3,
        country_name=WFP_COUNTRY_NAME,
        fallback_to_osm=True,
    )

    print(f"  Country:     {WFP_COUNTRY_NAME} ({WFP_COUNTRY_ISO3})")
    print(f"  Source:      HDX (fallback → OSM Overpass)")

    wfp_dir = OUTPUT_DIR / "wfp"
    client = WFPLogisticsClient(cfg)
    layers = client.fetch(aoi_geometry=aoi, output_dir=wfp_dir)

    for lname, fc in layers.items():
        n = _count(fc)
        if n:
            print(f"  {lname:<14s}: {n} features")

    if verbose:
        for lname, fc in layers.items():
            feat = fc.get("features", [])
            if feat:
                sample_props = list(feat[0].get("properties", {}).keys())[:5]
                print(f"    [{lname}] sample props: {sample_props}")

    return layers


def step7_fusion(
    conflicts: dict,
    wfp_layers: dict,
    aoi: dict,
    skip: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    if skip:
        print("  SKIPPED (--skip-fusion flag set)")
        return None

    from water_osm_pipeline.fusion import generate_mapbox_app, MapboxFusionConfig
    from water_osm_pipeline.aoi import AOIHandler

    mapbox_token = os.environ.get("MAPBOX_TOKEN", "")
    if not mapbox_token or mapbox_token.startswith("your-"):
        print("  WARNING: MAPBOX_TOKEN not set — map will not render in browser")
        print("  Set MAPBOX_TOKEN in .env to enable Mapbox tile layers")

    cx, cy = AOIHandler.get_centroid(aoi)
    cfg = MapboxFusionConfig(
        mapbox_token=mapbox_token,
        initial_center=(cx, cy),
        initial_zoom=MAP_ZOOM,
    )

    print(f"  ACLED events:  {_count(conflicts)}")
    for lname, fc in wfp_layers.items():
        n = _count(fc)
        if n:
            print(f"  WFP {lname:<12s}: {n} features")
    print(f"  Map title:     {MAP_TITLE}")

    fusion_dir = OUTPUT_DIR / "fusion"
    html_path  = fusion_dir / "map.html"

    out = generate_mapbox_app(
        acled_geojson=conflicts,
        wfp_layers=wfp_layers,
        output_path=html_path,
        config=cfg,
        title=MAP_TITLE,
        start_date=ACLED_START_DATE,
        end_date=ACLED_END_DATE,
    )
    print(f"  Mapbox app →   {out}")
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    conflicts: dict,
    infrastructure: dict,
    imagery_result: dict,
    yolo_result: dict,
    wfp_layers: dict,
    html_path,
):
    width = 62
    print()
    print("=" * width)
    print("PIPELINE SUMMARY")
    print("=" * width)
    print(f"  ACLED range:    {ACLED_START_DATE} → {ACLED_END_DATE}")
    print(f"  Bounding box:   {BBOX}")
    print(f"  Before/After:   {BEFORE_DATE} / {AFTER_DATE}")
    print("-" * width)
    print(f"  ACLED events:           {_count(conflicts)}")
    print(f"  OSM features:           {_count(infrastructure)}")
    if imagery_result:
        print(f"  Planet images:          {len(imagery_result.get('images', []))}")
        print(f"  Tile patches (640×640): {len(imagery_result.get('tiles', []))}")
    if yolo_result:
        comp = yolo_result.get("comparison", [])
        n_near = sum(1 for r in comp if r.get("near_osm_feature"))
        print(f"  xView clusters:         {len(comp)}  (near OSM: {n_near})")
    for lname, fc in wfp_layers.items():
        n = _count(fc)
        if n:
            print(f"  WFP {lname:<16s}: {n}")
    if html_path:
        print(f"  Mapbox web app:         {html_path}")
    print("-" * width)
    print(f"  Output directory:       {OUTPUT_DIR.absolute()}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    skip_imagery: bool = False,
    skip_yolo: bool = False,
    skip_wfp: bool = False,
    skip_fusion: bool = False,
    osm_extended: bool = False,
    verbose: bool = False,
) -> dict:
    """Execute the full pipeline and return a results dict."""
    if verbose:
        _set_verbose()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 62)
    print("Water & Food Infrastructure Monitoring Pipeline")
    print("=" * 62)
    print(f"  AOI:            {BBOX}")
    print(f"  ACLED window:   {ACLED_START_DATE} → {ACLED_END_DATE}")
    print(f"  Imagery pair:   {BEFORE_DATE} / {AFTER_DATE}")
    print(f"  WFP country:    {WFP_COUNTRY_NAME} ({WFP_COUNTRY_ISO3})")
    print(f"  Output dir:     {OUTPUT_DIR.absolute()}")
    print("=" * 62)

    results: dict = {}

    # ── Step 1 ──────────────────────────────────────────────────────────────
    with _step(1, "Load AOI"):
        aoi = step1_load_aoi(verbose=verbose)
        results["aoi"] = aoi

    # ── Step 2 ──────────────────────────────────────────────────────────────
    with _step(2, "Query conflict events (ACLED — full AOI)"):
        try:
            conflicts = step2_acled(aoi, verbose=verbose)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            print("  Hint: check ACLED_EMAIL / ACLED_PASSWORD in .env")
            conflicts = {"type": "FeatureCollection", "features": []}
        results["conflicts"] = conflicts

    # ── Step 3 ──────────────────────────────────────────────────────────────
    with _step(3, "Query infrastructure (OSM — water + food tags)"):
        try:
            infrastructure = step3_osm(aoi, extended=osm_extended, verbose=verbose)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            infrastructure = {"type": "FeatureCollection", "features": []}
        results["infrastructure"] = infrastructure

    # ── Step 4 ──────────────────────────────────────────────────────────────
    with _step(4, "Download Planet imagery (before + after pair)"):
        if skip_imagery:
            print("  SKIPPED (--skip-imagery flag set)")
            imagery_result = {}
        else:
            try:
                imagery_result = step4_imagery(aoi, verbose=verbose)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                if verbose:
                    traceback.print_exc()
                imagery_result = {}
        results["imagery"] = imagery_result

    # ── Step 5 ──────────────────────────────────────────────────────────────
    with _step(5, "xView / YOLOv8 inference + annotation"):
        try:
            yolo_result = step5_yolo(
                imagery_result,
                infrastructure,
                skip=skip_yolo or skip_imagery,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            print("  Hint: pip install ultralytics")
            yolo_result = {}
        results["yolo"] = yolo_result

    # ── Step 6 ──────────────────────────────────────────────────────────────
    with _step(6, "Download WFP logistics data (HDX)"):
        try:
            wfp_layers = step6_wfp(aoi, skip=skip_wfp, verbose=verbose)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            wfp_layers = {}
        results["wfp"] = wfp_layers

    # ── Step 7 ──────────────────────────────────────────────────────────────
    with _step(7, "Generate ACLED + WFP Mapbox web app"):
        try:
            html_path = step7_fusion(
                conflicts,
                wfp_layers,
                aoi,
                skip=skip_fusion,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            html_path = None
        results["html_path"] = html_path

    _print_summary(conflicts, infrastructure, imagery_result, yolo_result, wfp_layers, html_path)
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def debug_acled():
    """Authenticate with ACLED and run a short test query."""
    print()
    print("=" * 62)
    print("ACLED DIAGNOSTIC")
    print("=" * 62)

    from water_osm_pipeline.acled import ACLEDClient, ACLEDConfig

    cfg = ACLEDConfig()
    if not cfg.email and not cfg.access_token:
        print("  ERROR: ACLED_EMAIL / ACLED_PASSWORD not set in .env")
        return

    print(f"  Email:    {cfg.email}")
    client = ACLEDClient(cfg)

    try:
        token = client._get_token()
        print(f"  Token:    {token[:12]}… (OK)")
    except Exception as exc:
        print(f"  ERROR obtaining token: {exc}")
        return

    # Small test query (limit=5)
    from water_osm_pipeline.aoi import AOIHandler
    west, south, east, north = BBOX
    aoi = AOIHandler.from_bbox(west, south, east, north)

    df = client.query(
        geometry=aoi,
        start_date=ACLED_END_DATE,
        end_date=ACLED_END_DATE,
        limit=5,
    )
    print(f"  Test query result: {len(df)} rows (limit=5)")
    if not df.empty:
        print(f"  Columns: {list(df.columns)[:8]}")
    print("\n[ACLED diagnostic complete]")


def debug_imagery():
    """Planet API search diagnostic — no download."""
    import requests
    from requests.auth import HTTPBasicAuth

    api_key = os.environ.get("PL_API_KEY", "")
    if not api_key:
        print("ERROR: PL_API_KEY not set in .env")
        return

    print()
    print("=" * 62)
    print("PLANET IMAGERY DIAGNOSTIC (search only — no download)")
    print("=" * 62)

    west, south, east, north = BBOX
    aoi_geom = {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south],
                          [east, north], [west, north], [west, south]]],
    }

    for target, lbl in [(BEFORE_DATE, "before"), (AFTER_DATE, "after")]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(target, "%Y-%m-%d")
        win = PLANET_SEARCH_WINDOW_DAYS
        start = (dt - timedelta(days=win)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=win)).strftime("%Y-%m-%d")

        body = {
            "item_types": ["PSScene"],
            "filter": {
                "type": "AndFilter",
                "config": [
                    {"type": "GeometryFilter", "field_name": "geometry", "config": aoi_geom},
                    {"type": "DateRangeFilter", "field_name": "acquired",
                     "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
                    {"type": "RangeFilter", "field_name": "cloud_cover",
                     "config": {"lte": PLANET_MAX_CLOUD_COVER / 100}},
                ],
            },
        }

        resp = requests.post(
            "https://api.planet.com/data/v1/quick-search",
            json=body,
            auth=HTTPBasicAuth(api_key, ""),
            timeout=30,
        )
        print(f"\n  [{lbl}] {start} → {end}  HTTP {resp.status_code}")
        if resp.status_code == 200:
            feats = resp.json().get("features", [])
            print(f"  Scenes found: {len(feats)}")
            for i, f in enumerate(feats[:5]):
                p = f.get("properties", {})
                print(
                    f"    [{i+1}] {p.get('acquired','?')[:10]}  "
                    f"cloud={p.get('cloud_cover','?'):.3f}  "
                    f"id={f.get('id','?')}"
                )
        else:
            print(f"  ERROR: {resp.text[:300]}")

    print("\n[Planet imagery diagnostic complete]")


def debug_yolo():
    """xView model load diagnostic."""
    from water_osm_pipeline.yolov8 import (
        YOLOv8Config, XVIEW_CLASS_NAMES,
        XVIEW_TRAINING_GSD_M, PLANET_INFERENCE_GSD_M,
    )

    cfg = YOLOv8Config(
        model_path=YOLO_MODEL_PATH,
        training_gsd_m=XVIEW_TRAINING_GSD_M,
        inference_gsd_m=PLANET_INFERENCE_GSD_M,
        classes_of_interest=YOLO_CLASSES_OF_INTEREST,
    )

    print()
    print("=" * 62)
    print("xView YOLOv8 DIAGNOSTIC (model load only — no inference)")
    print("=" * 62)
    print(f"  Model path:       {cfg.model_path}")
    print(f"  Training GSD:     {cfg.training_gsd_m} m  (xView / WorldView)")
    print(f"  Inference GSD:    {cfg.inference_gsd_m} m  (Planet PSScene)")
    print(f"  Scale factor:     {cfg.scale_factor:.1f}×")
    print(f"  Effective imgsz:  {cfg.effective_imgsz} px")
    print(f"  Device:           {YOLO_DEVICE}")
    print(f"  Classes of interest:")
    for cls_id in (cfg.classes_of_interest or []):
        print(f"    {cls_id:3d}  {XVIEW_CLASS_NAMES.get(cls_id, 'unknown')}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ERROR: ultralytics not installed — run: pip install ultralytics")
        return

    print("\n  Loading model…")
    t0 = time.perf_counter()
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
        if hasattr(model, "names"):
            print(f"  Total classes: {len(model.names)}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()

    print("\n[xView diagnostic complete]")


def debug_all():
    debug_acled()
    debug_imagery()
    debug_yolo()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_flags():
    args = set(sys.argv[1:])
    return {
        "debug_acled":    "--debug-acled"   in args,
        "debug_imagery":  "--debug-imagery" in args,
        "debug_yolo":     "--debug-yolo"    in args,
        "debug_all":      "--debug-all"     in args,
        "skip_imagery":   "--skip-imagery"  in args,
        "skip_yolo":      "--skip-yolo"     in args,
        "skip_wfp":       "--skip-wfp"      in args,
        "skip_fusion":    "--skip-fusion"   in args,
        "osm_extended":   "--osm-extended"  in args,
        "verbose":        "--verbose"       in args,
    }


if __name__ == "__main__":
    flags = _parse_flags()

    if flags["debug_all"]:
        debug_all()
    elif flags["debug_acled"]:
        debug_acled()
    elif flags["debug_imagery"]:
        debug_imagery()
    elif flags["debug_yolo"]:
        debug_yolo()
    else:
        run_pipeline(
            skip_imagery=flags["skip_imagery"],
            skip_yolo=flags["skip_yolo"],
            skip_wfp=flags["skip_wfp"],
            skip_fusion=flags["skip_fusion"],
            osm_extended=flags["osm_extended"],
            verbose=flags["verbose"],
        )
