#!/usr/bin/env python3
"""
Example: Water & Food Infrastructure Monitoring Pipeline
=========================================================

Demonstrates how to configure and run the pipeline programmatically
for a custom AOI and date range.

This script is the recommended starting point for new analyses.
Edit the CONFIGURATION block below, then run:

    python example.py

Or run specific steps interactively in a Jupyter notebook / REPL.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # reads .env in the current directory

# =============================================================================
# CONFIGURATION — edit these for your analysis
# =============================================================================

# ── Area of Interest ──────────────────────────────────────────────────────────
# Option 1: Bounding box [west, south, east, north]
BBOX = [33.0, 11.0, 37.0, 15.0]   # Central Sudan

# Option 2: Path to a GeoJSON or shapefile
# AOI_FILE = "my_region.geojson"

# ── Date window for ACLED ────────────────────────────────────────────────────
ACLED_START = "2023-10-01"
ACLED_END   = "2024-04-01"

# ── Planet imagery dates (before / after the period of interest) ──────────────
BEFORE_DATE = "2023-10-01"   # baseline — best scene ±WINDOW_DAYS
AFTER_DATE  = "2024-04-01"   # comparison — best scene ±WINDOW_DAYS
WINDOW_DAYS = 7              # search window (days either side of target)

# ── WFP country ──────────────────────────────────────────────────────────────
WFP_ISO3 = "SDN"
WFP_NAME = "Sudan"

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path("./output/example")

# =============================================================================
# END CONFIGURATION
# =============================================================================


def main():
    # ── Step 1: Load AOI ────────────────────────────────────────────────────
    from water_osm_pipeline.aoi import AOIHandler
    aoi = AOIHandler.from_bbox(*BBOX)
    print(f"AOI bounds: {AOIHandler.get_bounds(aoi)}")
    print(f"AOI area:   ~{AOIHandler.area_sq_km(aoi):,.0f} km²\n")

    # ── Step 2: ACLED conflict events ────────────────────────────────────────
    from water_osm_pipeline.acled import ACLEDClient
    client = ACLEDClient()
    df = client.query(
        geometry=aoi,
        start_date=ACLED_START,
        end_date=ACLED_END,
        filter_violence=True,
    )
    conflicts = client.to_geojson(df)
    print(f"ACLED events: {len(conflicts['features'])}")

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    with open(OUT / "conflict_events.geojson", "w") as f:
        json.dump(conflicts, f, indent=2)

    # ── Step 3: OSM infrastructure ──────────────────────────────────────────
    from water_osm_pipeline.osm import query_infrastructure, save_geojson
    infra = query_infrastructure(aoi, extended=False)
    print(f"OSM features: {len(infra['features'])}")
    save_geojson(infra, OUT / "infrastructure.geojson")

    # Category breakdown
    cats: dict = {}
    for feat in infra["features"]:
        c = feat.get("properties", {}).get("infra_category", "other")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # ── Step 4: Planet imagery ───────────────────────────────────────────────
    import os
    if os.environ.get("PL_API_KEY"):
        from water_osm_pipeline.imagery import PlanetConfig, download_date_pair
        cfg = PlanetConfig(
            max_cloud_cover=10.0,
            min_coverage=0.90,
            tile_size=640,
            search_window_days=WINDOW_DAYS,
        )
        imagery = download_date_pair(
            aoi=aoi,
            before_date=BEFORE_DATE,
            after_date=AFTER_DATE,
            output_dir=OUT / "imagery",
            config=cfg,
        )
        print(f"\nPlanet: {len(imagery.get('images', []))} images, "
              f"{len(imagery.get('tiles', []))} tiles")
    else:
        print("\nSkipping imagery (PL_API_KEY not set)")
        imagery = {}

    # ── Step 5: xView inference ──────────────────────────────────────────────
    if imagery.get("tile_metadata_path"):
        from water_osm_pipeline.yolov8 import YOLOv8Config, detect_and_annotate
        yolo_cfg = YOLOv8Config(
            model_path="xview.pt",
            osm_proximity_radius_m=500,
            save_annotated_tiles=True,
        )
        yolo = detect_and_annotate(
            tile_metadata_path=imagery["tile_metadata_path"],
            output_dir=OUT / "yolo",
            osm_geojson=infra,
            config=yolo_cfg,
        )
        print(f"xView: {len(yolo.get('comparison', []))} detection clusters")
    else:
        print("Skipping xView (no tile metadata)")

    # ── Step 6: WFP logistics data ───────────────────────────────────────────
    from water_osm_pipeline.wfp import WFPConfig, WFPLogisticsClient
    wfp_cfg = WFPConfig(country_iso3=WFP_ISO3, country_name=WFP_NAME)
    wfp_client = WFPLogisticsClient(wfp_cfg)
    wfp_layers = wfp_client.fetch(aoi_geometry=aoi, output_dir=OUT / "wfp")
    for lname, fc in wfp_layers.items():
        n = len(fc.get("features", []))
        if n:
            print(f"WFP {lname}: {n} features")

    # ── Step 7: Mapbox web app ───────────────────────────────────────────────
    from water_osm_pipeline.fusion import generate_mapbox_app, MapboxFusionConfig
    from water_osm_pipeline.aoi import AOIHandler
    cx, cy = AOIHandler.get_centroid(aoi)

    map_cfg = MapboxFusionConfig(
        initial_center=(cx, cy),
        initial_zoom=7.0,
    )
    html_path = generate_mapbox_app(
        acled_geojson=conflicts,
        wfp_layers=wfp_layers,
        output_path=OUT / "fusion" / "map.html",
        config=map_cfg,
        title=f"Conflict & Logistics: {WFP_NAME}",
        start_date=ACLED_START,
        end_date=ACLED_END,
    )
    print(f"\nMapbox web app → {html_path}")
    print("Open the HTML file in a browser to explore the map.")


if __name__ == "__main__":
    main()
