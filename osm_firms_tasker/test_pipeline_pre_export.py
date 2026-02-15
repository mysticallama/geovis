#!/usr/bin/env python3
"""
Test script: Run pipeline up to enrichment (before export to Planet/Maxar).

Configure the analysis by editing the CONFIGURATION section below:
- BBOX: Bounding box [west, south, east, north]
- END_DATE: End date for queries (YYYY-MM-DD)
- DAYS_BACK: Number of days to look back from END_DATE
- THERMAL_SOURCE: "gibs" (no API key, ~30-90 days) or "firms_api" (requires MAP_KEY, up to 1 year)

Pipeline flow (FIRMS-first approach):
1. Load AOI
2. Query thermal anomalies from FIRMS (primary data source)
3. Query water infrastructure from OSM (filtered to areas near thermal detections)
4. Query conflict events from ACLED (filtered to areas near thermal detections)
5. Enrich and classify detections

STOPS BEFORE export to satellite APIs (Planet/Maxar).
"""

import json
import logging
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these values for your analysis
# =============================================================================

# Bounding box: [west, south, east, north]
BBOX = [42.5, 12.5, 45.5, 17.5]  # Aden, Yemen

# Date range for queries
# End date: The most recent date to query (YYYY-MM-DD format)
# Days back: How many days to look back from the end date
END_DATE = "2024-02-15"  # GIBS data has ~1-2 day latency from present
DAYS_BACK = 80            # Number of days to query

# Thermal data source: "gibs" or "firms_api"
# - "gibs": NASA GIBS WMTS (no API key required, ~30-90 day archive)
# - "firms_api": NASA FIRMS REST API (requires MAP_KEY, up to 365 days)
THERMAL_SOURCE = "firms_api"

# VIIRS source to query
# For GIBS: VIIRS_NOAA20, VIIRS_SNPP, MODIS_Terra, MODIS_Aqua
# For FIRMS API (Standard Processing, up to 365 days): VIIRS_NOAA20_SP, VIIRS_SNPP_SP, MODIS_SP
# For FIRMS API (NRT, ~60 days): VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT, MODIS_NRT
VIIRS_SOURCE = "VIIRS_NOAA20_SP"

# Confidence filter (FIRMS API only)
# - None: All detections
# - "high" or "h": High confidence only
# - "nominal" or "n": Nominal and high confidence
CONFIDENCE_FILTER = None

# Proximity radius for filtering OSM/ACLED around thermal detections (meters)
# OSM infrastructure and ACLED events will only be queried within this radius
# of each thermal detection point
PROXIMITY_RADIUS_METERS = 500

# =============================================================================
# API CREDENTIALS
# GIBS: No API key required!
# FIRMS API: Requires MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/
# ACLED: OAuth authentication with email/password
# =============================================================================
os.environ["FIRMS_MAP_KEY"] = ""
os.environ["ACLED_EMAIL"] = ""
os.environ["ACLED_PASSWORD"] = ""
# =============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from water_pipeline import (
    AOIHandler,
    query_water_infrastructure,
    enrich_detections,
    save_geojson,
)


def query_thermal_gibs(aoi, days, source, end_date):
    """Query thermal anomalies using NASA GIBS WMTS (no API key)."""
    from water_pipeline.firms import FIRMSClient, FIRMSConfig

    print(f"         Using GIBS WMTS (no API key required)")
    print(f"         Source: {source}")

    config = FIRMSConfig()
    client = FIRMSClient(config)

    # Map FIRMS API source names to GIBS names if needed
    gibs_source = source
    if source.endswith("_NRT") or source.endswith("_SP"):
        gibs_source = source.rsplit("_", 1)[0]

    features = client.query(aoi, days=days, source=gibs_source, date=end_date)
    return client.to_geojson(features)


def query_thermal_firms_api(aoi, days, source, end_date, confidence_filter=None):
    """Query thermal anomalies using NASA FIRMS REST API (requires MAP_KEY)."""
    from water_pipeline.firms_api import FIRMSAPIClient, FIRMSAPIConfig, MAX_DAYS_PER_REQUEST

    print(f"         Using FIRMS REST API (MAP_KEY required)")
    print(f"         Source: {source}")
    print(f"         Days to query: {days}")
    if days > MAX_DAYS_PER_REQUEST:
        num_chunks = (days + MAX_DAYS_PER_REQUEST - 1) // MAX_DAYS_PER_REQUEST
        print(f"         Note: Will split into {num_chunks} API requests (limit: {MAX_DAYS_PER_REQUEST} days/request)")
    if confidence_filter:
        print(f"         Confidence filter: {confidence_filter}")

    config = FIRMSAPIConfig()
    client = FIRMSAPIClient(config)

    features = client.query(
        aoi,
        days=days,
        source=source,
        date=end_date,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(features)


def create_buffered_aoi_from_points(thermal_geojson, radius_meters):
    """
    Create a buffered AOI geometry from thermal detection points.

    Args:
        thermal_geojson: GeoJSON FeatureCollection with thermal detection points
        radius_meters: Buffer radius in meters around each point

    Returns:
        GeoJSON geometry (union of all buffered points) or None if no features
    """
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union, transform
    import pyproj

    features = thermal_geojson.get('features', [])
    if not features:
        return None

    # Collect all point geometries
    points = []
    for f in features:
        geom = shape(f['geometry'])
        points.append(geom)

    if not points:
        return None

    # Get centroid for projection
    all_points = unary_union(points)
    centroid = all_points.centroid

    # Create a local UTM projection for accurate buffering
    # Use a simple equidistant projection centered on the centroid
    proj_string = f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +units=m"

    project_to_local = pyproj.Transformer.from_crs(
        "EPSG:4326", proj_string, always_xy=True
    ).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(
        proj_string, "EPSG:4326", always_xy=True
    ).transform

    # Buffer each point and union
    buffered_points = []
    for point in points:
        # Project to local CRS, buffer, project back
        local_point = transform(project_to_local, point)
        buffered = local_point.buffer(radius_meters)
        wgs84_buffered = transform(project_to_wgs84, buffered)
        buffered_points.append(wgs84_buffered)

    # Union all buffered polygons
    unified = unary_union(buffered_points)

    return mapping(unified)


def get_bbox_from_geometry(geometry):
    """Extract bounding box from a GeoJSON geometry."""
    from shapely.geometry import shape
    geom = shape(geometry)
    return geom.bounds  # (west, south, east, north)


def run_pipeline_pre_export():
    """Run pipeline up to enrichment step (before export)."""
    from datetime import datetime, timedelta

    # Use configuration from top of file
    bbox = BBOX

    # Calculate date range
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    start_date_str = start_dt.strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("Pipeline Test - Pre-Export (FIRMS-First Approach)")
    print("=" * 60)
    print(f"Bounding box: {bbox}")
    print(f"Date range: {start_date_str} to {END_DATE} ({DAYS_BACK} days)")
    print(f"Thermal source: {THERMAL_SOURCE.upper()}")
    print(f"Proximity radius: {PROXIMITY_RADIUS_METERS}m")
    print("=" * 60)

    # Create output directory
    output_dir = Path("./output/test_pre_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create AOI from bounding box
    print("\n[Step 1] Loading AOI...")
    aoi = AOIHandler.from_bbox(
        west=bbox[0],
        south=bbox[1],
        east=bbox[2],
        north=bbox[3]
    )
    print(f"AOI bounds: {AOIHandler.get_bounds(aoi)}")

    # Step 2: Query thermal anomalies FIRST (primary data source)
    print(f"\n[Step 2] Querying thermal anomalies (primary data)...")
    print(f"         Date range: {start_date_str} to {END_DATE}")
    try:
        if THERMAL_SOURCE.lower() == "gibs":
            thermal = query_thermal_gibs(aoi, DAYS_BACK, VIIRS_SOURCE, END_DATE)
        elif THERMAL_SOURCE.lower() == "firms_api":
            thermal = query_thermal_firms_api(
                aoi, DAYS_BACK, VIIRS_SOURCE, END_DATE, CONFIDENCE_FILTER
            )
        else:
            raise ValueError(f"Unknown THERMAL_SOURCE: {THERMAL_SOURCE}")

        thermal_count = len(thermal.get('features', []))
        print(f"Found {thermal_count} thermal detections")

        if thermal_count > 0:
            save_geojson(thermal, output_dir / "thermal_detections.geojson")
            print(f"Saved to: {output_dir / 'thermal_detections.geojson'}")

            # Print sample detection properties
            if thermal['features']:
                sample = thermal['features'][0]['properties']
                print(f"         Sample detection properties: {list(sample.keys())}")

    except Exception as e:
        print(f"Thermal query failed: {e}")
        import traceback
        traceback.print_exc()
        thermal = {"type": "FeatureCollection", "features": []}

    # If no thermal detections, we can't proceed with the FIRMS-first approach
    thermal_count = len(thermal.get('features', []))
    if thermal_count == 0:
        print("\n" + "=" * 60)
        print("NO THERMAL DETECTIONS FOUND")
        print("=" * 60)
        print("Cannot proceed with FIRMS-first approach without thermal data.")
        print("Try adjusting the date range or bounding box.")
        return {
            "aoi": aoi,
            "infrastructure": {"type": "FeatureCollection", "features": []},
            "thermal": thermal,
            "conflicts": {"type": "FeatureCollection", "features": []},
            "enriched": {"type": "FeatureCollection", "features": []},
        }

    # Step 3: Create buffered AOI around thermal detections for OSM/ACLED queries
    print(f"\n[Step 3] Creating {PROXIMITY_RADIUS_METERS}m buffer around thermal detections...")
    buffered_geometry = create_buffered_aoi_from_points(thermal, PROXIMITY_RADIUS_METERS)

    if buffered_geometry:
        buffered_bbox = get_bbox_from_geometry(buffered_geometry)
        print(f"         Buffered AOI bbox: {buffered_bbox}")

        # Create AOI from buffered geometry for OSM/ACLED queries
        thermal_aoi = buffered_geometry
    else:
        print("         Failed to create buffered AOI, using original AOI")
        thermal_aoi = aoi

    # Step 4: Query water infrastructure from OSM (filtered to thermal detection areas)
    print("\n[Step 4] Querying water infrastructure from OSM (near thermal detections)...")
    try:
        infrastructure = query_water_infrastructure(thermal_aoi)
        infra_count = len(infrastructure.get('features', []))
        print(f"Found {infra_count} infrastructure features within {PROXIMITY_RADIUS_METERS}m of detections")

        if infra_count > 0:
            save_geojson(infrastructure, output_dir / "infrastructure.geojson")
            print(f"Saved to: {output_dir / 'infrastructure.geojson'}")
    except Exception as e:
        print(f"OSM query failed: {e}")
        import traceback
        traceback.print_exc()
        infrastructure = {"type": "FeatureCollection", "features": []}
        infra_count = 0

    # Step 5: Query conflict events from ACLED (filtered to thermal detection areas)
    print("\n[Step 5] Querying conflict events from ACLED (near thermal detections)...")
    print(f"         Date range: {start_date_str} to {END_DATE}")
    try:
        from water_pipeline.acled import ACLEDClient, DEFAULT_VIOLENCE_SUB_EVENTS
        acled_client = ACLEDClient()
        # Filter to violence/destruction sub-event types relevant to infrastructure damage
        df = acled_client.query(
            geometry=thermal_aoi,
            start_date=start_date_str,
            end_date=END_DATE,
            filter_violence=True  # Filters to: Attack, Shelling, Air/drone strike, etc.
        )
        conflicts = acled_client.to_geojson(df)
        conflict_count = len(conflicts.get('features', []))
        print(f"Found {conflict_count} violence/destruction events within {PROXIMITY_RADIUS_METERS}m of detections")
        print(f"         Filtered to: {', '.join(DEFAULT_VIOLENCE_SUB_EVENTS[:3])}...")

        if conflict_count > 0:
            save_geojson(conflicts, output_dir / "conflict_events.geojson")
            print(f"Saved to: {output_dir / 'conflict_events.geojson'}")
    except Exception as e:
        print(f"ACLED query failed: {e}")
        print("Hint: Set ACLED_EMAIL and ACLED_PASSWORD environment variables")
        print("      Or set ACLED_ACCESS_TOKEN if you have a pre-obtained token")
        print("Note: Access tokens are valid for 24 hours, refresh tokens for 14 days")
        conflicts = {"type": "FeatureCollection", "features": []}
        conflict_count = 0

    # Step 6: Enrich and classify detections
    print("\n[Step 6] Enriching thermal detections with OSM and ACLED data...")
    enriched = enrich_detections(thermal, conflicts, infrastructure)
    enriched_count = len(enriched.get('features', []))
    print(f"Enriched {enriched_count} detections")

    # Save enriched results
    save_geojson(enriched, output_dir / "enriched_detections.geojson")
    print(f"Saved to: {output_dir / 'enriched_detections.geojson'}")

    # Print classification summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    classifications = {}
    novel_count = 0
    near_infra_count = 0
    for f in enriched.get('features', []):
        props = f.get('properties', {})
        cls = props.get('classification', 'unclassified')
        classifications[cls] = classifications.get(cls, 0) + 1
        if props.get('is_novel', False):
            novel_count += 1
        if props.get('near_infrastructure', False):
            near_infra_count += 1

    for cls, count in sorted(classifications.items()):
        print(f"  {cls}: {count}")

    print(f"\nNear infrastructure: {near_infra_count}")
    print(f"Novel detections (no ACLED match): {novel_count}")

    # Summary stats
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY (Pre-Export)")
    print("=" * 60)
    print(f"Query period: {start_date_str} to {END_DATE} ({DAYS_BACK} days)")
    print(f"Bounding box: {BBOX}")
    print(f"Thermal source: {THERMAL_SOURCE.upper()}")
    print(f"Proximity radius: {PROXIMITY_RADIUS_METERS}m")
    print("-" * 40)
    print(f"Thermal detections:      {thermal_count}")
    print(f"Infrastructure features: {infra_count} (within {PROXIMITY_RADIUS_METERS}m)")
    print(f"Conflict events:         {conflict_count} (within {PROXIMITY_RADIUS_METERS}m)")
    print(f"Enriched detections:     {enriched_count}")
    print(f"\nOutput directory: {output_dir.absolute()}")

    print("\n" + "=" * 60)
    print("STOPPED BEFORE EXPORT TO PLANET/MAXAR")
    print("=" * 60)
    print("\nThe enriched_detections.geojson contains the pre-export data.")
    print("To proceed to export, use the export functions from water_pipeline.")

    return {
        "aoi": aoi,
        "thermal_aoi": thermal_aoi,
        "infrastructure": infrastructure,
        "thermal": thermal,
        "conflicts": conflicts,
        "enriched": enriched,
    }


if __name__ == "__main__":
    results = run_pipeline_pre_export()
