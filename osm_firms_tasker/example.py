#!/usr/bin/env python3
"""
Water Infrastructure Monitoring Pipeline - Example Usage

This script demonstrates how to use the pipeline to:
1. Load an Area of Interest (AOI)
2. Query water infrastructure from OpenStreetMap
3. Query thermal anomalies from NASA FIRMS
4. Query conflict events from ACLED
5. Filter and enrich detections
6. Export for satellite APIs and ML models

Before running, set your API keys:
    export FIRMS_MAP_KEY='your-32-char-key'
    export ACLED_EMAIL='your@email.com'
    export ACLED_PASSWORD='your-acled-password'

Get your keys at:
    - FIRMS: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    - ACLED: https://acleddata.com/register/

Note: ACLED uses OAuth authentication. Access tokens are valid for 24 hours.
      The client will automatically handle token refresh using your credentials.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from water_pipeline import (
    run_water_infrastructure_analysis,
    IntegratedPipeline,
    PipelineConfig,
    AOIHandler,
    query_water_infrastructure,
    query_thermal_anomalies,
    query_conflict_events,
    filter_by_proximity,
    enrich_detections,
    export_for_ml,
)


def example_quick_analysis():
    """
    Quick analysis using the high-level function.

    This is the simplest way to run the pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 1: Quick Analysis")
    print("=" * 60)

    # Define AOI as bounding box [west, south, east, north]
    # This example uses a region in Yemen (Aden area)
    aoi_bbox = [44.9, 12.7, 45.1, 12.9]

    # Run complete analysis
    results = run_water_infrastructure_analysis(
        aoi=aoi_bbox,
        output_dir="./output/quick_analysis",
        thermal_lookback_days=7,
        proximity_radius_meters=50,
    )

    print(f"\nResults saved to: ./output/quick_analysis")
    return results


def example_step_by_step():
    """
    Step-by-step analysis with more control.

    This shows how to use individual components.
    """
    print("\n" + "=" * 60)
    print("Example 2: Step-by-Step Analysis")
    print("=" * 60)

    # Step 1: Create AOI from coordinates
    aoi = AOIHandler.from_bbox(
        west=44.9,
        south=12.7,
        east=45.1,
        north=12.9
    )
    print(f"AOI bounds: {AOIHandler.get_bounds(aoi)}")

    # Step 2: Query water infrastructure
    print("\nQuerying water infrastructure...")
    infrastructure = query_water_infrastructure(aoi)
    print(f"Found {len(infrastructure.get('features', []))} infrastructure features")

    # Step 3: Query thermal anomalies (requires FIRMS_MAP_KEY)
    print("\nQuerying thermal anomalies...")
    try:
        thermal = query_thermal_anomalies(aoi, days=7)
        print(f"Found {len(thermal.get('features', []))} thermal detections")
    except Exception as e:
        print(f"FIRMS query failed (set FIRMS_MAP_KEY): {e}")
        thermal = {"type": "FeatureCollection", "features": []}

    # Step 4: Query conflict events (requires ACLED credentials)
    print("\nQuerying conflict events...")
    try:
        conflicts = query_conflict_events(aoi, days=7)
        print(f"Found {len(conflicts.get('features', []))} conflict events")
    except Exception as e:
        print(f"ACLED query failed (set ACLED_EMAIL and ACLED_PASSWORD): {e}")
        conflicts = {"type": "FeatureCollection", "features": []}

    # Step 5: Filter detections near infrastructure
    if thermal.get("features") and infrastructure.get("features"):
        print("\nFiltering detections near infrastructure...")
        filtered = filter_by_proximity(thermal, infrastructure, radius_meters=50)
        print(f"Filtered to {len(filtered.get('features', []))} detections")
    else:
        filtered = thermal

    # Step 6: Enrich with ACLED data
    print("\nEnriching detections...")
    enriched = enrich_detections(filtered, conflicts, infrastructure)
    print(f"Enriched {len(enriched.get('features', []))} detections")

    # Step 7: Export for ML
    print("\nExporting for ML...")
    exports = export_for_ml(
        enriched,
        output_dir="./output/step_by_step/ml",
        formats=["geojson", "yolo", "clustering"],
    )
    print(f"Exports: {list(exports.keys())}")

    return enriched


def example_custom_config():
    """
    Analysis with custom configuration.

    This shows how to customize the pipeline behavior.
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    # Create custom config
    config = PipelineConfig(
        thermal_lookback_days=14,  # Extended lookback
        acled_lookback_days=14,
        proximity_radius_meters=100,  # Larger radius
        firms_sources=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],  # Multiple sources
        export_formats=["geojson", "yolo", "clustering"],
        output_dir="./output/custom_config",
    )

    # Create pipeline with config
    pipeline = IntegratedPipeline(config)

    # Load AOI from GeoJSON dict
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [44.9, 12.7],
            [45.1, 12.7],
            [45.1, 12.9],
            [44.9, 12.9],
            [44.9, 12.7]
        ]]
    }

    # Run pipeline
    results = pipeline.run(aoi, aoi_name="custom_analysis")

    return results


def example_osm_only():
    """
    Query only OSM water infrastructure (no API keys needed).

    This is useful for testing or when you only need infrastructure data.
    """
    print("\n" + "=" * 60)
    print("Example 4: OSM-Only Query (No API Keys Required)")
    print("=" * 60)

    from water_pipeline import Tag

    # Define AOI
    aoi = AOIHandler.from_bbox(
        west=-122.5,
        south=37.7,
        east=-122.3,
        north=37.9
    )

    # Query with default water tags
    print("\nQuerying with default water tags...")
    infrastructure = query_water_infrastructure(aoi)
    print(f"Found {len(infrastructure.get('features', []))} features")

    # Print feature types
    types = {}
    for f in infrastructure.get("features", []):
        props = f.get("properties", {})
        ftype = (
            props.get("water") or
            props.get("waterway") or
            props.get("man_made") or
            "other"
        )
        types[ftype] = types.get(ftype, 0) + 1

    print("\nFeature types:")
    for ftype, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count}")

    # Save to file
    from water_pipeline import save_geojson
    output_path = "./output/osm_only/infrastructure.geojson"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_geojson(infrastructure, output_path)
    print(f"\nSaved to: {output_path}")

    return infrastructure


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# Water Infrastructure Monitoring Pipeline - Examples")
    print("#" * 60)

    # Example 4 requires no API keys - good for testing
    example_osm_only()

    # The following examples require API keys:
    # Uncomment after setting FIRMS_MAP_KEY, ACLED_EMAIL, ACLED_API_KEY

    # example_quick_analysis()
    # example_step_by_step()
    # example_custom_config()

    print("\n" + "#" * 60)
    print("# Examples complete!")
    print("# Check ./output/ for results")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
