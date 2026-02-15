#!/usr/bin/env python3
"""
Query Template - Planet Pipeline

A configurable template for querying Planet imagery and OSM data.
Edit the CONFIGURATION section below to customize your queries.
"""

import json
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these values for your specific query
# ============================================================================

# Your Planet API key (or set via environment variable PL_API_KEY)
PLANET_API_KEY = None  # Set to your key, or leave None to use env var

# Output directory for all data
OUTPUT_DIR = "./data"

# Area of Interest (GeoJSON Polygon of Aden)
# Coordinates are in [longitude, latitude] order
AOI = {
    "type": "Polygon",
    "coordinates": [[
        [44.95, 12.75],  # Southwest corner
        [45.05, 12.75],  # Southeast corner
        [45.05, 12.85],  # Northeast corner
        [44.95, 12.85],  # Northwest corner
        [44.95, 12.75]   # Close the polygon (back to start)
    ]]
}

# Date range for Planet imagery search
START_DATE = "2024-01-01"
END_DATE = "2024-06-30"

# Planet search parameters
CLOUD_COVER_MAX = 0.1  # Max cloud cover (0.1 = 10%)
MAX_GSD = 5          # Max ground sample distance in meters
SCENE_LIMIT = 5       # Max number of scenes to return

# OSM feature types to query (set to False to skip)
QUERY_BUILDINGS = True
QUERY_WATER = True
QUERY_ROADS = False
QUERY_LANDUSE = False
QUERY_POIS = False

# POI amenity types (if QUERY_POIS is True)
POI_AMENITIES = ["restaurant", "cafe", "school", "hospital"]

# ML export options
EXPORT_YOLO = True
EXPORT_MASKS = True
EXPORT_CLUSTERING = False

# YOLO class mapping (building type -> class ID)
YOLO_CLASS_MAP = {
    "yes": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
}

# Mask class mapping
MASK_CLASS_MAP = {
    "yes": 1,
    "residential": 2,
    "commercial": 3,
}

# Image parameters (for YOLO/mask generation)
# These should match your actual imagery dimensions
IMAGE_SIZE = (1024, 1024)

# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_bounds_from_aoi(aoi):
    """Extract bounding box from AOI polygon."""
    coords = aoi["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def run_planet_query():
    """Query and optionally download Planet imagery."""
    from planet_pipeline import PlanetPipeline

    print("\n" + "=" * 60)
    print("PLANET IMAGERY QUERY")
    print("=" * 60)

    # Initialize pipeline
    pipeline = PlanetPipeline(
        api_key=PLANET_API_KEY,
        storage_dir=Path(OUTPUT_DIR) / "planet"
    )

    # Add AOI
    pipeline.add_aoi("query_site", AOI)
    print(f"\nAOI registered: query_site")

    # Search for imagery
    print(f"\nSearching Planet API...")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Cloud cover max: {CLOUD_COVER_MAX * 100}%")
    print(f"  Max GSD: {MAX_GSD}m")

    scenes = pipeline.search(
        aoi_name="query_site",
        start_date=START_DATE,
        end_date=END_DATE,
        cloud_cover_max=CLOUD_COVER_MAX,
        max_gsd=MAX_GSD,
        limit=SCENE_LIMIT
    )

    print(f"\nFound {len(scenes)} scenes")

    # Display results
    if scenes:
        print("\nTop scenes:")
        for i, scene in enumerate(scenes[:5]):
            props = scene.get("properties", {})
            print(f"  {i+1}. {scene['id']}")
            print(f"      Date: {props.get('acquired', 'N/A')[:10]}")
            print(f"      Cloud: {props.get('cloud_cover', 0)*100:.1f}%")
            print(f"      GSD: {props.get('gsd', 'N/A')}m")

        # Save scene list
        output_file = Path(OUTPUT_DIR) / "planet" / "search_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(scenes, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return scenes


def run_osm_queries():
    """Query OSM for various feature types."""
    from planet_pipeline import (
        query_buildings, query_water, query_roads,
        query_landuse, query_pois, save_geojson
    )

    print("\n" + "=" * 60)
    print("OSM DATA QUERIES")
    print("=" * 60)

    output_dir = Path(OUTPUT_DIR) / "osm"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if QUERY_BUILDINGS:
        print("\nQuerying buildings...")
        buildings = query_buildings(AOI)
        results["buildings"] = buildings
        save_geojson(buildings, output_dir / "buildings.geojson")
        print(f"  Found {len(buildings['features'])} buildings")

    if QUERY_WATER:
        print("\nQuerying water features...")
        water = query_water(AOI)
        results["water"] = water
        save_geojson(water, output_dir / "water.geojson")
        print(f"  Found {len(water['features'])} water features")

    if QUERY_ROADS:
        print("\nQuerying roads...")
        roads = query_roads(AOI)
        results["roads"] = roads
        save_geojson(roads, output_dir / "roads.geojson")
        print(f"  Found {len(roads['features'])} road segments")

    if QUERY_LANDUSE:
        print("\nQuerying land use...")
        landuse = query_landuse(AOI)
        results["landuse"] = landuse
        save_geojson(landuse, output_dir / "landuse.geojson")
        print(f"  Found {len(landuse['features'])} land use areas")

    if QUERY_POIS:
        print("\nQuerying POIs...")
        pois = query_pois(AOI, amenities=POI_AMENITIES)
        results["pois"] = pois
        save_geojson(pois, output_dir / "pois.geojson")
        print(f"  Found {len(pois['features'])} POIs")

    print(f"\nOSM data saved to: {output_dir}")
    return results


def run_ml_exports(osm_data):
    """Generate ML-ready exports from OSM data."""
    import numpy as np
    from planet_pipeline import export_yolo, generate_mask, export_clustering

    print("\n" + "=" * 60)
    print("ML EXPORTS")
    print("=" * 60)

    bounds = get_bounds_from_aoi(AOI)

    if EXPORT_YOLO and "buildings" in osm_data:
        print("\nGenerating YOLO annotations...")
        output_dir = Path(OUTPUT_DIR) / "yolo"
        output_dir.mkdir(parents=True, exist_ok=True)

        label_path = export_yolo(
            geojson=osm_data["buildings"],
            output_dir=output_dir,
            image_name="tile_001",
            image_bounds=bounds,
            image_size=IMAGE_SIZE,
            class_map=YOLO_CLASS_MAP
        )
        print(f"  YOLO labels saved to: {label_path}")

    if EXPORT_MASKS and "buildings" in osm_data:
        print("\nGenerating segmentation masks...")
        output_dir = Path(OUTPUT_DIR) / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        mask = generate_mask(
            geojson=osm_data["buildings"],
            bounds=bounds,
            size=IMAGE_SIZE,
            class_map=MASK_CLASS_MAP,
            class_key="building"
        )

        mask_path = output_dir / "building_mask.npy"
        np.save(mask_path, mask)
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique classes: {np.unique(mask)}")
        print(f"  Mask saved to: {mask_path}")

        # Save PNG if Pillow available
        try:
            from PIL import Image
            png_path = output_dir / "building_mask.png"
            img = Image.fromarray((mask * 85).astype(np.uint8))  # Scale for visibility
            img.save(png_path)
            print(f"  PNG preview saved to: {png_path}")
        except ImportError:
            pass

    if EXPORT_CLUSTERING and "pois" in osm_data:
        print("\nExporting clustering data...")
        output_dir = Path(OUTPUT_DIR) / "clustering"
        output_dir.mkdir(parents=True, exist_ok=True)

        export_clustering(osm_data["pois"], output_dir / "pois.npy", format="npy")
        export_clustering(osm_data["pois"], output_dir / "pois.csv", format="csv")
        print(f"  Clustering data saved to: {output_dir}")


def print_summary():
    """Print configuration summary."""
    bounds = get_bounds_from_aoi(AOI)

    print("\n" + "=" * 60)
    print("QUERY CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\nAOI Bounds:")
    print(f"  West:  {bounds[0]:.6f}")
    print(f"  South: {bounds[1]:.6f}")
    print(f"  East:  {bounds[2]:.6f}")
    print(f"  North: {bounds[3]:.6f}")
    print(f"\nPlanet Query:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Cloud cover max: {CLOUD_COVER_MAX * 100}%")
    print(f"\nOSM Queries:")
    print(f"  Buildings: {QUERY_BUILDINGS}")
    print(f"  Water: {QUERY_WATER}")
    print(f"  Roads: {QUERY_ROADS}")
    print(f"  Land use: {QUERY_LANDUSE}")
    print(f"  POIs: {QUERY_POIS}")
    print(f"\nML Exports:")
    print(f"  YOLO: {EXPORT_YOLO}")
    print(f"  Masks: {EXPORT_MASKS}")
    print(f"  Clustering: {EXPORT_CLUSTERING}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


def main():
    """Run the configured queries."""
    import sys

    print("\n" + "=" * 60)
    print("PLANET PIPELINE - QUERY TEMPLATE")
    print("=" * 60)

    # Print configuration
    print_summary()

    # Parse command line args
    run_planet = True
    run_osm = True

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "planet":
            run_osm = False
        elif arg == "osm":
            run_planet = False
        elif arg == "help":
            print("\nUsage:")
            print("  python query_template.py          # Run all queries")
            print("  python query_template.py planet   # Planet only")
            print("  python query_template.py osm      # OSM only")
            print("\nEdit the CONFIGURATION section at the top of this file")
            print("to customize your AOI, date range, and query parameters.")
            return

    # Confirm before running
    response = input("\nProceed with queries? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return

    # Run queries
    scenes = None
    osm_data = {}

    if run_planet:
        try:
            scenes = run_planet_query()
        except Exception as e:
            print(f"\nPlanet query error: {e}")
            print("(Make sure PL_API_KEY is set)")

    if run_osm:
        try:
            osm_data = run_osm_queries()
        except Exception as e:
            print(f"\nOSM query error: {e}")

    # ML exports
    if osm_data and (EXPORT_YOLO or EXPORT_MASKS or EXPORT_CLUSTERING):
        try:
            run_ml_exports(osm_data)
        except Exception as e:
            print(f"\nML export error: {e}")

    # Done
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nDirectory structure:")
    print("  data/")
    print("  ├── planet/       # Planet search results")
    print("  ├── osm/          # GeoJSON files")
    print("  ├── yolo/         # YOLO annotations")
    print("  ├── masks/        # Segmentation masks")
    print("  └── clustering/   # Clustering data")


if __name__ == "__main__":
    main()
