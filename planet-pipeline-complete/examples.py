#!/usr/bin/env python3
"""
Planet Pipeline Examples

Demonstrates Planet imagery and OSM integration for ML workflows.
"""

import json
from pathlib import Path


def example_planet_search():
    """
    Example 1: Search Planet API for imagery.

    Query Planet Labs for satellite scenes matching AOI, date range, and quality criteria.
    """
    print("=" * 60)
    print("Example 1: Planet API Search")
    print("=" * 60)

    from planet_pipeline import PlanetPipeline

    # Initialize pipeline
    pipeline = PlanetPipeline(storage_dir="./data/planet")

    # Define AOI (San Francisco)
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.45, 37.75],
            [-122.45, 37.80],
            [-122.40, 37.80],
            [-122.40, 37.75],
            [-122.45, 37.75]
        ]]
    }

    # Add AOI
    pipeline.add_aoi("sf_downtown", aoi, metadata={"city": "San Francisco"})

    # Search for imagery
    print("\nSearching Planet API...")
    scenes = pipeline.search(
        aoi_name="sf_downtown",
        start_date="2024-01-01",
        end_date="2024-03-31",
        cloud_cover_max=0.1,
        max_gsd=3.5,
        limit=10
    )

    print(f"Found {len(scenes)} scenes")

    # Show sample scene info
    if scenes:
        scene = scenes[0]
        props = scene.get("properties", {})
        print(f"\nSample scene:")
        print(f"  ID: {scene['id']}")
        print(f"  Date: {props.get('acquired', 'N/A')[:10]}")
        print(f"  Cloud cover: {props.get('cloud_cover', 0)*100:.1f}%")
        print(f"  GSD: {props.get('gsd', 'N/A')}m")

    print("\nExample 1 complete!")
    return scenes


def example_planet_download():
    """
    Example 2: Download Planet imagery.

    Download activated scenes and organize by AOI/date.
    """
    print("\n" + "=" * 60)
    print("Example 2: Download Imagery")
    print("=" * 60)

    from planet_pipeline import PlanetPipeline

    pipeline = PlanetPipeline(storage_dir="./data/planet")

    # Add AOI
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.42, 37.78],
            [-122.42, 37.79],
            [-122.41, 37.79],
            [-122.41, 37.78],
            [-122.42, 37.78]
        ]]
    }
    pipeline.add_aoi("test_site", aoi)

    # Search
    scenes = pipeline.search("test_site", "2024-01-01", "2024-01-31", limit=2)

    if scenes:
        print(f"\nDownloading {len(scenes)} scenes...")
        # Download (this will activate assets and download)
        downloaded = pipeline.download(
            aoi_name="test_site",
            scenes=scenes,
            asset_types=["ortho_analytic_4b"]
        )
        print(f"Downloaded {len(downloaded)} files")
    else:
        print("No scenes found to download")

    print("\nExample 2 complete!")


def example_spectral_indices():
    """
    Example 3: Calculate spectral indices.

    Compute NDVI, NDWI, and other indices from imagery.
    """
    print("\n" + "=" * 60)
    print("Example 3: Spectral Indices")
    print("=" * 60)

    from planet_pipeline import ImageProcessor, list_indices

    # List available indices
    print("\nAvailable indices:")
    for idx in list_indices():
        print(f"  - {idx}")

    # Example usage with actual imagery
    print("""
Example code:
    from planet_pipeline import ImageProcessor

    # Calculate single index
    ndvi_path = ImageProcessor.calculate_index(
        "imagery.tif",
        "ndvi",
        output_path="ndvi.tif"
    )

    # Calculate multiple indices
    results = ImageProcessor.calculate_indices(
        "imagery.tif",
        ["ndvi", "ndwi", "evi"],
        output_dir="./indices"
    )
""")

    print("\nExample 3 complete!")


def example_osm_buildings():
    """
    Example 4: Query OSM buildings.

    Fetch building footprints for an AOI using Overpass API.
    """
    print("\n" + "=" * 60)
    print("Example 4: OSM Building Query")
    print("=" * 60)

    from planet_pipeline import query_buildings, save_geojson

    # Define AOI
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.42, 37.78],
            [-122.42, 37.79],
            [-122.41, 37.79],
            [-122.41, 37.78],
            [-122.42, 37.78]
        ]]
    }

    print("\nQuerying OSM for buildings...")
    buildings = query_buildings(aoi)

    print(f"Found {len(buildings['features'])} buildings")

    # Save to file
    output_dir = Path("./data/osm")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_geojson(buildings, output_dir / "buildings.geojson")

    # Show sample
    if buildings["features"]:
        sample = buildings["features"][0]
        print(f"\nSample building:")
        print(f"  OSM ID: {sample['properties'].get('@id')}")
        print(f"  Type: {sample['properties'].get('building', 'yes')}")

    print("\nExample 4 complete!")
    return buildings


def example_osm_custom():
    """
    Example 5: Custom OSM queries.

    Query multiple feature types with custom tags.
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom OSM Queries")
    print("=" * 60)

    from planet_pipeline import query_features, Tag, save_geojson

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.45, 37.76],
            [-122.45, 37.78],
            [-122.43, 37.78],
            [-122.43, 37.76],
            [-122.45, 37.76]
        ]]
    }

    print("\nQuerying custom features...")

    # Query parks and water
    features = query_features(
        geometry=aoi,
        tags=[
            Tag("leisure", ["park", "garden"]),
            Tag("natural", "water"),
            Tag("landuse", "forest"),
        ]
    )

    print(f"Found {len(features['features'])} features")

    # Count by type
    by_type = {}
    for f in features["features"]:
        props = f["properties"]
        ftype = props.get("leisure") or props.get("natural") or props.get("landuse") or "other"
        by_type[ftype] = by_type.get(ftype, 0) + 1

    print("\nBy type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")

    print("\nExample 5 complete!")
    return features


def example_yolo_export():
    """
    Example 6: Export YOLO annotations.

    Generate YOLOv8 format bounding box labels from OSM data.
    """
    print("\n" + "=" * 60)
    print("Example 6: YOLO Annotation Export")
    print("=" * 60)

    from planet_pipeline import query_buildings, export_yolo, extract_bboxes

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.42, 37.78],
            [-122.42, 37.79],
            [-122.41, 37.79],
            [-122.41, 37.78],
            [-122.42, 37.78]
        ]]
    }

    print("\nQuerying buildings for YOLO export...")
    buildings = query_buildings(aoi)

    # Define class mapping
    class_map = {
        "yes": 0,
        "residential": 1,
        "commercial": 2,
        "industrial": 3,
    }

    # Image parameters (would come from actual imagery)
    image_bounds = (-122.42, 37.78, -122.41, 37.79)
    image_size = (1024, 1024)

    # Export YOLO format
    output_dir = Path("./data/yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = export_yolo(
        geojson=buildings,
        output_dir=output_dir,
        image_name="tile_001",
        image_bounds=image_bounds,
        image_size=image_size,
        class_map=class_map
    )

    print(f"Labels saved to: {label_path}")

    # Show sample
    bboxes = extract_bboxes(buildings, image_bounds, image_size, class_map)
    if bboxes:
        bb = bboxes[0]
        print(f"\nSample bbox (YOLO format):")
        if "bbox_yolo" in bb:
            x, y, w, h = bb["bbox_yolo"]
            print(f"  class={bb['class_id']} x={x:.4f} y={y:.4f} w={w:.4f} h={h:.4f}")

    print("\nExample 6 complete!")


def example_clustering_export():
    """
    Example 7: Export for DBSCAN/HDBSCAN clustering.

    Extract point data for spatial clustering analysis.
    """
    print("\n" + "=" * 60)
    print("Example 7: Clustering Data Export")
    print("=" * 60)

    from planet_pipeline import query_pois, export_clustering, extract_centroids

    # Larger area for meaningful clustering
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.45, 37.75],
            [-122.45, 37.80],
            [-122.40, 37.80],
            [-122.40, 37.75],
            [-122.45, 37.75]
        ]]
    }

    print("\nQuerying POIs for clustering...")
    pois = query_pois(aoi, amenities=["restaurant", "cafe", "bar"])

    print(f"Found {len(pois['features'])} POIs")

    # Export for clustering
    output_dir = Path("./data/clustering")
    output_dir.mkdir(parents=True, exist_ok=True)

    export_clustering(pois, output_dir / "pois.npy", format="npy")
    export_clustering(pois, output_dir / "pois.csv", format="csv")

    print(f"\nExported to {output_dir}")

    # Show example clustering code
    print("""
Example DBSCAN usage:
    import numpy as np
    from sklearn.cluster import DBSCAN

    points = np.load("./data/clustering/pois.npy")
    # eps ~0.001 degree = 100m
    clustering = DBSCAN(eps=0.002, min_samples=3).fit(points)
    labels = clustering.labels_
""")

    print("\nExample 7 complete!")


def example_siamese_pairs():
    """
    Example 8: Generate Siamese network training pairs.

    Create positive/negative pairs for contrastive learning.
    """
    print("\n" + "=" * 60)
    print("Example 8: Siamese Network Pairs")
    print("=" * 60)

    from planet_pipeline import query_landuse, export_pairs, create_pairs

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.45, 37.76],
            [-122.45, 37.80],
            [-122.40, 37.80],
            [-122.40, 37.76],
            [-122.45, 37.76]
        ]]
    }

    print("\nQuerying land use for Siamese pairs...")
    landuse = query_landuse(
        aoi,
        types=["residential", "commercial", "industrial", "park"]
    )

    print(f"Found {len(landuse['features'])} land use features")

    # Create pairs
    pairs = create_pairs(landuse, same_ratio=0.5, class_key="landuse")

    print(f"\nGenerated {len(pairs)} pairs")
    same = sum(1 for p in pairs if p["same_class"])
    print(f"  Same class: {same}")
    print(f"  Different class: {len(pairs) - same}")

    # Export
    output_dir = Path("./data/siamese")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_pairs(landuse, output_dir / "pairs.json")

    print("\nExample 8 complete!")


def example_segmentation_mask():
    """
    Example 9: Generate segmentation masks.

    Create raster masks from OSM polygons for semantic segmentation.
    """
    print("\n" + "=" * 60)
    print("Example 9: Segmentation Masks")
    print("=" * 60)

    from planet_pipeline import query_buildings, generate_mask
    import numpy as np

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-122.42, 37.78],
            [-122.42, 37.79],
            [-122.41, 37.79],
            [-122.41, 37.78],
            [-122.42, 37.78]
        ]]
    }

    print("\nQuerying buildings for mask generation...")
    buildings = query_buildings(aoi)

    # Generate mask
    bounds = (-122.42, 37.78, -122.41, 37.79)
    size = (512, 512)

    class_map = {
        "yes": 1,
        "residential": 2,
        "commercial": 3,
    }

    print("\nGenerating raster mask...")
    mask = generate_mask(
        geojson=buildings,
        bounds=bounds,
        size=size,
        class_map=class_map,
        class_key="building"
    )

    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")

    coverage = (mask > 0).sum() / mask.size * 100
    print(f"Building coverage: {coverage:.1f}%")

    # Save
    output_dir = Path("./data/masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "building_mask.npy", mask)

    print(f"\nMask saved to {output_dir}")
    print("\nExample 9 complete!")


def example_ml_dataset():
    """
    Example 10: Prepare ML dataset.

    Create training chips from imagery with train/val/test splits.
    """
    print("\n" + "=" * 60)
    print("Example 10: ML Dataset Preparation")
    print("=" * 60)

    from planet_pipeline import DatasetCreator, DatasetConfig

    print("""
Example code for preparing ML dataset:

    from planet_pipeline import DatasetCreator, DatasetConfig, prepare_dataset

    # Simple approach
    prepare_dataset(
        imagery_files=list(Path("./imagery").glob("*.tif")),
        output_dir="./dataset",
        chip_size=256,
        normalize=True,
        augment=True
    )

    # Or with more control
    config = DatasetConfig(
        chip_size=256,
        overlap=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        normalize=True,
        augment=True
    )

    creator = DatasetCreator(config)
    creator.create_chips(imagery_files, "./dataset")

Output structure:
    dataset/
    ├── train/
    │   ├── chip_000000.npy
    │   └── ...
    ├── val/
    ├── test/
    ├── dataset_info.json
    ├── pytorch_loader.py
    └── tensorflow_loader.py
""")

    print("\nExample 10 complete!")


def example_integrated_workflow():
    """
    Example 11: Full integrated workflow.

    Complete pipeline from imagery search to ML-ready dataset.
    """
    print("\n" + "=" * 60)
    print("Example 11: Integrated Workflow")
    print("=" * 60)

    print("""
Complete workflow example:

    from planet_pipeline import (
        PlanetPipeline,
        ImageProcessor,
        query_buildings,
        export_yolo,
        generate_mask,
        prepare_detection_dataset
    )
    from pathlib import Path

    # 1. Initialize pipeline
    pipeline = PlanetPipeline(storage_dir="./project")

    # 2. Add AOI
    aoi = {...}  # GeoJSON polygon
    pipeline.add_aoi("site1", aoi)

    # 3. Search and download
    scenes = pipeline.search("site1", "2024-01-01", "2024-06-30")
    pipeline.download("site1", scenes, limit=50)

    # 4. Process imagery
    for img in pipeline.get_imagery("site1"):
        ImageProcessor.calculate_indices(img, ["ndvi", "ndwi"], "./indices")

    # 5. Get OSM labels
    buildings = query_buildings(aoi)

    # 6. Generate labels for each image
    for img in pipeline.get_imagery("site1"):
        with rasterio.open(img) as src:
            bounds = src.bounds
            size = (src.width, src.height)

        # YOLO format
        export_yolo(buildings, "./labels", img.stem, bounds, size, class_map)

        # Or segmentation mask
        mask = generate_mask(buildings, bounds, size, class_map)
        np.save(f"./masks/{img.stem}.npy", mask)

    # 7. Create ML dataset
    prepare_detection_dataset("./imagery", "./labels", "./dataset")

    print("Pipeline complete!")
""")

    print("\nExample 11 complete!")


def main():
    """Run examples."""
    import sys

    examples = {
        "1": ("Planet Search", example_planet_search),
        "2": ("Planet Download", example_planet_download),
        "3": ("Spectral Indices", example_spectral_indices),
        "4": ("OSM Buildings", example_osm_buildings),
        "5": ("OSM Custom Query", example_osm_custom),
        "6": ("YOLO Export", example_yolo_export),
        "7": ("Clustering Export", example_clustering_export),
        "8": ("Siamese Pairs", example_siamese_pairs),
        "9": ("Segmentation Mask", example_segmentation_mask),
        "10": ("ML Dataset", example_ml_dataset),
        "11": ("Integrated Workflow", example_integrated_workflow),
    }

    print("\n" + "=" * 60)
    print("Planet Pipeline Examples")
    print("=" * 60)

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  all. Run all examples")
    print("  osm. Run OSM examples only (4-9)")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        return

    if choice.lower() == 'all':
        for key, (_, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\nError in example {key}: {e}")

    elif choice.lower() == 'osm':
        for key in ["4", "5", "6", "7", "8", "9"]:
            try:
                examples[key][1]()
            except Exception as e:
                print(f"\nError in example {key}: {e}")

    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")

    print("\n" + "=" * 60)
    print("Check ./data/ for outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
