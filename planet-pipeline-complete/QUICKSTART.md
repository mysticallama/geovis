# Quick Start Guide

## 1. Setup

```bash
# Install
pip install -e .

# Set your Planet API key
export PL_API_KEY='your-api-key-here'
```

## 2. Planet Imagery Workflow

```python
from planet_pipeline import PlanetPipeline

# Initialize
pipeline = PlanetPipeline(storage_dir="./data")

# Define your area of interest
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

# Add AOI and search
pipeline.add_aoi("my_site", aoi)
scenes = pipeline.search("my_site", "2024-01-01", "2024-06-30", cloud_cover_max=0.1)

# Download (requires valid Planet subscription)
pipeline.download("my_site", scenes, limit=5)
```

## 3. OSM Data (No API Key Required)

```python
from planet_pipeline import query_buildings, save_geojson

# Query buildings in your AOI
buildings = query_buildings(aoi)
print(f"Found {len(buildings['features'])} buildings")

# Save to file
save_geojson(buildings, "./buildings.geojson")
```

## 4. Generate ML Labels

```python
from planet_pipeline import export_yolo, generate_mask

# For object detection (YOLOv8)
export_yolo(
    buildings,
    output_dir="./labels",
    image_name="tile_001",
    image_bounds=(-122.45, 37.75, -122.40, 37.80),
    image_size=(1024, 1024),
    class_map={"yes": 0, "residential": 1, "commercial": 2}
)

# For segmentation
mask = generate_mask(
    buildings,
    bounds=(-122.45, 37.75, -122.40, 37.80),
    size=(512, 512)
)
```

## 5. Run Examples

```bash
# Interactive menu
python examples.py

# Run OSM examples (no Planet API needed)
python examples.py osm

# Run specific example
python examples.py 4
```

## Key Imports

```python
from planet_pipeline import (
    # Planet
    PlanetPipeline,

    # OSM queries
    query_buildings,
    query_water,
    query_landuse,
    query_roads,

    # ML exports
    export_yolo,
    export_clustering,
    generate_mask,

    # Processing
    ImageProcessor,
    list_indices,

    # Dataset prep
    prepare_dataset,
)
```

## Need Help?

- See full documentation: [README.md](README.md)
- Run examples: `python examples.py`
- Planet API docs: https://developers.planet.com/docs/apis/
