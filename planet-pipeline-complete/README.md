# Planet Pipeline

A modular Python pipeline for querying Planet Labs satellite imagery and OpenStreetMap data, with integrated ML export workflows for YOLOv8, Siamese networks, DBSCAN/HDBSCAN, and semantic segmentation.

## Features

- **Planet Labs Integration**: Query, download, and process satellite imagery
- **OSM Overpass Queries**: Fetch buildings, roads, water, land use, POIs
- **Spectral Indices**: NDVI, NDWI, EVI, SAVI, and 7 more
- **ML Export Formats**: YOLO annotations, clustering data, Siamese pairs, segmentation masks
- **Dataset Preparation**: Train/val/test splits, chipping, normalization, augmentation

## Installation

```bash
# Clone and install
git clone <your-repo-url>
cd planet-pipeline
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# Set Planet API key
export PL_API_KEY='your-planet-api-key'
```

## Quick Start

### Planet Imagery

```python
from planet_pipeline import PlanetPipeline

# Initialize
pipeline = PlanetPipeline(storage_dir="./data")

# Add AOI
pipeline.add_aoi("site1", {
    "type": "Polygon",
    "coordinates": [[[-122.4, 37.7], [-122.4, 37.8], [-122.3, 37.8], [-122.3, 37.7], [-122.4, 37.7]]]
})

# Search and download
scenes = pipeline.search("site1", "2024-01-01", "2024-03-31", cloud_cover_max=0.1)
pipeline.download("site1", scenes, limit=10)
```

### OSM Data

```python
from planet_pipeline import query_buildings, query_water, export_yolo, generate_mask

# Query buildings
aoi = {"type": "Polygon", "coordinates": [[...]]}
buildings = query_buildings(aoi)

# Export YOLO annotations
export_yolo(buildings, "./labels", "tile_001",
            image_bounds=(-122.4, 37.7, -122.3, 37.8),
            image_size=(1024, 1024),
            class_map={"residential": 0, "commercial": 1})

# Or generate segmentation mask
mask = generate_mask(buildings, bounds=(-122.4, 37.7, -122.3, 37.8), size=(512, 512))
```

### Spectral Indices

```python
from planet_pipeline import ImageProcessor, list_indices

# See available indices
print(list_indices())  # ['ndvi', 'ndwi', 'evi', 'savi', ...]

# Calculate indices
ImageProcessor.calculate_index("imagery.tif", "ndvi", "ndvi.tif")
ImageProcessor.calculate_indices("imagery.tif", ["ndvi", "ndwi", "evi"], "./indices")
```

### ML Dataset

```python
from planet_pipeline import prepare_dataset, prepare_detection_dataset

# Create training chips
prepare_dataset(
    imagery_files=list(Path("./imagery").glob("*.tif")),
    output_dir="./dataset",
    chip_size=256,
    normalize=True,
    augment=True
)

# Or YOLOv8 detection dataset
prepare_detection_dataset("./imagery", "./labels", "./yolo_dataset")
```

## Module Structure

```
planet_pipeline/
├── __init__.py      # Public API
├── core.py          # Planet API client, downloads, storage
├── processing.py    # Spectral indices, preprocessing
├── osm.py           # OSM Overpass queries, ML exports
└── ml.py            # Dataset preparation
```

## API Reference

### Core (Planet)

| Function | Description |
|----------|-------------|
| `PlanetPipeline(storage_dir)` | Main pipeline interface |
| `pipeline.add_aoi(name, geometry)` | Register an AOI |
| `pipeline.search(aoi, start, end)` | Search for imagery |
| `pipeline.download(aoi, scenes)` | Download scenes |
| `PlanetClient` | Low-level API client |
| `Storage` | File organization |

### Processing

| Function | Description |
|----------|-------------|
| `ImageProcessor.calculate_index(path, index)` | Calculate spectral index |
| `ImageProcessor.normalize(data)` | Normalize imagery |
| `ImageProcessor.clip_to_geometry(path, geom)` | Clip to AOI |
| `list_indices()` | List available indices |

### OSM

| Function | Description |
|----------|-------------|
| `query_buildings(geometry)` | Fetch building footprints |
| `query_water(geometry)` | Fetch water features |
| `query_landuse(geometry)` | Fetch land use polygons |
| `query_roads(geometry)` | Fetch road networks |
| `query_pois(geometry)` | Fetch points of interest |
| `query_features(geometry, tags)` | Custom tag query |
| `export_yolo(geojson, ...)` | Export YOLO annotations |
| `export_clustering(geojson, path)` | Export for DBSCAN |
| `export_pairs(geojson, path)` | Export Siamese pairs |
| `generate_mask(geojson, bounds, size)` | Create segmentation mask |

### ML

| Function | Description |
|----------|-------------|
| `prepare_dataset(files, output_dir)` | Create training chips |
| `prepare_detection_dataset(img, labels, out)` | YOLOv8 dataset |
| `prepare_segmentation_dataset(img, masks, out)` | Segmentation dataset |
| `DatasetCreator` | Fine-grained control |
| `normalize_chip(chip)` | Normalize array |

## Spectral Indices

| Index | Description | Use Case |
|-------|-------------|----------|
| NDVI | Normalized Difference Vegetation | Vegetation health |
| NDWI | Normalized Difference Water | Water bodies |
| EVI | Enhanced Vegetation | Dense vegetation |
| SAVI | Soil-Adjusted Vegetation | Sparse vegetation |
| MSAVI | Modified SAVI | Mixed land cover |
| GNDVI | Green NDVI | Chlorophyll content |
| ARVI | Atmospherically Resistant VI | Hazy conditions |
| VARI | Visible Atmospherically Resistant | RGB-only analysis |
| BAI | Burned Area Index | Fire damage |
| GCI | Green Chlorophyll Index | Crop health |
| SIPI | Structure Insensitive Pigment | Leaf pigments |

## Examples

Run the interactive examples:

```bash
python examples.py        # Interactive menu
python examples.py 4      # Run specific example
python examples.py osm    # Run OSM examples (4-9)
python examples.py all    # Run all examples
```

Available examples:
1. Planet Search
2. Planet Download
3. Spectral Indices
4. OSM Buildings
5. OSM Custom Query
6. YOLO Export
7. Clustering Export (DBSCAN)
8. Siamese Pairs
9. Segmentation Masks
10. ML Dataset Prep
11. Integrated Workflow

## Output Structure

```
data/
├── planet/
│   ├── aois/{name}/geometry.geojson
│   ├── imagery/{aoi}/{date}/{item_id}/
│   ├── processed/
│   └── indices/
├── osm/
│   └── buildings.geojson
├── yolo/
│   └── tile_001.txt
├── clustering/
│   └── pois.npy
└── dataset/
    ├── train/
    ├── val/
    ├── test/
    └── pytorch_loader.py
```

## Requirements

- Python 3.8+
- requests, numpy, rasterio, shapely, tqdm

Optional:
- torch (PyTorch loaders)
- tensorflow (TF loaders)
- scikit-learn (clustering)
- Pillow (mask visualization)

## License

MIT License

## Links

- [Planet API Documentation](https://developers.planet.com/docs/apis/)
- [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
