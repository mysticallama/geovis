"""
Planet Pipeline - Satellite Imagery Processing for ML Workflows

A modular pipeline for querying, downloading, and processing Planet Labs
satellite imagery with OSM integration for ML model training.

Modules:
    core - Planet API client, downloads, storage
    processing - Image preprocessing, spectral indices
    osm - OpenStreetMap Overpass queries
    ml - ML dataset preparation

Example:
    from planet_pipeline import PlanetPipeline, query_buildings

    # Planet imagery workflow
    pipeline = PlanetPipeline()
    pipeline.add_aoi("site1", geometry)
    scenes = pipeline.search("site1", "2024-01-01", "2024-03-31")
    pipeline.download("site1", scenes)

    # OSM labels
    buildings = query_buildings(geometry)
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# Core - Planet API, Downloads, Storage
# =============================================================================

from .core import (
    # Config
    PlanetConfig,
    # Classes
    Storage,
    PlanetClient,
    Downloader,
    PlanetPipeline,
)

# =============================================================================
# Processing - Indices, Preprocessing
# =============================================================================

from .processing import (
    # Band mappings
    PLANET_4B,
    PLANET_8B,
    # Index functions
    ndvi, ndwi, evi, savi, msavi, gndvi, arvi, vari, bai, gci, sipi,
    INDICES,
    # Processor class
    ImageProcessor,
    # Convenience functions
    list_indices,
    process_imagery,
)

# =============================================================================
# OSM - Overpass Queries
# =============================================================================

from .osm import (
    # Config
    OverpassConfig,
    Tag,
    # Classes
    OverpassClient,
    QueryBuilder,
    # Query functions
    query_features,
    query_buildings,
    query_water,
    query_landuse,
    query_roads,
    query_pois,
    # ML extraction
    extract_bboxes,
    extract_centroids,
    extract_points,
    create_pairs,
    generate_mask,
    # Export functions
    save_geojson,
    export_yolo,
    export_clustering,
    export_pairs,
)

# =============================================================================
# ML - Dataset Preparation
# =============================================================================

from .ml import (
    # Config
    DatasetConfig,
    # Functions
    normalize_chip,
    augment_chip,
    split_files,
    # Classes
    DatasetCreator,
    LabelManager,
    # Convenience functions
    prepare_dataset,
    prepare_detection_dataset,
    prepare_segmentation_dataset,
)

# =============================================================================
# Version
# =============================================================================

__version__ = "2.0.0"

__all__ = [
    # Core
    "PlanetConfig", "Storage", "PlanetClient", "Downloader", "PlanetPipeline",
    # Processing
    "PLANET_4B", "PLANET_8B", "INDICES", "ImageProcessor",
    "ndvi", "ndwi", "evi", "savi", "msavi", "gndvi", "arvi", "vari", "bai", "gci", "sipi",
    "list_indices", "process_imagery",
    # OSM
    "OverpassConfig", "Tag", "OverpassClient", "QueryBuilder",
    "query_features", "query_buildings", "query_water", "query_landuse", "query_roads", "query_pois",
    "extract_bboxes", "extract_centroids", "extract_points", "create_pairs", "generate_mask",
    "save_geojson", "export_yolo", "export_clustering", "export_pairs",
    # ML
    "DatasetConfig", "DatasetCreator", "LabelManager",
    "normalize_chip", "augment_chip", "split_files",
    "prepare_dataset", "prepare_detection_dataset", "prepare_segmentation_dataset",
]
