"""
Water Infrastructure Monitoring Pipeline

A modular Python pipeline for monitoring water infrastructure using satellite data,
OpenStreetMap, NASA FIRMS thermal detection, and ACLED conflict event data.

Quick Start:
    from water_pipeline import run_water_infrastructure_analysis

    # Run analysis on a region
    results = run_water_infrastructure_analysis(
        aoi="my_region.geojson",
        output_dir="./results"
    )

    # Or from bounding box coordinates
    results = run_water_infrastructure_analysis(
        aoi=[-122.5, 37.7, -122.3, 37.9],  # [west, south, east, north]
        output_dir="./results"
    )

    # Access results
    print(f"Infrastructure: {len(results['infrastructure']['features'])}")
    print(f"Thermal detections: {len(results['thermal_detections']['features'])}")
    print(f"Novel detections: {results['summary']['novel_count']}")

Modules:
    - aoi: AOI input handling (coordinates, GeoJSON, shapefiles)
    - firms: NASA GIBS WMTS for thermal anomalies (NRT, no API key)
    - firms_api: NASA FIRMS REST API (requires MAP_KEY, longer archives)
    - acled: ACLED conflict event API
    - osm: OpenStreetMap Overpass API for water infrastructure
    - spatial: Proximity analysis and spatial operations
    - enrichment: Metadata enrichment and classification
    - export: Export for satellite APIs and ML models
    - pipeline: Integrated workflow orchestration

Environment Variables:
    FIRMS_MAP_KEY: NASA FIRMS API key (required for thermal data)
    ACLED_EMAIL: ACLED registered email (required for conflict data)
    ACLED_API_KEY: ACLED API key (required for conflict data)
    PL_API_KEY: Planet Labs API key (optional, for Planet exports)
    MAXAR_API_KEY: Maxar API key (optional, for Maxar Discovery exports)
"""

__version__ = "1.0.0"

# AOI handling
from .aoi import (
    AOIHandler,
    AOIConfig,
    load_aoi,
)

# NASA FIRMS thermal detection (GIBS WMTS - NRT only, no API key)
from .firms import (
    FIRMSClient,
    FIRMSConfig,
    FIRMS_SOURCES,
    query_thermal_anomalies,
    query_thermal_anomalies_multi,
)

# NASA FIRMS REST API (requires MAP_KEY, supports longer date ranges)
from .firms_api import (
    FIRMSAPIClient,
    FIRMSAPIConfig,
    FIRMS_SOURCES as FIRMS_API_SOURCES,
    query_firms_thermal_anomalies,
    query_firms_multi_source,
)

# ACLED conflict events
from .acled import (
    ACLEDClient,
    ACLEDConfig,
    ACLED_EVENT_TYPES,
    DEFAULT_VIOLENCE_SUB_EVENTS,
    query_conflict_events,
    query_violence_events,
)

# OpenStreetMap Overpass API
from .osm import (
    OverpassClient,
    OverpassConfig,
    QueryBuilder,
    Tag,
    DEFAULT_WATER_TAGS,
    EXTENDED_WATER_TAGS,
    query_water_infrastructure,
    query_features,
    query_buildings,
    query_roads,
    query_landuse,
    save_geojson,
)

# Spatial analysis
from .spatial import (
    SpatialAnalyzer,
    SpatialConfig,
    filter_by_proximity,
    enrich_with_proximity,
)

# Metadata enrichment
from .enrichment import (
    MetadataEnricher,
    EnrichmentConfig,
    enrich_detections,
    classify_and_summarize,
    CLASSIFICATION_CONFIRMED,
    CLASSIFICATION_POTENTIAL,
    CLASSIFICATION_KNOWN,
    CLASSIFICATION_UNCLASSIFIED,
)

# Export for satellite APIs and ML
from .export import (
    APIExporter,
    MLExporter,
    ExportConfig,
    export_for_satellite_api,
    export_for_ml,
)

# Integrated pipeline
from .pipeline import (
    IntegratedPipeline,
    PipelineConfig,
    run_water_infrastructure_analysis,
)

__all__ = [
    # Version
    "__version__",

    # AOI
    "AOIHandler",
    "AOIConfig",
    "load_aoi",

    # FIRMS (GIBS WMTS)
    "FIRMSClient",
    "FIRMSConfig",
    "FIRMS_SOURCES",
    "query_thermal_anomalies",
    "query_thermal_anomalies_multi",

    # FIRMS REST API
    "FIRMSAPIClient",
    "FIRMSAPIConfig",
    "FIRMS_API_SOURCES",
    "query_firms_thermal_anomalies",
    "query_firms_multi_source",

    # ACLED
    "ACLEDClient",
    "ACLEDConfig",
    "ACLED_EVENT_TYPES",
    "query_conflict_events",
    "query_violence_events",

    # OSM
    "OverpassClient",
    "OverpassConfig",
    "QueryBuilder",
    "Tag",
    "DEFAULT_WATER_TAGS",
    "EXTENDED_WATER_TAGS",
    "query_water_infrastructure",
    "query_features",
    "query_buildings",
    "query_roads",
    "query_landuse",
    "save_geojson",

    # Spatial
    "SpatialAnalyzer",
    "SpatialConfig",
    "filter_by_proximity",
    "enrich_with_proximity",

    # Enrichment
    "MetadataEnricher",
    "EnrichmentConfig",
    "enrich_detections",
    "classify_and_summarize",
    "CLASSIFICATION_CONFIRMED",
    "CLASSIFICATION_POTENTIAL",
    "CLASSIFICATION_KNOWN",
    "CLASSIFICATION_UNCLASSIFIED",

    # Export
    "APIExporter",
    "MLExporter",
    "ExportConfig",
    "export_for_satellite_api",
    "export_for_ml",

    # Pipeline
    "IntegratedPipeline",
    "PipelineConfig",
    "run_water_infrastructure_analysis",
]
