"""
Water & Food Infrastructure Monitoring Pipeline (No FIRMS)

A modular Python pipeline for monitoring water and food infrastructure using:
  - OpenStreetMap (Overpass API) — water + food infrastructure features
  - ACLED — conflict event data over a user-defined date window
  - Planet Labs — satellite imagery for a custom before/after date pair
  - xView / YOLOv8 — overhead object detection with change analysis
  - WFP Logistics Cluster (HDX) — humanitarian infrastructure shapefiles
  - Mapbox GL JS — interactive web app fusing ACLED and WFP data

Quick Start:
    from water_osm_pipeline import (
        AOIHandler, ACLEDClient, query_infrastructure,
        PlanetImageryClient, YOLOv8Detector, WFPLogisticsClient,
        generate_mapbox_app,
    )
"""

__version__ = "1.0.0"

from .aoi import AOIHandler, AOIConfig, load_aoi
from .osm import (
    OverpassClient, OverpassConfig, QueryBuilder, Tag,
    WATER_FOOD_TAGS, EXTENDED_WATER_FOOD_TAGS,
    query_infrastructure, save_geojson,
)
from .acled import (
    ACLEDClient, ACLEDConfig,
    ACLED_EVENT_TYPES, DEFAULT_VIOLENCE_SUB_EVENTS,
    query_conflict_events,
)
from .imagery import PlanetImageryClient, PlanetConfig, download_date_pair
from .yolov8 import YOLOv8Detector, YOLOv8Config, detect_and_annotate
from .wfp import WFPLogisticsClient, WFPConfig, fetch_logistics_data
from .fusion import generate_mapbox_app, MapboxFusionConfig

__all__ = [
    "__version__",
    # AOI
    "AOIHandler", "AOIConfig", "load_aoi",
    # OSM
    "OverpassClient", "OverpassConfig", "QueryBuilder", "Tag",
    "WATER_FOOD_TAGS", "EXTENDED_WATER_FOOD_TAGS",
    "query_infrastructure", "save_geojson",
    # ACLED
    "ACLEDClient", "ACLEDConfig",
    "ACLED_EVENT_TYPES", "DEFAULT_VIOLENCE_SUB_EVENTS",
    "query_conflict_events",
    # Imagery
    "PlanetImageryClient", "PlanetConfig", "download_date_pair",
    # YOLOv8
    "YOLOv8Detector", "YOLOv8Config", "detect_and_annotate",
    # WFP
    "WFPLogisticsClient", "WFPConfig", "fetch_logistics_data",
    # Fusion
    "generate_mapbox_app", "MapboxFusionConfig",
]
