"""
Integrated pipeline for water infrastructure monitoring.

Orchestrates the complete workflow:
1. Load AOI (coordinates, GeoJSON, shapefile)
2. Query OSM for water infrastructure
3. Query FIRMS for thermal anomalies
4. Filter thermal detections near infrastructure (50m radius)
5. Query ACLED for conflict events
6. Overlay ACLED with filtered detections
7. Enrich detections with ACLED data or mark as "novel"
8. Export for satellite APIs and ML models
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from .aoi import AOIHandler, load_aoi
from .osm import Tag, query_water_infrastructure, query_features, DEFAULT_WATER_TAGS
from .firms import FIRMSClient, FIRMSConfig, query_thermal_anomalies
from .acled import ACLEDClient, ACLEDConfig, query_conflict_events
from .spatial import SpatialAnalyzer, filter_by_proximity, enrich_with_proximity
from .enrichment import (
    MetadataEnricher,
    EnrichmentConfig,
    enrich_detections,
    classify_and_summarize,
)
from .export import (
    APIExporter,
    MLExporter,
    export_for_satellite_api,
    export_for_ml,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Pipeline configuration.

    Can be loaded from YAML file or set programmatically.
    """
    # API keys (can use environment variables)
    planet_api_key: Optional[str] = None
    firms_api_key: Optional[str] = None
    acled_email: Optional[str] = None
    acled_api_key: Optional[str] = None

    # Processing parameters
    thermal_lookback_days: int = 7
    acled_lookback_days: int = 7
    proximity_radius_meters: float = 50.0
    acled_match_radius_meters: float = 500.0

    # FIRMS settings
    firms_sources: List[str] = field(default_factory=lambda: ["VIIRS_SNPP_NRT"])
    firms_confidence_filter: Optional[str] = None  # low, nominal, high

    # OSM water infrastructure tags
    water_tags: List[Dict] = field(default_factory=lambda: DEFAULT_WATER_TAGS)

    # Output settings
    output_dir: Union[str, Path] = "./pipeline_output"
    export_formats: List[str] = field(default_factory=lambda: ["geojson", "yolo"])
    satellite_apis: List[str] = field(default_factory=lambda: ["planet"])

    def __post_init__(self):
        """Load API keys from environment if not set."""
        if self.planet_api_key is None:
            self.planet_api_key = os.environ.get("PL_API_KEY")
        if self.firms_api_key is None:
            self.firms_api_key = os.environ.get("FIRMS_MAP_KEY")
        if self.acled_email is None:
            self.acled_email = os.environ.get("ACLED_EMAIL")
        if self.acled_api_key is None:
            self.acled_api_key = os.environ.get("ACLED_API_KEY")

        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        config_data = {}

        if "api_keys" in data:
            config_data["planet_api_key"] = data["api_keys"].get("planet")
            config_data["firms_api_key"] = data["api_keys"].get("firms")
            config_data["acled_email"] = data["api_keys"].get("acled_email")
            config_data["acled_api_key"] = data["api_keys"].get("acled_key")

        if "temporal" in data:
            config_data["thermal_lookback_days"] = data["temporal"].get("thermal_lookback_days", 7)
            config_data["acled_lookback_days"] = data["temporal"].get("acled_lookback_days", 7)

        if "spatial" in data:
            config_data["proximity_radius_meters"] = data["spatial"].get("proximity_radius_meters", 50.0)
            config_data["acled_match_radius_meters"] = data["spatial"].get("acled_match_radius_meters", 500.0)

        if "firms" in data:
            config_data["firms_sources"] = data["firms"].get("sources", ["VIIRS_SNPP_NRT"])
            config_data["firms_confidence_filter"] = data["firms"].get("confidence_threshold")

        if "water_infrastructure" in data:
            config_data["water_tags"] = data["water_infrastructure"]

        if "pipeline" in data:
            config_data["output_dir"] = data["pipeline"].get("output_dir", "./pipeline_output")

        if "export" in data:
            config_data["export_formats"] = data["export"].get("ml_formats", ["geojson", "yolo"])
            config_data["satellite_apis"] = data["export"].get("satellite_apis", ["planet"])

        return cls(**config_data)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        data = {
            "pipeline": {
                "output_dir": str(self.output_dir),
            },
            "api_keys": {
                "planet": self.planet_api_key,
                "firms": self.firms_api_key,
                "acled_email": self.acled_email,
                "acled_key": self.acled_api_key,
            },
            "temporal": {
                "thermal_lookback_days": self.thermal_lookback_days,
                "acled_lookback_days": self.acled_lookback_days,
            },
            "spatial": {
                "proximity_radius_meters": self.proximity_radius_meters,
                "acled_match_radius_meters": self.acled_match_radius_meters,
            },
            "firms": {
                "sources": self.firms_sources,
                "confidence_threshold": self.firms_confidence_filter,
            },
            "water_infrastructure": self.water_tags,
            "export": {
                "satellite_apis": self.satellite_apis,
                "ml_formats": self.export_formats,
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class IntegratedPipeline:
    """
    Complete integrated pipeline for water infrastructure monitoring.

    Workflow:
        1. Load AOI (coordinates, GeoJSON, shapefile)
        2. Query OSM for water infrastructure
        3. Query FIRMS for thermal anomalies
        4. Filter thermal detections near infrastructure (50m radius)
        5. Query ACLED for conflict events
        6. Overlay ACLED with filtered detections
        7. Enrich detections with ACLED data or mark as "novel"
        8. Export for satellite APIs and ML models

    Example:
        pipeline = IntegratedPipeline()
        results = pipeline.run("my_region.geojson", aoi_name="study_area")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.aois: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}

        # Initialize API clients
        self.firms_client = FIRMSClient(FIRMSConfig(api_key=self.config.firms_api_key))
        self.acled_client = ACLEDClient(ACLEDConfig(
            email=self.config.acled_email,
            api_key=self.config.acled_api_key,
        ))
        self.enricher = MetadataEnricher(EnrichmentConfig(
            acled_match_radius_meters=self.config.acled_match_radius_meters,
            infrastructure_radius_meters=self.config.proximity_radius_meters,
        ))

    def load_aoi(
        self,
        source: Union[str, Path, Dict, List],
        name: str = "aoi",
    ) -> Dict:
        """
        Load AOI from any supported format.

        Args:
            source: AOI source (file path, GeoJSON dict, coordinates)
            name: Name identifier for this AOI

        Returns:
            GeoJSON geometry dict
        """
        geometry = load_aoi(source)
        self.aois[name] = geometry
        logger.info(f"Loaded AOI '{name}' with bounds {AOIHandler.get_bounds(geometry)}")
        return geometry

    def query_infrastructure(
        self,
        aoi_name: str,
        custom_tags: Optional[List[Union[Tag, Dict]]] = None,
    ) -> Dict:
        """
        Query OSM for water infrastructure within AOI.

        Args:
            aoi_name: Name of previously loaded AOI
            custom_tags: Custom OSM tag filters (default: config.water_tags)

        Returns:
            GeoJSON FeatureCollection with infrastructure features
        """
        if aoi_name not in self.aois:
            raise ValueError(f"AOI '{aoi_name}' not loaded. Call load_aoi() first.")

        geometry = self.aois[aoi_name]
        tags = custom_tags or self.config.water_tags

        logger.info(f"Querying water infrastructure for '{aoi_name}'...")

        result = query_water_infrastructure(geometry, tags=tags)

        logger.info(f"Found {len(result.get('features', []))} infrastructure features")
        return result

    def query_thermal_detections(
        self,
        aoi_name: str,
        days: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict:
        """
        Query FIRMS for thermal anomalies within AOI.

        Args:
            aoi_name: Name of previously loaded AOI
            days: Number of days to query (default: config.thermal_lookback_days)
            sources: FIRMS sources to query

        Returns:
            GeoJSON FeatureCollection with thermal detection points
        """
        if aoi_name not in self.aois:
            raise ValueError(f"AOI '{aoi_name}' not loaded. Call load_aoi() first.")

        geometry = self.aois[aoi_name]
        days = days or self.config.thermal_lookback_days
        sources = sources or self.config.firms_sources

        logger.info(f"Querying thermal detections for '{aoi_name}' ({days} days)...")

        if len(sources) > 1:
            df = self.firms_client.query_multiple_sources(
                geometry,
                days=days,
                sources=sources,
                confidence_filter=self.config.firms_confidence_filter,
            )
        else:
            df = self.firms_client.query(
                geometry,
                days=days,
                source=sources[0],
                confidence_filter=self.config.firms_confidence_filter,
            )

        result = self.firms_client.to_geojson(df)

        logger.info(f"Found {len(result.get('features', []))} thermal detections")
        return result

    def filter_near_infrastructure(
        self,
        detections: Dict,
        infrastructure: Dict,
        radius_meters: Optional[float] = None,
    ) -> Dict:
        """
        Filter detections within radius of infrastructure.

        Args:
            detections: GeoJSON of detection points
            infrastructure: GeoJSON of infrastructure features
            radius_meters: Proximity threshold (default: config.proximity_radius_meters)

        Returns:
            Filtered GeoJSON with only detections near infrastructure
        """
        radius = radius_meters or self.config.proximity_radius_meters

        logger.info(f"Filtering detections within {radius}m of infrastructure...")

        result = filter_by_proximity(detections, infrastructure, radius_meters=radius)

        logger.info(f"Filtered to {len(result.get('features', []))} detections near infrastructure")
        return result

    def query_conflict_events(
        self,
        aoi_name: str,
        days: Optional[int] = None,
        event_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Query ACLED for conflict events within AOI.

        Args:
            aoi_name: Name of previously loaded AOI
            days: Number of days to query (default: config.acled_lookback_days)
            event_types: Filter by event types

        Returns:
            GeoJSON FeatureCollection with conflict event points
        """
        if aoi_name not in self.aois:
            raise ValueError(f"AOI '{aoi_name}' not loaded. Call load_aoi() first.")

        geometry = self.aois[aoi_name]
        days = days or self.config.acled_lookback_days

        logger.info(f"Querying conflict events for '{aoi_name}' ({days} days)...")

        df = self.acled_client.query_within_aoi(
            geometry,
            days=days,
            event_types=event_types,
        )

        result = self.acled_client.to_geojson(df)

        logger.info(f"Found {len(result.get('features', []))} conflict events")
        return result

    def enrich_detections(
        self,
        filtered_detections: Dict,
        acled_events: Dict,
        infrastructure: Dict,
    ) -> Dict:
        """
        Enrich detections with ACLED and infrastructure metadata.

        Adds ACLED match data or marks as "novel", classifies by proximity.

        Args:
            filtered_detections: GeoJSON of filtered detection points
            acled_events: GeoJSON of ACLED events
            infrastructure: GeoJSON of infrastructure features

        Returns:
            Enriched and classified GeoJSON
        """
        logger.info("Enriching detections with ACLED and infrastructure data...")

        result = enrich_detections(
            filtered_detections,
            acled_events=acled_events,
            infrastructure=infrastructure,
            config=EnrichmentConfig(
                acled_match_radius_meters=self.config.acled_match_radius_meters,
                infrastructure_radius_meters=self.config.proximity_radius_meters,
            ),
        )

        return result

    def export_results(
        self,
        enriched_data: Dict,
        aoi_name: str,
        formats: Optional[List[str]] = None,
        satellite_apis: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Export results in specified formats.

        Args:
            enriched_data: Enriched and classified GeoJSON
            aoi_name: AOI name for output organization
            formats: ML export formats
            satellite_apis: Satellite APIs to export for

        Returns:
            Dict mapping format to file path
        """
        formats = formats or self.config.export_formats
        satellite_apis = satellite_apis or self.config.satellite_apis

        output_dir = self.config.output_dir / aoi_name
        output_dir.mkdir(parents=True, exist_ok=True)

        all_exports = {}

        # Export for satellite APIs
        for api in satellite_apis:
            try:
                exports = export_for_satellite_api(
                    enriched_data,
                    output_dir / "satellite",
                    api=api,
                )
                all_exports.update(exports)
            except Exception as e:
                logger.warning(f"Failed to export for {api}: {e}")

        # Get image bounds from AOI
        geometry = self.aois.get(aoi_name)
        image_bounds = None
        if geometry:
            image_bounds = AOIHandler.get_bounds(geometry)

        # Export for ML
        try:
            ml_exports = export_for_ml(
                enriched_data,
                output_dir / "ml",
                formats=formats,
                image_bounds=image_bounds,
            )
            all_exports.update(ml_exports)
        except Exception as e:
            logger.warning(f"Failed to export ML formats: {e}")

        logger.info(f"Exported results to {output_dir}")
        return all_exports

    def run(
        self,
        aoi_source: Union[str, Path, Dict, List],
        aoi_name: str = "analysis",
    ) -> Dict:
        """
        Run complete pipeline.

        Args:
            aoi_source: AOI source (file path, GeoJSON dict, coordinates)
            aoi_name: Name identifier for this analysis

        Returns:
            Dict with:
                - infrastructure: GeoJSON
                - thermal_detections: GeoJSON (all)
                - filtered_detections: GeoJSON (near infrastructure)
                - acled_events: GeoJSON
                - enriched_detections: GeoJSON (final output)
                - exports: Dict[format, Path]
                - summary: Dict with statistics
        """
        logger.info(f"Starting pipeline for '{aoi_name}'...")

        # Step 1: Load AOI
        self.load_aoi(aoi_source, aoi_name)

        # Step 2: Query infrastructure
        try:
            infrastructure = self.query_infrastructure(aoi_name)
        except Exception as e:
            logger.error(f"Failed to query infrastructure: {e}")
            infrastructure = {"type": "FeatureCollection", "features": []}

        # Step 3: Query thermal detections
        try:
            thermal_detections = self.query_thermal_detections(aoi_name)
        except Exception as e:
            logger.warning(f"Failed to query FIRMS (check API key): {e}")
            thermal_detections = {"type": "FeatureCollection", "features": []}

        # Step 4: Filter near infrastructure
        if infrastructure.get("features") and thermal_detections.get("features"):
            filtered_detections = self.filter_near_infrastructure(
                thermal_detections, infrastructure
            )
        else:
            filtered_detections = thermal_detections

        # Step 5: Query ACLED events
        try:
            acled_events = self.query_conflict_events(aoi_name)
        except Exception as e:
            logger.warning(f"Failed to query ACLED (check credentials): {e}")
            acled_events = {"type": "FeatureCollection", "features": []}

        # Step 6-7: Enrich and classify
        enriched_detections = self.enrich_detections(
            filtered_detections, acled_events, infrastructure
        )

        # Generate summary
        _, summary = classify_and_summarize(enriched_detections)

        # Step 8: Export
        exports = self.export_results(enriched_detections, aoi_name)

        # Store and return results
        results = {
            "aoi_name": aoi_name,
            "aoi_geometry": self.aois[aoi_name],
            "infrastructure": infrastructure,
            "thermal_detections": thermal_detections,
            "filtered_detections": filtered_detections,
            "acled_events": acled_events,
            "enriched_detections": enriched_detections,
            "exports": exports,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self.results[aoi_name] = results

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict) -> None:
        """Print pipeline summary."""
        summary = results["summary"]

        print("\n" + "=" * 60)
        print(f"Pipeline Results: {results['aoi_name']}")
        print("=" * 60)
        print(f"Infrastructure features: {len(results['infrastructure'].get('features', []))}")
        print(f"Thermal detections: {len(results['thermal_detections'].get('features', []))}")
        print(f"Near infrastructure: {len(results['filtered_detections'].get('features', []))}")
        print(f"ACLED events: {len(results['acled_events'].get('features', []))}")
        print("-" * 60)
        print("Classifications:")
        for cls, count in summary.get("classifications", {}).items():
            print(f"  {cls}: {count}")
        print("-" * 60)
        print(f"Novel detections: {summary.get('novel_count', 0)}")
        print(f"ACLED matched: {summary.get('acled_matched_count', 0)}")
        print(f"Output directory: {self.config.output_dir / results['aoi_name']}")
        print("=" * 60 + "\n")


def run_water_infrastructure_analysis(
    aoi: Union[str, Path, Dict, List],
    output_dir: Union[str, Path] = "./output",
    config_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict:
    """
    High-level function to run complete water infrastructure analysis.

    Args:
        aoi: AOI source (file path, GeoJSON dict, coordinates, bbox)
        output_dir: Output directory for results
        config_file: Optional YAML config file path
        **kwargs: Override config parameters:
            - thermal_lookback_days: Days to query FIRMS
            - acled_lookback_days: Days to query ACLED
            - proximity_radius_meters: Infrastructure proximity threshold
            - export_formats: ML export formats

    Returns:
        Dict with all pipeline results and summary

    Example:
        # From GeoJSON file
        results = run_water_infrastructure_analysis(
            aoi="my_region.geojson",
            output_dir="./results"
        )

        # From bounding box
        results = run_water_infrastructure_analysis(
            aoi=[-122.5, 37.7, -122.3, 37.9],  # [west, south, east, north]
            output_dir="./results",
            thermal_lookback_days=14
        )

        # Access results
        print(f"Novel detections: {results['summary']['novel_count']}")
    """
    # Load config
    if config_file:
        config = PipelineConfig.from_yaml(config_file)
    else:
        config = PipelineConfig()

    # Apply overrides
    config.output_dir = Path(output_dir)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Run pipeline
    pipeline = IntegratedPipeline(config)

    # Determine AOI name
    if isinstance(aoi, (str, Path)) and Path(aoi).exists():
        aoi_name = Path(aoi).stem
    else:
        aoi_name = "analysis"

    return pipeline.run(aoi, aoi_name=aoi_name)
