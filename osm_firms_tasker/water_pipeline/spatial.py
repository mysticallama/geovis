"""
Spatial analysis module for proximity filtering and spatial operations.

Provides tools for:
- Proximity filtering (find features within distance of reference features)
- Spatial joins and overlays
- Distance calculations
- Buffering and clipping
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


@dataclass
class SpatialConfig:
    """Spatial analysis configuration."""
    default_buffer_meters: float = 50.0
    default_crs: str = "EPSG:4326"
    use_precise_distance: bool = True  # Use geodesic distance when possible


class SpatialAnalyzer:
    """
    Spatial analysis operations for GeoJSON data.

    Provides proximity filtering, spatial joins, and distance calculations.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()

    @staticmethod
    def _meters_to_degrees(meters: float, latitude: float = 0.0) -> float:
        """
        Convert meters to approximate degrees.

        More accurate when latitude is provided (accounts for longitude scaling).

        Args:
            meters: Distance in meters
            latitude: Reference latitude for longitude scaling

        Returns:
            Approximate distance in degrees
        """
        # 1 degree of latitude ~ 111,320 meters
        lat_factor = 111320

        # Longitude varies by latitude
        lon_factor = 111320 * math.cos(math.radians(latitude))

        # Use average for general buffer
        avg_factor = (lat_factor + lon_factor) / 2

        return meters / avg_factor

    @staticmethod
    def _get_centroid(geometry: Dict) -> Tuple[float, float]:
        """Get centroid of geometry as (lon, lat)."""
        geom = shape(geometry)
        centroid = geom.centroid
        return (centroid.x, centroid.y)

    @staticmethod
    def _extract_geometries(geojson: Dict) -> List[Any]:
        """Extract Shapely geometries from GeoJSON."""
        if geojson.get("type") == "FeatureCollection":
            return [shape(f["geometry"]) for f in geojson.get("features", [])
                    if f.get("geometry")]
        elif geojson.get("type") == "Feature":
            return [shape(geojson["geometry"])] if geojson.get("geometry") else []
        else:
            return [shape(geojson)]

    @staticmethod
    def proximity_filter(
        source_geojson: Dict,
        target_geojson: Dict,
        distance_meters: float = 50.0,
        return_distances: bool = False,
    ) -> Dict:
        """
        Filter source features within distance of any target feature.

        Args:
            source_geojson: Features to filter (e.g., thermal detections)
            target_geojson: Reference features (e.g., water infrastructure)
            distance_meters: Maximum distance threshold in meters
            return_distances: Include distance to nearest target in properties

        Returns:
            Filtered GeoJSON with only features within distance threshold
        """
        source_features = source_geojson.get("features", [])
        target_features = target_geojson.get("features", [])

        if not source_features or not target_features:
            return {"type": "FeatureCollection", "features": []}

        # Extract target geometries and build spatial index
        target_geoms = [shape(f["geometry"]) for f in target_features if f.get("geometry")]
        if not target_geoms:
            return {"type": "FeatureCollection", "features": []}

        target_union = unary_union(target_geoms)

        # Get reference latitude for degree conversion
        bounds = target_union.bounds
        ref_lat = (bounds[1] + bounds[3]) / 2
        distance_degrees = SpatialAnalyzer._meters_to_degrees(distance_meters, ref_lat)

        # Buffer targets for quick containment check
        buffered_targets = target_union.buffer(distance_degrees)

        filtered_features = []

        for feature in source_features:
            if not feature.get("geometry"):
                continue

            source_geom = shape(feature["geometry"])

            # Quick check: is source within buffered area?
            if not buffered_targets.intersects(source_geom):
                continue

            # Precise distance calculation
            min_distance = source_geom.distance(target_union)

            if min_distance <= distance_degrees:
                # Convert distance back to meters for property
                distance_m = min_distance * 111320 * math.cos(math.radians(ref_lat))

                # Create copy with proximity metadata
                filtered_feature = {
                    "type": "Feature",
                    "geometry": feature["geometry"],
                    "properties": dict(feature.get("properties", {}))
                }

                if return_distances:
                    filtered_feature["properties"]["distance_to_infrastructure_m"] = round(distance_m, 2)
                    filtered_feature["properties"]["within_radius"] = True

                filtered_features.append(filtered_feature)

        logger.info(f"Proximity filter: {len(filtered_features)}/{len(source_features)} "
                   f"features within {distance_meters}m")

        return {"type": "FeatureCollection", "features": filtered_features}

    @staticmethod
    def spatial_join(
        left_geojson: Dict,
        right_geojson: Dict,
        distance_meters: Optional[float] = None,
        join_type: str = "inner",
    ) -> Dict:
        """
        Spatial join between two GeoJSON datasets.

        Enriches left features with properties from nearest right feature.

        Args:
            left_geojson: Primary features
            right_geojson: Features to join from
            distance_meters: Max distance for join (None = no limit)
            join_type: "inner" (only matched) or "left" (all left features)

        Returns:
            GeoJSON with joined properties
        """
        left_features = left_geojson.get("features", [])
        right_features = right_geojson.get("features", [])

        if not left_features:
            return {"type": "FeatureCollection", "features": []}

        if not right_features:
            if join_type == "left":
                return left_geojson
            return {"type": "FeatureCollection", "features": []}

        # Build spatial index for right features
        right_geoms = [shape(f["geometry"]) for f in right_features if f.get("geometry")]
        right_tree = STRtree(right_geoms)

        # Get reference latitude
        all_bounds = [g.bounds for g in right_geoms]
        ref_lat = sum(b[1] + b[3] for b in all_bounds) / (2 * len(all_bounds))

        distance_degrees = None
        if distance_meters:
            distance_degrees = SpatialAnalyzer._meters_to_degrees(distance_meters, ref_lat)

        joined_features = []

        for left_feature in left_features:
            if not left_feature.get("geometry"):
                if join_type == "left":
                    joined_features.append(left_feature)
                continue

            left_geom = shape(left_feature["geometry"])

            # Find nearest right feature
            nearest_idx = right_tree.nearest(left_geom)
            nearest_geom = right_geoms[nearest_idx]
            nearest_feature = right_features[nearest_idx]

            distance = left_geom.distance(nearest_geom)
            distance_m = distance * 111320 * math.cos(math.radians(ref_lat))

            # Check distance threshold
            if distance_degrees and distance > distance_degrees:
                if join_type == "left":
                    joined_features.append(left_feature)
                continue

            # Create joined feature
            joined_props = dict(left_feature.get("properties", {}))

            # Add right feature properties with prefix
            for key, value in nearest_feature.get("properties", {}).items():
                joined_props[f"joined_{key}"] = value

            joined_props["join_distance_m"] = round(distance_m, 2)

            joined_features.append({
                "type": "Feature",
                "geometry": left_feature["geometry"],
                "properties": joined_props
            })

        return {"type": "FeatureCollection", "features": joined_features}

    @staticmethod
    def buffer_features(
        geojson: Dict,
        distance_meters: float,
        cap_style: str = "round",
    ) -> Dict:
        """
        Buffer all features by specified distance.

        Args:
            geojson: Input GeoJSON
            distance_meters: Buffer distance in meters
            cap_style: Buffer cap style: "round", "flat", "square"

        Returns:
            GeoJSON with buffered geometries
        """
        features = geojson.get("features", [])
        if not features:
            return {"type": "FeatureCollection", "features": []}

        cap_styles = {"round": 1, "flat": 2, "square": 3}
        cap = cap_styles.get(cap_style, 1)

        buffered_features = []

        for feature in features:
            if not feature.get("geometry"):
                continue

            geom = shape(feature["geometry"])
            centroid = geom.centroid
            distance_degrees = SpatialAnalyzer._meters_to_degrees(
                distance_meters, centroid.y
            )

            buffered_geom = geom.buffer(distance_degrees, cap_style=cap)

            buffered_features.append({
                "type": "Feature",
                "geometry": mapping(buffered_geom),
                "properties": dict(feature.get("properties", {}))
            })

        return {"type": "FeatureCollection", "features": buffered_features}

    @staticmethod
    def calculate_distances(
        source_geojson: Dict,
        target_geojson: Dict,
    ) -> List[Dict]:
        """
        Calculate distance from each source feature to nearest target.

        Args:
            source_geojson: Source features
            target_geojson: Target features

        Returns:
            List of dicts with source_idx, target_idx, distance_meters
        """
        source_features = source_geojson.get("features", [])
        target_features = target_geojson.get("features", [])

        if not source_features or not target_features:
            return []

        target_geoms = [shape(f["geometry"]) for f in target_features if f.get("geometry")]
        target_tree = STRtree(target_geoms)

        # Get reference latitude
        all_bounds = [g.bounds for g in target_geoms]
        ref_lat = sum(b[1] + b[3] for b in all_bounds) / (2 * len(all_bounds))

        results = []

        for source_idx, source_feature in enumerate(source_features):
            if not source_feature.get("geometry"):
                continue

            source_geom = shape(source_feature["geometry"])
            nearest_idx = target_tree.nearest(source_geom)
            nearest_geom = target_geoms[nearest_idx]

            distance_deg = source_geom.distance(nearest_geom)
            distance_m = distance_deg * 111320 * math.cos(math.radians(ref_lat))

            results.append({
                "source_idx": source_idx,
                "target_idx": nearest_idx,
                "distance_meters": round(distance_m, 2),
                "source_id": source_feature.get("properties", {}).get("osm_id"),
                "target_id": target_features[nearest_idx].get("properties", {}).get("osm_id"),
            })

        return results

    @staticmethod
    def clip_to_aoi(
        geojson: Dict,
        aoi_geometry: Dict,
    ) -> Dict:
        """
        Clip features to AOI boundary.

        Args:
            geojson: Input features
            aoi_geometry: AOI geometry to clip to

        Returns:
            Clipped GeoJSON
        """
        features = geojson.get("features", [])
        if not features:
            return {"type": "FeatureCollection", "features": []}

        aoi = shape(aoi_geometry)
        clipped_features = []

        for feature in features:
            if not feature.get("geometry"):
                continue

            geom = shape(feature["geometry"])

            if not aoi.intersects(geom):
                continue

            clipped_geom = aoi.intersection(geom)

            if clipped_geom.is_empty:
                continue

            clipped_features.append({
                "type": "Feature",
                "geometry": mapping(clipped_geom),
                "properties": dict(feature.get("properties", {}))
            })

        return {"type": "FeatureCollection", "features": clipped_features}

    @staticmethod
    def extract_centroids(geojson: Dict) -> Dict:
        """
        Extract centroids from all features.

        Args:
            geojson: Input GeoJSON

        Returns:
            GeoJSON with Point geometries at centroids
        """
        features = geojson.get("features", [])
        centroid_features = []

        for feature in features:
            if not feature.get("geometry"):
                continue

            geom = shape(feature["geometry"])
            centroid = geom.centroid

            centroid_features.append({
                "type": "Feature",
                "geometry": mapping(centroid),
                "properties": dict(feature.get("properties", {}))
            })

        return {"type": "FeatureCollection", "features": centroid_features}

    @staticmethod
    def union_features(geojson: Dict) -> Dict:
        """
        Union all features into a single geometry.

        Args:
            geojson: Input GeoJSON

        Returns:
            GeoJSON Feature with unified geometry
        """
        features = geojson.get("features", [])
        if not features:
            return {"type": "Feature", "geometry": None, "properties": {}}

        geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
        if not geoms:
            return {"type": "Feature", "geometry": None, "properties": {}}

        unified = unary_union(geoms)

        return {
            "type": "Feature",
            "geometry": mapping(unified),
            "properties": {"feature_count": len(geoms)}
        }


def filter_by_proximity(
    detections: Dict,
    infrastructure: Dict,
    radius_meters: float = 50.0,
) -> Dict:
    """
    High-level: Filter detections within radius of infrastructure.

    Args:
        detections: GeoJSON of detection points (thermal, etc.)
        infrastructure: GeoJSON of infrastructure features
        radius_meters: Proximity threshold (default 50m)

    Returns:
        Filtered GeoJSON with proximity metadata
    """
    analyzer = SpatialAnalyzer()
    return analyzer.proximity_filter(
        detections,
        infrastructure,
        distance_meters=radius_meters,
        return_distances=True,
    )


def enrich_with_proximity(
    features: Dict,
    reference: Dict,
    radius_meters: float = 50.0,
) -> Dict:
    """
    Enrich features with proximity info to reference features.

    Adds properties:
        - nearest_infrastructure_id
        - nearest_infrastructure_type
        - distance_meters
        - within_radius: bool

    Args:
        features: Features to enrich
        reference: Reference features
        radius_meters: Radius for within_radius flag

    Returns:
        Enriched GeoJSON
    """
    analyzer = SpatialAnalyzer()

    # Calculate distances
    distances = analyzer.calculate_distances(features, reference)

    # Create distance lookup
    distance_map = {d["source_idx"]: d for d in distances}

    # Enrich features
    enriched_features = []
    reference_features = reference.get("features", [])

    for idx, feature in enumerate(features.get("features", [])):
        enriched = {
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": dict(feature.get("properties", {}))
        }

        if idx in distance_map:
            dist_info = distance_map[idx]
            target_idx = dist_info["target_idx"]
            target_props = reference_features[target_idx].get("properties", {})

            enriched["properties"]["nearest_infrastructure_id"] = target_props.get("osm_id")
            enriched["properties"]["nearest_infrastructure_type"] = (
                target_props.get("water") or
                target_props.get("waterway") or
                target_props.get("man_made") or
                target_props.get("landuse") or
                "unknown"
            )
            enriched["properties"]["distance_meters"] = dist_info["distance_meters"]
            enriched["properties"]["within_radius"] = dist_info["distance_meters"] <= radius_meters

        enriched_features.append(enriched)

    return {"type": "FeatureCollection", "features": enriched_features}
