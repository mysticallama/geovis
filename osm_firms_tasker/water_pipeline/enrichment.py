"""
Metadata enrichment module.

Enriches detection data with ACLED conflict event data and infrastructure context.
Classifies detections as confirmed incidents, potential incidents (novel), or known events.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


# Classification categories
CLASSIFICATION_CONFIRMED = "confirmed_incident"  # ACLED match + near infrastructure
CLASSIFICATION_POTENTIAL = "potential_incident"  # Near infrastructure, no ACLED (novel)
CLASSIFICATION_KNOWN = "known_event"  # ACLED match, not near infrastructure
CLASSIFICATION_UNCLASSIFIED = "unclassified"  # No matches


@dataclass
class EnrichmentConfig:
    """Enrichment configuration."""
    acled_match_radius_meters: float = 500.0  # Radius to match ACLED events
    acled_match_days: int = 7  # Days window for temporal matching
    infrastructure_radius_meters: float = 50.0  # Radius to consider "near" infrastructure
    novel_threshold_days: int = 30  # Days to consider detection as novel


class MetadataEnricher:
    """
    Enrich detection data with contextual metadata.

    Matches detections with ACLED events, classifies by proximity to infrastructure,
    and marks novel detections that have no corresponding ACLED data.
    """

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        self.config = config or EnrichmentConfig()

    @staticmethod
    def _meters_to_degrees(meters: float, latitude: float = 0.0) -> float:
        """Convert meters to approximate degrees."""
        lat_factor = 111320
        lon_factor = 111320 * math.cos(math.radians(latitude))
        avg_factor = (lat_factor + lon_factor) / 2
        return meters / avg_factor

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%d %B %Y",
            "%Y%m%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def enrich_with_acled(
        self,
        detections: Dict,
        acled_events: Dict,
        match_radius_meters: Optional[float] = None,
        match_days: Optional[int] = None,
    ) -> Dict:
        """
        Match detections with ACLED events and add metadata.

        Adds properties:
            - acled_event_id: Matched event ID or None
            - acled_event_type: Event type if matched
            - acled_sub_event_type: Sub-event type if matched
            - acled_event_date: Event date if matched
            - acled_fatalities: Fatalities count if matched
            - acled_actors: Involved actors (actor1, actor2)
            - acled_notes: Event notes
            - is_novel: True if no ACLED match (potential new incident)

        Args:
            detections: GeoJSON of detection points
            acled_events: GeoJSON of ACLED events
            match_radius_meters: Spatial matching radius
            match_days: Temporal matching window

        Returns:
            Enriched GeoJSON with ACLED metadata
        """
        match_radius = match_radius_meters or self.config.acled_match_radius_meters
        match_days = match_days or self.config.acled_match_days

        detection_features = detections.get("features", [])
        acled_features = acled_events.get("features", [])

        if not detection_features:
            return {"type": "FeatureCollection", "features": []}

        # Build ACLED spatial index if events exist
        acled_geoms = []
        acled_dates = []

        for f in acled_features:
            if f.get("geometry"):
                acled_geoms.append(shape(f["geometry"]))
                # Parse event date
                event_date = self._parse_date(
                    f.get("properties", {}).get("event_date", "")
                )
                acled_dates.append(event_date)

        acled_tree = STRtree(acled_geoms) if acled_geoms else None

        # Get reference latitude for distance conversion
        if acled_geoms:
            bounds = [g.bounds for g in acled_geoms]
            ref_lat = sum(b[1] + b[3] for b in bounds) / (2 * len(bounds))
        else:
            ref_lat = 0

        match_radius_deg = self._meters_to_degrees(match_radius, ref_lat)

        enriched_features = []

        for det_feature in detection_features:
            if not det_feature.get("geometry"):
                continue

            det_geom = shape(det_feature["geometry"])
            det_props = dict(det_feature.get("properties", {}))

            # Parse detection date
            det_date = self._parse_date(
                det_props.get("acq_date", "") or det_props.get("date", "")
            )

            # Initialize ACLED properties
            det_props["acled_event_id"] = None
            det_props["acled_event_type"] = None
            det_props["acled_sub_event_type"] = None
            det_props["acled_event_date"] = None
            det_props["acled_fatalities"] = None
            det_props["acled_actors"] = None
            det_props["acled_notes"] = None
            det_props["is_novel"] = True

            # Find matching ACLED events
            if acled_tree and acled_geoms:
                # Get nearby ACLED events
                candidates = acled_tree.query(det_geom.buffer(match_radius_deg))

                best_match = None
                best_distance = float("inf")

                for idx in candidates:
                    if idx >= len(acled_features):
                        continue

                    acled_geom = acled_geoms[idx]
                    acled_date = acled_dates[idx]
                    acled_feature = acled_features[idx]

                    # Check spatial distance
                    distance = det_geom.distance(acled_geom)
                    if distance > match_radius_deg:
                        continue

                    # Check temporal match
                    if det_date and acled_date:
                        date_diff = abs((det_date - acled_date).days)
                        if date_diff > match_days:
                            continue

                    # Keep closest match
                    if distance < best_distance:
                        best_distance = distance
                        best_match = acled_feature

                # Apply match
                if best_match:
                    acled_props = best_match.get("properties", {})

                    det_props["acled_event_id"] = acled_props.get("event_id_cnty")
                    det_props["acled_event_type"] = acled_props.get("event_type")
                    det_props["acled_sub_event_type"] = acled_props.get("sub_event_type")
                    det_props["acled_event_date"] = acled_props.get("event_date")
                    det_props["acled_fatalities"] = acled_props.get("fatalities")
                    det_props["acled_notes"] = acled_props.get("notes")
                    det_props["is_novel"] = False

                    # Combine actors
                    actors = []
                    if acled_props.get("actor1"):
                        actors.append(acled_props["actor1"])
                    if acled_props.get("actor2"):
                        actors.append(acled_props["actor2"])
                    det_props["acled_actors"] = "; ".join(actors) if actors else None

            enriched_features.append({
                "type": "Feature",
                "geometry": det_feature["geometry"],
                "properties": det_props
            })

        novel_count = sum(1 for f in enriched_features
                        if f["properties"].get("is_novel", True))
        logger.info(f"ACLED enrichment: {len(enriched_features) - novel_count} matched, "
                   f"{novel_count} novel detections")

        return {"type": "FeatureCollection", "features": enriched_features}

    def enrich_with_infrastructure(
        self,
        detections: Dict,
        infrastructure: Dict,
        radius_meters: Optional[float] = None,
    ) -> Dict:
        """
        Enrich detections with infrastructure proximity data.

        Adds properties:
            - infrastructure_id: Nearest infrastructure OSM ID
            - infrastructure_type: Type of infrastructure
            - infrastructure_name: Name if available
            - distance_to_infrastructure_m: Distance in meters
            - near_infrastructure: True if within radius

        Args:
            detections: GeoJSON of detection points
            infrastructure: GeoJSON of infrastructure features
            radius_meters: Radius to consider "near"

        Returns:
            Enriched GeoJSON with infrastructure metadata
        """
        radius = radius_meters or self.config.infrastructure_radius_meters

        detection_features = detections.get("features", [])
        infra_features = infrastructure.get("features", [])

        if not detection_features:
            return {"type": "FeatureCollection", "features": []}

        # Build infrastructure spatial index
        infra_geoms = [shape(f["geometry"]) for f in infra_features if f.get("geometry")]

        if not infra_geoms:
            # No infrastructure - add empty properties
            enriched = []
            for f in detection_features:
                props = dict(f.get("properties", {}))
                props["infrastructure_id"] = None
                props["infrastructure_type"] = None
                props["infrastructure_name"] = None
                props["distance_to_infrastructure_m"] = None
                props["near_infrastructure"] = False
                enriched.append({
                    "type": "Feature",
                    "geometry": f["geometry"],
                    "properties": props
                })
            return {"type": "FeatureCollection", "features": enriched}

        infra_tree = STRtree(infra_geoms)

        # Get reference latitude
        bounds = [g.bounds for g in infra_geoms]
        ref_lat = sum(b[1] + b[3] for b in bounds) / (2 * len(bounds))

        enriched_features = []

        for det_feature in detection_features:
            if not det_feature.get("geometry"):
                continue

            det_geom = shape(det_feature["geometry"])
            det_props = dict(det_feature.get("properties", {}))

            # Find nearest infrastructure
            nearest_idx = infra_tree.nearest(det_geom)
            nearest_geom = infra_geoms[nearest_idx]
            nearest_feature = infra_features[nearest_idx]
            nearest_props = nearest_feature.get("properties", {})

            # Calculate distance in meters
            distance_deg = det_geom.distance(nearest_geom)
            distance_m = distance_deg * 111320 * math.cos(math.radians(ref_lat))

            # Get infrastructure type
            infra_type = (
                nearest_props.get("water") or
                nearest_props.get("waterway") or
                nearest_props.get("man_made") or
                nearest_props.get("landuse") or
                nearest_props.get("natural") or
                nearest_props.get("amenity") or
                "unknown"
            )

            det_props["infrastructure_id"] = nearest_props.get("osm_id")
            det_props["infrastructure_type"] = infra_type
            det_props["infrastructure_name"] = nearest_props.get("name")
            det_props["distance_to_infrastructure_m"] = round(distance_m, 2)
            det_props["near_infrastructure"] = distance_m <= radius

            enriched_features.append({
                "type": "Feature",
                "geometry": det_feature["geometry"],
                "properties": det_props
            })

        near_count = sum(1 for f in enriched_features
                        if f["properties"].get("near_infrastructure", False))
        logger.info(f"Infrastructure enrichment: {near_count}/{len(enriched_features)} "
                   f"within {radius}m")

        return {"type": "FeatureCollection", "features": enriched_features}

    def classify_detections(self, enriched_detections: Dict) -> Dict:
        """
        Classify detections into categories.

        Categories:
            - confirmed_incident: ACLED match AND near infrastructure
            - potential_incident: Near infrastructure, no ACLED (novel)
            - known_event: ACLED match, not near infrastructure
            - unclassified: No matches

        Args:
            enriched_detections: GeoJSON already enriched with ACLED and infrastructure

        Returns:
            GeoJSON with classification property added
        """
        features = enriched_detections.get("features", [])
        classified_features = []

        counts = {
            CLASSIFICATION_CONFIRMED: 0,
            CLASSIFICATION_POTENTIAL: 0,
            CLASSIFICATION_KNOWN: 0,
            CLASSIFICATION_UNCLASSIFIED: 0,
        }

        for feature in features:
            props = dict(feature.get("properties", {}))

            has_acled = props.get("acled_event_id") is not None
            near_infra = props.get("near_infrastructure", False)

            if has_acled and near_infra:
                classification = CLASSIFICATION_CONFIRMED
            elif near_infra and not has_acled:
                classification = CLASSIFICATION_POTENTIAL
            elif has_acled and not near_infra:
                classification = CLASSIFICATION_KNOWN
            else:
                classification = CLASSIFICATION_UNCLASSIFIED

            props["classification"] = classification
            counts[classification] += 1

            classified_features.append({
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": props
            })

        logger.info(f"Classification: confirmed={counts[CLASSIFICATION_CONFIRMED]}, "
                   f"potential={counts[CLASSIFICATION_POTENTIAL]}, "
                   f"known={counts[CLASSIFICATION_KNOWN]}, "
                   f"unclassified={counts[CLASSIFICATION_UNCLASSIFIED]}")

        return {"type": "FeatureCollection", "features": classified_features}

    def generate_summary(self, enriched_detections: Dict) -> Dict:
        """
        Generate summary statistics for enriched data.

        Args:
            enriched_detections: Enriched and classified GeoJSON

        Returns:
            Summary dict with counts and statistics
        """
        features = enriched_detections.get("features", [])

        summary = {
            "total_detections": len(features),
            "novel_count": 0,
            "acled_matched_count": 0,
            "near_infrastructure_count": 0,
            "classifications": {
                CLASSIFICATION_CONFIRMED: 0,
                CLASSIFICATION_POTENTIAL: 0,
                CLASSIFICATION_KNOWN: 0,
                CLASSIFICATION_UNCLASSIFIED: 0,
            },
            "infrastructure_types": {},
            "acled_event_types": {},
            "average_distance_to_infrastructure_m": None,
        }

        distances = []

        for feature in features:
            props = feature.get("properties", {})

            # Count novel
            if props.get("is_novel", True):
                summary["novel_count"] += 1

            # Count ACLED matches
            if props.get("acled_event_id"):
                summary["acled_matched_count"] += 1

                # Count event types
                event_type = props.get("acled_event_type", "unknown")
                summary["acled_event_types"][event_type] = (
                    summary["acled_event_types"].get(event_type, 0) + 1
                )

            # Count near infrastructure
            if props.get("near_infrastructure"):
                summary["near_infrastructure_count"] += 1

            # Count infrastructure types
            infra_type = props.get("infrastructure_type")
            if infra_type:
                summary["infrastructure_types"][infra_type] = (
                    summary["infrastructure_types"].get(infra_type, 0) + 1
                )

            # Track distances
            dist = props.get("distance_to_infrastructure_m")
            if dist is not None:
                distances.append(dist)

            # Count classifications
            classification = props.get("classification", CLASSIFICATION_UNCLASSIFIED)
            summary["classifications"][classification] = (
                summary["classifications"].get(classification, 0) + 1
            )

        # Calculate average distance
        if distances:
            summary["average_distance_to_infrastructure_m"] = round(
                sum(distances) / len(distances), 2
            )

        return summary


def enrich_detections(
    thermal_detections: Dict,
    acled_events: Optional[Dict] = None,
    infrastructure: Optional[Dict] = None,
    config: Optional[EnrichmentConfig] = None,
) -> Dict:
    """
    High-level enrichment function.

    Applies ACLED matching, infrastructure proximity, and classification.

    Args:
        thermal_detections: GeoJSON of thermal detection points
        acled_events: Optional GeoJSON of ACLED events
        infrastructure: Optional GeoJSON of infrastructure features
        config: Optional EnrichmentConfig

    Returns:
        Enriched and classified GeoJSON
    """
    enricher = MetadataEnricher(config)

    result = thermal_detections

    # Enrich with ACLED if provided
    if acled_events and acled_events.get("features"):
        result = enricher.enrich_with_acled(result, acled_events)

    # Enrich with infrastructure if provided
    if infrastructure and infrastructure.get("features"):
        result = enricher.enrich_with_infrastructure(result, infrastructure)

    # Classify detections
    result = enricher.classify_detections(result)

    return result


def classify_and_summarize(
    enriched_detections: Dict,
    config: Optional[EnrichmentConfig] = None,
) -> Tuple[Dict, Dict]:
    """
    Classify detections and generate summary.

    Args:
        enriched_detections: Already enriched GeoJSON
        config: Optional EnrichmentConfig

    Returns:
        Tuple of (classified GeoJSON, summary dict)
    """
    enricher = MetadataEnricher(config)

    classified = enricher.classify_detections(enriched_detections)
    summary = enricher.generate_summary(classified)

    return classified, summary
