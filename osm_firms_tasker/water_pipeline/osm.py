"""
OpenStreetMap Overpass API client module.

Query OSM features for water infrastructure and other geographic elements.
API Documentation: https://wiki.openstreetmap.org/wiki/Overpass_API
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    MultiPolygon,
    mapping,
    shape,
)
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# Default water infrastructure tags
DEFAULT_WATER_TAGS = [
    {"key": "water", "value": "reservoir"},
    {"key": "water", "value": "basin"},
    {"key": "waterway", "value": "dam"},
    {"key": "waterway", "value": "canal"},
    {"key": "waterway", "value": "ditch"},
]

# Extended water infrastructure tags
EXTENDED_WATER_TAGS = [
    # Water bodies
    {"key": "water", "value": "reservoir"},
    {"key": "water", "value": "basin"},
    {"key": "water", "value": "pond"},
    {"key": "water", "value": "lake"},
    {"key": "natural", "value": "water"},
    {"key": "natural", "value": "spring"},
    # Waterways
    {"key": "waterway", "value": "dam"},
    {"key": "waterway", "value": "canal"},
    {"key": "waterway", "value": "ditch"},
    {"key": "waterway", "value": "stream"},
    {"key": "waterway", "value": "river"},
    {"key": "waterway", "value": "drain"},
    # Man-made water infrastructure
    {"key": "man_made", "value": "water_tower"},
    {"key": "man_made", "value": "water_tank"},
    {"key": "man_made", "value": "water_well"},
    {"key": "man_made", "value": "pumping_station"},
    {"key": "man_made", "value": "water_works"},
    {"key": "man_made", "value": "reservoir_covered"},
    {"key": "man_made", "value": "pipeline"},
    # Landuse
    {"key": "landuse", "value": "reservoir"},
    {"key": "landuse", "value": "basin"},
    # Amenities
    {"key": "amenity", "value": "drinking_water"},
    {"key": "amenity", "value": "water_point"},
]


@dataclass
class Tag:
    """
    OSM tag query specification.

    Examples:
        Tag("water", "reservoir")           # water=reservoir
        Tag("waterway", ["dam", "canal"])   # waterway in [dam, canal]
        Tag("natural")                      # Any natural=* feature
        Tag("highway", regex="primary|secondary")
    """
    key: str
    value: Optional[Union[str, List[str]]] = None
    regex: Optional[str] = None
    negate: bool = False

    def to_filter(self) -> str:
        """Convert to Overpass QL filter."""
        if self.negate:
            return f'[!"{self.key}"]'
        if self.regex:
            return f'["{self.key}"~"{self.regex}"]'
        if self.value is None:
            return f'["{self.key}"]'
        if isinstance(self.value, list):
            values_regex = "|".join(self.value)
            return f'["{self.key}"~"^({values_regex})$"]'
        return f'["{self.key}"="{self.value}"]'


@dataclass
class OverpassConfig:
    """Overpass API configuration."""
    endpoint: str = "https://overpass-api.de/api/interpreter"
    timeout: int = 180
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0
    fallback_endpoints: List[str] = field(default_factory=lambda: [
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ])


class OverpassClient:
    """OSM Overpass API client with retry logic and fallback endpoints."""

    def __init__(self, config: Optional[OverpassConfig] = None):
        self.config = config or OverpassConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "water-pipeline/1.0"})
        self._last_request = 0.0
        self._endpoint_idx = 0

    def _get_endpoint(self) -> str:
        """Get current endpoint."""
        if self._endpoint_idx == 0:
            return self.config.endpoint
        return self.config.fallback_endpoints[self._endpoint_idx - 1]

    def _cycle_endpoint(self):
        """Cycle to next endpoint on failure."""
        total = 1 + len(self.config.fallback_endpoints)
        self._endpoint_idx = (self._endpoint_idx + 1) % total

    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def query(self, overpass_ql: str) -> Dict:
        """
        Execute Overpass QL query.

        Args:
            overpass_ql: Overpass QL query string

        Returns:
            JSON response from Overpass API
        """
        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self._get_endpoint(),
                    data={"data": overpass_ql},
                    timeout=self.config.timeout
                )
                self._last_request = time.time()

                if response.status_code in (429, 503):
                    delay = float(response.headers.get(
                        "Retry-After",
                        self.config.retry_delay * (2 ** attempt)
                    ))
                    logger.warning(f"Overpass rate limited, waiting {delay}s")
                    time.sleep(delay)
                    self._cycle_endpoint()
                    continue

                response.raise_for_status()
                return response.json()

            except (requests.Timeout, requests.RequestException) as e:
                logger.warning(f"Overpass request failed (attempt {attempt + 1}): {e}")
                self._cycle_endpoint()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Max retries exceeded")


class QueryBuilder:
    """Fluent interface for building Overpass QL queries."""

    def __init__(self):
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._polygon: Optional[str] = None
        self._tags: List[Tag] = []
        self._elements: List[str] = ["node", "way", "relation"]
        self._timeout: int = 180
        self._out_format: str = "geom"

    def bbox(self, south: float, west: float, north: float, east: float) -> "QueryBuilder":
        """Set bounding box filter (south, west, north, east)."""
        self._bbox = (south, west, north, east)
        return self

    def geometry(self, geom: Dict) -> "QueryBuilder":
        """Set polygon geometry filter from GeoJSON."""
        geom_shape = shape(geom)

        # Get bbox for initial filter
        bounds = geom_shape.bounds
        self._bbox = (bounds[1], bounds[0], bounds[3], bounds[2])  # south, west, north, east

        # For complex polygons, use poly filter
        if geom_shape.geom_type in ["Polygon", "MultiPolygon"]:
            # Convert to coordinate string for poly filter
            if geom_shape.geom_type == "Polygon":
                coords = list(geom_shape.exterior.coords)
            else:
                # Use convex hull for MultiPolygon
                coords = list(geom_shape.convex_hull.exterior.coords)

            # Format: "lat1 lon1 lat2 lon2 ..."
            poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)
            self._polygon = poly_str

        return self

    def tags(self, tag_list: List[Union[Tag, Dict]]) -> "QueryBuilder":
        """Add multiple tags."""
        for t in tag_list:
            if isinstance(t, dict):
                t = Tag(**t)
            self._tags.append(t)
        return self

    def tag(self, key: str, value: Optional[Union[str, List[str]]] = None) -> "QueryBuilder":
        """Add single tag filter."""
        self._tags.append(Tag(key, value))
        return self

    def elements(self, element_types: List[str]) -> "QueryBuilder":
        """Set element types to query (node, way, relation)."""
        self._elements = element_types
        return self

    def timeout(self, seconds: int) -> "QueryBuilder":
        """Set query timeout."""
        self._timeout = seconds
        return self

    def build(self) -> str:
        """Build Overpass QL query string."""
        if not self._bbox and not self._polygon:
            raise ValueError("Must set bbox or geometry")

        # Build tag filters
        tag_filters = "".join(t.to_filter() for t in self._tags) if self._tags else ""

        # Build area filter
        if self._polygon:
            area_filter = f'(poly:"{self._polygon}")'
        else:
            south, west, north, east = self._bbox
            area_filter = f"({south},{west},{north},{east})"

        # Build element queries
        queries = []
        for elem in self._elements:
            queries.append(f"{elem}{tag_filters}{area_filter};")

        query = f"""
[out:json][timeout:{self._timeout}];
(
  {"".join(queries)}
);
out {self._out_format};
"""
        return query.strip()


def _osm_to_geojson(data: Dict) -> Dict:
    """
    Convert Overpass JSON response to GeoJSON FeatureCollection.

    Handles nodes, ways, and relations with geometry.
    """
    features = []

    for element in data.get("elements", []):
        elem_type = element.get("type")
        tags = element.get("tags", {})

        geometry = None

        if elem_type == "node":
            if "lat" in element and "lon" in element:
                geometry = {
                    "type": "Point",
                    "coordinates": [element["lon"], element["lat"]]
                }

        elif elem_type == "way":
            if "geometry" in element:
                coords = [[p["lon"], p["lat"]] for p in element["geometry"]]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    # Closed way = polygon
                    geometry = {"type": "Polygon", "coordinates": [coords]}
                else:
                    # Open way = linestring
                    geometry = {"type": "LineString", "coordinates": coords}

        elif elem_type == "relation":
            if "members" in element:
                # Try to build geometry from members
                polygons = []
                lines = []

                for member in element.get("members", []):
                    if "geometry" in member:
                        coords = [[p["lon"], p["lat"]] for p in member["geometry"]]
                        if len(coords) >= 4 and coords[0] == coords[-1]:
                            polygons.append(coords)
                        elif len(coords) >= 2:
                            lines.append(coords)

                if polygons:
                    if len(polygons) == 1:
                        geometry = {"type": "Polygon", "coordinates": polygons}
                    else:
                        geometry = {"type": "MultiPolygon", "coordinates": [[p] for p in polygons]}
                elif lines:
                    if len(lines) == 1:
                        geometry = {"type": "LineString", "coordinates": lines[0]}
                    else:
                        geometry = {"type": "MultiLineString", "coordinates": lines}

        if geometry:
            # Add OSM metadata to properties
            properties = dict(tags)
            properties["osm_type"] = elem_type
            properties["osm_id"] = element.get("id")

            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            })

    return {"type": "FeatureCollection", "features": features}


def query_water_infrastructure(
    geometry: Dict,
    tags: Optional[List[Union[Tag, Dict]]] = None,
    include_extended: bool = False,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """
    Query water infrastructure features within geometry.

    Args:
        geometry: GeoJSON geometry
        tags: Custom tag filters (default: DEFAULT_WATER_TAGS)
        include_extended: Include extended water tags
        config: Optional OverpassConfig

    Returns:
        GeoJSON FeatureCollection with water infrastructure features
    """
    client = OverpassClient(config)

    # Use default or extended tags
    if tags is None:
        tags = EXTENDED_WATER_TAGS if include_extended else DEFAULT_WATER_TAGS

    # Build and execute query
    builder = QueryBuilder()
    builder.geometry(geometry).tags(tags)

    query = builder.build()
    logger.debug(f"Overpass query:\n{query}")

    result = client.query(query)

    geojson = _osm_to_geojson(result)
    logger.info(f"Retrieved {len(geojson['features'])} water infrastructure features")

    return geojson


def query_features(
    geometry: Dict,
    tags: List[Union[Tag, Dict]],
    elements: Optional[List[str]] = None,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """
    Query OSM features with custom tags.

    Args:
        geometry: GeoJSON geometry
        tags: Tag filters
        elements: Element types (default: node, way, relation)
        config: Optional OverpassConfig

    Returns:
        GeoJSON FeatureCollection
    """
    client = OverpassClient(config)

    builder = QueryBuilder()
    builder.geometry(geometry).tags(tags)

    if elements:
        builder.elements(elements)

    query = builder.build()
    result = client.query(query)

    geojson = _osm_to_geojson(result)
    logger.info(f"Retrieved {len(geojson['features'])} OSM features")

    return geojson


def query_buildings(
    geometry: Dict,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """Query building footprints."""
    return query_features(
        geometry,
        tags=[Tag("building")],
        elements=["way", "relation"],
        config=config,
    )


def query_roads(
    geometry: Dict,
    road_types: Optional[List[str]] = None,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """Query road network."""
    if road_types is None:
        road_types = ["motorway", "trunk", "primary", "secondary", "tertiary",
                      "residential", "unclassified"]

    return query_features(
        geometry,
        tags=[Tag("highway", road_types)],
        elements=["way"],
        config=config,
    )


def query_landuse(
    geometry: Dict,
    landuse_types: Optional[List[str]] = None,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """Query landuse polygons."""
    tags = [Tag("landuse", landuse_types)] if landuse_types else [Tag("landuse")]

    return query_features(
        geometry,
        tags=tags,
        elements=["way", "relation"],
        config=config,
    )


def save_geojson(geojson: Dict, filepath: str) -> None:
    """Save GeoJSON to file."""
    with open(filepath, "w") as f:
        json.dump(geojson, f, indent=2)
    logger.info(f"Saved GeoJSON to {filepath}")
