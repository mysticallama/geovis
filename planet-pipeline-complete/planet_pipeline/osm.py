"""
OpenStreetMap Overpass API module.

Query OSM features and export for ML model integration (YOLOv8, HDBSCAN, Siamese, segmentation).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

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


# =============================================================================
# Tag Query
# =============================================================================

@dataclass
class Tag:
    """
    OSM tag query specification.

    Examples:
        Tag("building")                     # All buildings
        Tag("natural", "water")             # Water bodies
        Tag("landuse", ["farmland", "orchard"])  # Multiple values
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
            return f'["{self.key}"~"^({"|".join(self.value)})$"]'
        return f'["{self.key}"="{self.value}"]'


# =============================================================================
# Overpass Client
# =============================================================================

class OverpassClient:
    """OSM Overpass API client with retry logic."""

    def __init__(self, config: Optional[OverpassConfig] = None):
        self.config = config or OverpassConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "planet-pipeline/1.0"})
        self._last_request = 0.0
        self._endpoint_idx = 0

    def _get_endpoint(self) -> str:
        if self._endpoint_idx == 0:
            return self.config.endpoint
        return self.config.fallback_endpoints[self._endpoint_idx - 1]

    def _cycle_endpoint(self):
        total = 1 + len(self.config.fallback_endpoints)
        self._endpoint_idx = (self._endpoint_idx + 1) % total

    def query(self, overpass_ql: str) -> Dict:
        """Execute Overpass QL query with retries."""
        # Rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self._get_endpoint(),
                    data={"data": overpass_ql},
                    timeout=self.config.timeout
                )
                self._last_request = time.time()

                if response.status_code in (429, 503):
                    delay = float(response.headers.get("Retry-After", self.config.retry_delay * (2 ** attempt)))
                    time.sleep(delay)
                    self._cycle_endpoint()
                    continue

                response.raise_for_status()
                return response.json()

            except (requests.Timeout, requests.RequestException) as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                self._cycle_endpoint()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Max retries exceeded")


# =============================================================================
# Query Builder
# =============================================================================

class QueryBuilder:
    """Fluent interface for building Overpass queries."""

    def __init__(self):
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._polygon: Optional[Polygon] = None
        self._tags: List[Tag] = []
        self._elements: List[str] = ["node", "way", "relation"]
        self._timeout: int = 180

    def bbox(self, south: float, west: float, north: float, east: float) -> "QueryBuilder":
        """Set bounding box (south, west, north, east)."""
        self._bbox = (south, west, north, east)
        return self

    def geometry(self, geom: Union[Dict, Polygon]) -> "QueryBuilder":
        """Set geometry (uses bounding box, stores polygon for filtering)."""
        if isinstance(geom, dict):
            geom = shape(geom)
        if isinstance(geom, Polygon):
            self._polygon = geom
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        self._bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
        return self

    def tags(self, *tag_specs: Tag) -> "QueryBuilder":
        """Add tag queries."""
        self._tags.extend(tag_specs)
        return self

    def tag(self, key: str, value: Optional[Union[str, List[str]]] = None,
            regex: Optional[str] = None, negate: bool = False) -> "QueryBuilder":
        """Add a single tag query."""
        self._tags.append(Tag(key, value, regex, negate))
        return self

    def elements(self, *types: str) -> "QueryBuilder":
        """Set element types: node, way, relation."""
        self._elements = list(types)
        return self

    def nodes_only(self) -> "QueryBuilder":
        self._elements = ["node"]
        return self

    def ways_only(self) -> "QueryBuilder":
        self._elements = ["way"]
        return self

    def timeout(self, seconds: int) -> "QueryBuilder":
        self._timeout = seconds
        return self

    def build(self) -> str:
        """Build Overpass QL query string."""
        if not self._bbox:
            raise ValueError("No bounding box or geometry set")
        if not self._tags:
            raise ValueError("No tags specified")

        bbox_str = f"{self._bbox[0]},{self._bbox[1]},{self._bbox[2]},{self._bbox[3]}"
        lines = [f"[out:json][timeout:{self._timeout}];", "("]

        for tag in self._tags:
            for elem in self._elements:
                lines.append(f"  {elem}{tag.to_filter()}({bbox_str});")

        lines.extend([");", "out body;", ">;", "out skel qt;"])
        return "\n".join(lines)

    def get_filter_polygon(self) -> Optional[Polygon]:
        return self._polygon


# =============================================================================
# GeoJSON Converter
# =============================================================================

def _to_geojson(response: Dict, polygon_filter: Optional[Polygon] = None) -> Dict:
    """Convert Overpass response to GeoJSON FeatureCollection."""
    elements = response.get("elements", [])

    # Index nodes and ways
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, List[int]] = {}

    for elem in elements:
        if elem["type"] == "node" and "lat" in elem:
            nodes[elem["id"]] = (elem["lon"], elem["lat"])
        elif elem["type"] == "way":
            ways[elem["id"]] = elem.get("nodes", [])

    features = []
    for elem in elements:
        feature = _element_to_feature(elem, nodes, ways)
        if feature:
            if polygon_filter:
                geom = shape(feature["geometry"])
                if not polygon_filter.intersects(geom):
                    continue
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(features)
        }
    }


def _element_to_feature(elem: Dict, nodes: Dict, ways: Dict) -> Optional[Dict]:
    """Convert OSM element to GeoJSON Feature."""
    etype, eid = elem.get("type"), elem.get("id")
    tags = elem.get("tags", {})
    geometry = None

    if etype == "node" and "lat" in elem:
        geometry = {"type": "Point", "coordinates": [elem["lon"], elem["lat"]]}

    elif etype == "way":
        coords = [nodes[n] for n in elem.get("nodes", []) if n in nodes]
        if len(coords) >= 2:
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geometry = {"type": "Polygon", "coordinates": [coords]}
            else:
                geometry = {"type": "LineString", "coordinates": coords}

    elif etype == "relation" and tags.get("type") == "multipolygon":
        geometry = _build_multipolygon(elem, nodes, ways)

    if not geometry:
        return None

    props = {**tags, "@id": eid, "@type": etype}
    return {"type": "Feature", "id": f"{etype}/{eid}", "properties": props, "geometry": geometry}


def _build_multipolygon(relation: Dict, nodes: Dict, ways: Dict) -> Optional[Dict]:
    """Build multipolygon geometry from relation."""
    outers, inners = [], []
    for member in relation.get("members", []):
        if member.get("type") != "way" or member.get("ref") not in ways:
            continue
        coords = [nodes[n] for n in ways[member["ref"]] if n in nodes]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        (outers if member.get("role") == "outer" else inners).append(coords)

    if not outers:
        return None
    if len(outers) == 1:
        return {"type": "Polygon", "coordinates": [outers[0]] + inners}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in outers]}


# =============================================================================
# High-Level Query Functions
# =============================================================================

def query_features(
    geometry: Union[Dict, Polygon],
    tags: List[Tag],
    elements: List[str] = None,
    config: Optional[OverpassConfig] = None
) -> Dict:
    """
    Query OSM features within geometry.

    Args:
        geometry: GeoJSON geometry or Shapely Polygon
        tags: List of Tag queries
        elements: Element types (default: all)
        config: Optional config

    Returns:
        GeoJSON FeatureCollection
    """
    builder = QueryBuilder().geometry(geometry).tags(*tags)
    if elements:
        builder.elements(*elements)

    client = OverpassClient(config)
    response = client.query(builder.build())
    geojson = _to_geojson(response, builder.get_filter_polygon())

    logger.info(f"Retrieved {len(geojson['features'])} features")
    return geojson


def query_buildings(geometry: Union[Dict, Polygon], types: List[str] = None) -> Dict:
    """Query building footprints."""
    tags = [Tag("building", types)] if types else [Tag("building")]
    return query_features(geometry, tags, ["way", "relation"])


def query_water(geometry: Union[Dict, Polygon], include_waterways: bool = True) -> Dict:
    """Query water features."""
    tags = [Tag("natural", ["water", "wetland"]), Tag("water"), Tag("landuse", ["reservoir", "basin"])]
    if include_waterways:
        tags.append(Tag("waterway"))
    return query_features(geometry, tags)


def query_landuse(geometry: Union[Dict, Polygon], types: List[str] = None) -> Dict:
    """Query land use polygons."""
    tags = [Tag("landuse", types)] if types else [Tag("landuse")]
    return query_features(geometry, tags, ["way", "relation"])


def query_roads(geometry: Union[Dict, Polygon], types: List[str] = None) -> Dict:
    """Query roads/highways."""
    tags = [Tag("highway", types)] if types else [Tag("highway")]
    return query_features(geometry, tags, ["way"])


def query_pois(geometry: Union[Dict, Polygon], amenities: List[str] = None) -> Dict:
    """Query points of interest."""
    tags = [Tag("amenity", amenities)] if amenities else [Tag("amenity")]
    return query_features(geometry, tags, ["node", "way"])


# =============================================================================
# ML Export Functions
# =============================================================================

def extract_bboxes(
    geojson: Dict,
    image_bounds: Optional[Tuple[float, float, float, float]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    class_map: Optional[Dict[str, int]] = None,
    class_key: str = "building"
) -> List[Dict]:
    """
    Extract bounding boxes for object detection (YOLOv8).

    Args:
        geojson: GeoJSON FeatureCollection
        image_bounds: (west, south, east, north) for pixel conversion
        image_size: (width, height) in pixels
        class_map: Map tag values to class IDs
        class_key: Property key for classification

    Returns:
        List of bbox dicts with bbox_geo, bbox_yolo (if image params provided), class_id
    """
    results = []
    for feature in geojson.get("features", []):
        geom = shape(feature["geometry"])
        props = feature.get("properties", {})
        minx, miny, maxx, maxy = geom.bounds

        class_name = props.get(class_key, "unknown")
        class_id = class_map.get(class_name, 0) if class_map else 0

        bbox = {
            "bbox_geo": [minx, miny, maxx, maxy],
            "class_id": class_id,
            "class_name": class_name,
            "osm_id": props.get("@id")
        }

        if image_bounds and image_size:
            west, south, east, north = image_bounds
            w, h = image_size
            px_minx = (minx - west) / (east - west) * w
            px_maxx = (maxx - west) / (east - west) * w
            px_miny = (north - miny) / (north - south) * h
            px_maxy = (north - maxy) / (north - south) * h

            bbox["bbox_yolo"] = [
                (px_minx + px_maxx) / 2 / w,
                (px_miny + px_maxy) / 2 / h,
                abs(px_maxx - px_minx) / w,
                abs(px_maxy - px_miny) / h
            ]

        results.append(bbox)
    return results


def extract_centroids(geojson: Dict) -> np.ndarray:
    """Extract feature centroids for clustering (DBSCAN/HDBSCAN)."""
    points = []
    for feature in geojson.get("features", []):
        centroid = shape(feature["geometry"]).centroid
        points.append([centroid.x, centroid.y])
    return np.array(points) if points else np.empty((0, 2))


def extract_points(geojson: Dict, geometry_type: str = None) -> np.ndarray:
    """Extract point coordinates for spatial analysis."""
    points = []
    for feature in geojson.get("features", []):
        gtype = feature["geometry"]["type"]
        if geometry_type and gtype != geometry_type:
            continue
        geom = shape(feature["geometry"])
        if gtype == "Point":
            points.append([geom.x, geom.y])
        else:
            points.append([geom.centroid.x, geom.centroid.y])
    return np.array(points) if points else np.empty((0, 2))


def create_pairs(
    geojson: Dict,
    same_ratio: float = 0.5,
    class_key: str = "building",
    seed: int = 42
) -> List[Dict]:
    """Create feature pairs for Siamese network training."""
    np.random.seed(seed)
    features = geojson.get("features", [])
    if len(features) < 2:
        return []

    by_class: Dict[str, List[Dict]] = {}
    for f in features:
        cls = f.get("properties", {}).get(class_key, "unknown")
        by_class.setdefault(cls, []).append(f)

    pairs = []
    n = len(features)
    n_same = int(n * same_ratio)

    # Same-class pairs
    multi_class = [c for c, fs in by_class.items() if len(fs) >= 2]
    for _ in range(n_same):
        if not multi_class:
            break
        cls = np.random.choice(multi_class)
        i1, i2 = np.random.choice(len(by_class[cls]), 2, replace=False)
        pairs.append({
            "feature1": by_class[cls][i1], "feature2": by_class[cls][i2],
            "same_class": True, "class1": cls, "class2": cls
        })

    # Different-class pairs
    classes = list(by_class.keys())
    for _ in range(n - n_same):
        if len(classes) < 2:
            break
        c1, c2 = np.random.choice(classes, 2, replace=False)
        i1 = np.random.choice(len(by_class[c1]))
        i2 = np.random.choice(len(by_class[c2]))
        pairs.append({
            "feature1": by_class[c1][i1], "feature2": by_class[c2][i2],
            "same_class": False, "class1": c1, "class2": c2
        })

    np.random.shuffle(pairs)
    return pairs


def generate_mask(
    geojson: Dict,
    bounds: Tuple[float, float, float, float],
    size: Tuple[int, int],
    class_map: Optional[Dict[str, int]] = None,
    class_key: str = "building",
    background: int = 0
) -> np.ndarray:
    """
    Generate raster mask for segmentation.

    Args:
        geojson: GeoJSON FeatureCollection
        bounds: (west, south, east, north)
        size: (width, height) pixels
        class_map: Map tag values to pixel values
        class_key: Property key for classification
        background: Background pixel value

    Returns:
        numpy array (height, width)
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    west, south, east, north = bounds
    width, height = size
    transform = from_bounds(west, south, east, north, width, height)

    shapes = []
    for feature in geojson.get("features", []):
        geom = shape(feature["geometry"])
        cls = feature.get("properties", {}).get(class_key, "unknown")
        val = class_map.get(cls, 1) if class_map else 1
        shapes.append((geom, val))

    if not shapes:
        return np.full((height, width), background, dtype=np.uint8)

    return rasterize(shapes, (height, width), transform=transform, fill=background, dtype=np.uint8)


# =============================================================================
# Export Functions
# =============================================================================

def save_geojson(geojson: Dict, path: Union[str, Path], indent: int = 2) -> Path:
    """Save GeoJSON to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(geojson, f, indent=indent)
    logger.info(f"Saved GeoJSON: {path}")
    return path


def export_yolo(
    geojson: Dict,
    output_dir: Union[str, Path],
    image_name: str,
    image_bounds: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    class_map: Dict[str, int],
    class_key: str = "building"
) -> Path:
    """Export YOLO format annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bboxes = extract_bboxes(geojson, image_bounds, image_size, class_map, class_key)
    output_path = output_dir / f"{image_name}.txt"

    with open(output_path, 'w') as f:
        for bb in bboxes:
            if "bbox_yolo" in bb:
                x, y, w, h = bb["bbox_yolo"]
                f.write(f"{bb['class_id']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    logger.info(f"Exported {len(bboxes)} YOLO annotations: {output_path}")
    return output_path


def export_clustering(
    geojson: Dict,
    output_path: Union[str, Path],
    format: str = "npy"
) -> Path:
    """Export centroids for clustering (npy or csv)."""
    points = extract_centroids(geojson)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        np.save(output_path, points)
    else:
        np.savetxt(output_path, points, delimiter=",", header="lon,lat", comments="")

    logger.info(f"Exported {len(points)} points: {output_path}")
    return output_path


def export_pairs(
    geojson: Dict,
    output_path: Union[str, Path],
    same_ratio: float = 0.5,
    class_key: str = "building"
) -> Path:
    """Export Siamese network pairs."""
    pairs = create_pairs(geojson, same_ratio, class_key)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Exported {len(pairs)} pairs: {output_path}")
    return output_path
