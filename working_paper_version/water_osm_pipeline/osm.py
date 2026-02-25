"""
OpenStreetMap Overpass API client.

Queries OSM for water AND food/agricultural infrastructure within an AOI.
The tag set is intentionally broad — all tags that could plausibly touch on
water supply or food systems are included so that downstream analysis can
decide which features are relevant.

API docs: https://wiki.openstreetmap.org/wiki/Overpass_API
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag sets
# ---------------------------------------------------------------------------

# Core water infrastructure (always included)
WATER_FOOD_TAGS: List[Dict[str, str]] = [
    # ── Water bodies ─────────────────────────────────────────────────────────
    {"key": "natural",   "value": "water"},
    {"key": "natural",   "value": "spring"},
    {"key": "water",     "value": "reservoir"},
    {"key": "water",     "value": "basin"},
    {"key": "water",     "value": "pond"},
    {"key": "water",     "value": "lake"},
    # ── Waterways ─────────────────────────────────────────────────────────────
    {"key": "waterway",  "value": "dam"},
    {"key": "waterway",  "value": "canal"},
    {"key": "waterway",  "value": "ditch"},
    {"key": "waterway",  "value": "stream"},
    {"key": "waterway",  "value": "river"},
    {"key": "waterway",  "value": "drain"},
    {"key": "waterway",  "value": "weir"},
    {"key": "waterway",  "value": "lock_gate"},
    # ── Drinking water & sanitation ─────────────────────────────────────────
    {"key": "amenity",   "value": "drinking_water"},
    {"key": "amenity",   "value": "water_point"},
    {"key": "amenity",   "value": "watering_place"},
    {"key": "emergency", "value": "drinking_water"},
    # ── Man-made water infrastructure ────────────────────────────────────────
    {"key": "man_made",  "value": "water_tower"},
    {"key": "man_made",  "value": "water_tank"},
    {"key": "man_made",  "value": "water_well"},
    {"key": "man_made",  "value": "pumping_station"},
    {"key": "man_made",  "value": "water_works"},
    {"key": "man_made",  "value": "reservoir_covered"},
    {"key": "man_made",  "value": "pipeline"},
    {"key": "man_made",  "value": "wastewater_plant"},
    {"key": "man_made",  "value": "desalination_plant"},
    # ── Irrigation ──────────────────────────────────────────────────────────
    {"key": "waterway",  "value": "irrigation_canal"},
    {"key": "man_made",  "value": "irrigation_dam"},
    {"key": "landuse",   "value": "irrigation"},
    # ── Landuse (water) ─────────────────────────────────────────────────────
    {"key": "landuse",   "value": "reservoir"},
    {"key": "landuse",   "value": "basin"},
    # ── Water source tags ────────────────────────────────────────────────────
    {"key": "water_source", "value": "main"},
    {"key": "water_source", "value": "water_works"},
    {"key": "water_source", "value": "tube_well"},
    {"key": "water_source", "value": "water_tank"},
    # ── Food storage & distribution ──────────────────────────────────────────
    {"key": "man_made",  "value": "silo"},
    {"key": "man_made",  "value": "grain_silo"},
    {"key": "building",  "value": "silo"},
    {"key": "building",  "value": "warehouse"},
    {"key": "building",  "value": "barn"},
    {"key": "building",  "value": "farm"},
    {"key": "building",  "value": "greenhouse"},
    {"key": "building",  "value": "storage_tank"},
    # ── Food markets & retail ────────────────────────────────────────────────
    {"key": "amenity",   "value": "marketplace"},
    {"key": "amenity",   "value": "market"},
    {"key": "shop",      "value": "supermarket"},
    {"key": "shop",      "value": "wholesale"},
    {"key": "shop",      "value": "convenience"},
    {"key": "shop",      "value": "bakery"},
    {"key": "shop",      "value": "greengrocer"},
    # ── Agricultural landuse ─────────────────────────────────────────────────
    {"key": "landuse",   "value": "farmland"},
    {"key": "landuse",   "value": "orchard"},
    {"key": "landuse",   "value": "greenhouse_horticulture"},
    {"key": "landuse",   "value": "allotments"},
    {"key": "landuse",   "value": "plant_nursery"},
    # ── Food aid & humanitarian ──────────────────────────────────────────────
    {"key": "amenity",   "value": "food_bank"},
    {"key": "social_facility", "value": "food_bank"},
    # ── Power (critical infrastructure often co-located with water) ──────────
    {"key": "power",     "value": "generator"},
    {"key": "power",     "value": "plant"},
    {"key": "power",     "value": "substation"},
]

# Extended set — adds more granular tags for thorough queries
EXTENDED_WATER_FOOD_TAGS: List[Dict[str, str]] = WATER_FOOD_TAGS + [
    # Additional waterway features
    {"key": "waterway",  "value": "floodgate"},
    {"key": "waterway",  "value": "boatyard"},
    {"key": "natural",   "value": "wetland"},
    {"key": "natural",   "value": "marsh"},
    # Additional storage
    {"key": "man_made",  "value": "storage_tank"},
    {"key": "building",  "value": "storage"},
    {"key": "industrial","value": "slaughterhouse"},
    {"key": "industrial","value": "cold_storage"},
    # Ports and logistics (relevant to food supply chains)
    {"key": "landuse",   "value": "harbour"},
    {"key": "waterway",  "value": "dock"},
    {"key": "man_made",  "value": "pier"},
    # Livestock (water + food)
    {"key": "landuse",   "value": "meadow"},
    {"key": "landuse",   "value": "pasture"},
    {"key": "amenity",   "value": "watering_place"},
    # Additional agricultural
    {"key": "landuse",   "value": "vineyard"},
    {"key": "landuse",   "value": "forest"},
    {"key": "landuse",   "value": "logging"},
    # Mills and processing
    {"key": "man_made",  "value": "flour_mill"},
    {"key": "craft",     "value": "bakery"},
    {"key": "industrial","value": "dairy"},
    {"key": "industrial","value": "food_processing"},
    # Well / borehole tagging variants
    {"key": "man_made",  "value": "borehole"},
    {"key": "man_made",  "value": "monitoring_station"},
    {"key": "man_made",  "value": "water_tap"},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Tag:
    """
    Single OSM tag query specification.

    Examples::

        Tag("water", "reservoir")            # water=reservoir
        Tag("waterway", ["dam", "canal"])    # waterway∈{dam,canal}
        Tag("natural")                       # any natural=*
        Tag("highway", regex="primary|secondary")
    """
    key: str
    value: Optional[Union[str, List[str]]] = None
    regex: Optional[str] = None
    negate: bool = False

    def to_filter(self) -> str:
        """Return Overpass QL filter string."""
        if self.negate:
            return f'[!"{self.key}"]'
        if self.regex:
            return f'["{self.key}"~"{self.regex}"]'
        if self.value is None:
            return f'["{self.key}"]'
        if isinstance(self.value, list):
            regex = "|".join(self.value)
            return f'["{self.key}"~"^({regex})$"]'
        return f'["{self.key}"="{self.value}"]'


@dataclass
class OverpassConfig:
    """Overpass API connection configuration."""
    endpoint: str = "https://overpass-api.de/api/interpreter"
    timeout: int = 300
    max_retries: int = 4
    retry_delay: float = 5.0
    rate_limit_delay: float = 1.5
    fallback_endpoints: List[str] = field(default_factory=lambda: [
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ])


# ---------------------------------------------------------------------------
# Overpass client
# ---------------------------------------------------------------------------

class OverpassClient:
    """
    Overpass API client with endpoint cycling and exponential backoff.

    Automatically rotates between fallback endpoints on errors.
    """

    def __init__(self, config: Optional[OverpassConfig] = None):
        self.config = config or OverpassConfig()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "water-osm-pipeline/1.0"
        self._last_request = 0.0
        self._endpoint_idx = 0

    def _endpoint(self) -> str:
        if self._endpoint_idx == 0:
            return self.config.endpoint
        return self.config.fallback_endpoints[self._endpoint_idx - 1]

    def _cycle(self):
        total = 1 + len(self.config.fallback_endpoints)
        self._endpoint_idx = (self._endpoint_idx + 1) % total
        logger.info(f"Switching Overpass endpoint → {self._endpoint()}")

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def query(self, overpass_ql: str) -> Dict:
        """
        Execute an Overpass QL query and return the JSON response.

        Retries with exponential backoff on 429/502/503/504.
        """
        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                resp = self.session.post(
                    self._endpoint(),
                    data={"data": overpass_ql},
                    timeout=self.config.timeout,
                )
                self._last_request = time.time()

                if resp.status_code in (429, 502, 503, 504):
                    delay = float(resp.headers.get(
                        "Retry-After",
                        self.config.retry_delay * (2 ** attempt),
                    ))
                    logger.warning(
                        f"Overpass HTTP {resp.status_code} — "
                        f"cycling endpoint, retry in {delay:.0f}s"
                    )
                    time.sleep(delay)
                    self._cycle()
                    continue

                resp.raise_for_status()
                return resp.json()

            except (requests.Timeout, requests.RequestException) as exc:
                logger.warning(f"Overpass request failed (attempt {attempt+1}): {exc}")
                self._cycle()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Overpass: max retries exceeded")


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Fluent Overpass QL query builder."""

    def __init__(self):
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._polygon: Optional[str] = None
        self._tags: List[Tag] = []
        self._elements: List[str] = ["node", "way", "relation"]
        self._timeout: int = 180

    def bbox(self, south: float, west: float, north: float, east: float) -> "QueryBuilder":
        self._bbox = (south, west, north, east)
        return self

    def geometry(self, geom: Dict) -> "QueryBuilder":
        """Set spatial filter from a GeoJSON geometry."""
        geom_shape = shape(geom)
        bounds = geom_shape.bounds
        self._bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

        if geom_shape.geom_type in ("Polygon", "MultiPolygon"):
        # Build poly string for Overpass (lat lon pairs, space-separated)
            if geom_shape.geom_type == "Polygon":
                coords = list(geom_shape.exterior.coords)
            else:
                coords = list(geom_shape.convex_hull.exterior.coords)
            self._polygon = " ".join(f"{lat} {lon}" for lon, lat in coords)

        return self

    def tags(self, tag_list: List[Union[Tag, Dict]]) -> "QueryBuilder":
        for t in tag_list:
            if isinstance(t, dict):
                t = Tag(**t)
            self._tags.append(t)
        return self

    def tag(self, key: str, value=None) -> "QueryBuilder":
        self._tags.append(Tag(key, value))
        return self

    def elements(self, types: List[str]) -> "QueryBuilder":
        self._elements = types
        return self

    def timeout(self, seconds: int) -> "QueryBuilder":
        self._timeout = seconds
        return self

    def build(self) -> str:
        """Compile to an Overpass QL query string."""
        if not self._bbox and not self._polygon:
            raise ValueError("Must call .bbox() or .geometry() first")

        tag_filters = "".join(t.to_filter() for t in self._tags)

        if self._polygon:
            area = f'(poly:"{self._polygon}")'
        else:
            s, w, n, e = self._bbox
            area = f"({s},{w},{n},{e})"

        queries = [f"{elem}{tag_filters}{area};" for elem in self._elements]

        return (
            f"[out:json][timeout:{self._timeout}];\n"
            f"(\n  {'  '.join(queries)}\n);\n"
            "out geom;"
        )


# ---------------------------------------------------------------------------
# GeoJSON conversion
# ---------------------------------------------------------------------------

def _osm_to_geojson(data: Dict) -> Dict:
    """Convert raw Overpass JSON to a GeoJSON FeatureCollection."""
    features = []

    for el in data.get("elements", []):
        el_type = el.get("type")
        tags = el.get("tags", {})
        geometry = None

        if el_type == "node" and "lat" in el:
            geometry = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}

        elif el_type == "way" and "geometry" in el:
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geometry = {"type": "Polygon", "coordinates": [coords]}
            else:
                geometry = {"type": "LineString", "coordinates": coords}

        elif el_type == "relation" and "members" in el:
            polygons, lines = [], []
            for member in el.get("members", []):
                if "geometry" in member:
                    coords = [[p["lon"], p["lat"]] for p in member["geometry"]]
                    if len(coords) >= 4 and coords[0] == coords[-1]:
                        polygons.append(coords)
                    elif len(coords) >= 2:
                        lines.append(coords)
            if polygons:
                geometry = (
                    {"type": "Polygon", "coordinates": polygons}
                    if len(polygons) == 1
                    else {"type": "MultiPolygon", "coordinates": [[p] for p in polygons]}
                )
            elif lines:
                geometry = (
                    {"type": "LineString", "coordinates": lines[0]}
                    if len(lines) == 1
                    else {"type": "MultiLineString", "coordinates": lines}
                )

        if geometry:
            props = dict(tags)
            props["osm_type"] = el_type
            props["osm_id"] = el.get("id")
            # Classify the feature for easier downstream filtering
            props["infra_category"] = _classify_infra(tags)
            features.append({"type": "Feature", "geometry": geometry, "properties": props})

    return {"type": "FeatureCollection", "features": features}


def _classify_infra(tags: Dict) -> str:
    """
    Assign a high-level infrastructure category based on OSM tags.

    Returns one of: "water_supply", "waterway", "irrigation",
    "food_storage", "food_market", "agriculture", "power", "other".
    """
    water_supply_vals = {
        "reservoir", "basin", "water_tower", "water_tank", "water_well",
        "pumping_station", "water_works", "reservoir_covered", "pipeline",
        "wastewater_plant", "desalination_plant", "drinking_water",
        "water_point", "watering_place", "spring",
    }
    for k, v in tags.items():
        if k in ("water", "man_made", "amenity", "natural") and str(v) in water_supply_vals:
            return "water_supply"

    waterway_vals = {"dam", "canal", "ditch", "stream", "river", "drain", "weir"}
    if tags.get("waterway") in waterway_vals:
        return "waterway"

    if tags.get("waterway") == "irrigation_canal" or tags.get("landuse") == "irrigation":
        return "irrigation"

    food_storage_vals = {"silo", "grain_silo", "warehouse", "barn", "storage_tank"}
    if tags.get("man_made") in food_storage_vals or tags.get("building") in food_storage_vals:
        return "food_storage"

    if tags.get("amenity") in ("marketplace", "market", "food_bank") or \
       tags.get("shop") in ("supermarket", "wholesale", "bakery", "greengrocer"):
        return "food_market"

    agri_landuse = {"farmland", "orchard", "greenhouse_horticulture", "allotments", "vineyard"}
    if tags.get("landuse") in agri_landuse or "building" == "farm":
        return "agriculture"

    if tags.get("power") in ("generator", "plant", "substation"):
        return "power"

    return "other"


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def query_infrastructure(
    geometry: Dict,
    tags: Optional[List[Union[Tag, Dict]]] = None,
    extended: bool = False,
    config: Optional[OverpassConfig] = None,
) -> Dict:
    """
    Query water and food infrastructure features within *geometry*.

    Args:
        geometry: GeoJSON geometry dict (AOI).
        tags: Custom tag list. Defaults to :data:`WATER_FOOD_TAGS` or
              :data:`EXTENDED_WATER_FOOD_TAGS` when ``extended=True``.
        extended: Use the larger extended tag set.
        config: Optional :class:`OverpassConfig`.

    Returns:
        GeoJSON FeatureCollection with an ``infra_category`` property on
        each feature indicating water_supply / waterway / irrigation /
        food_storage / food_market / agriculture / power / other.
    """
    client = OverpassClient(config)

    if tags is None:
        tags = EXTENDED_WATER_FOOD_TAGS if extended else WATER_FOOD_TAGS

    # Build a union-style query: one statement per tag pair so we retrieve
    # *all* features that match *any* tag in the list.
    qb = QueryBuilder()
    qb.geometry(geometry)

    # Build the Overpass QL manually to support union semantics
    geom_shape = shape(geometry)
    bounds = geom_shape.bounds
    if geom_shape.geom_type in ("Polygon", "MultiPolygon"):
        if geom_shape.geom_type == "Polygon":
            coords = list(geom_shape.exterior.coords)
        else:
            coords = list(geom_shape.convex_hull.exterior.coords)
        poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)
        area = f'(poly:"{poly_str}")'
    else:
        s, w, n, e = bounds[1], bounds[0], bounds[3], bounds[2]
        area = f"({s},{w},{n},{e})"

    tag_statements = []
    for t_dict in tags:
        t = t_dict if isinstance(t_dict, Tag) else Tag(**t_dict)
        tf = t.to_filter()
        for elem in ("node", "way", "relation"):
            tag_statements.append(f"  {elem}{tf}{area};")

    query = (
        f"[out:json][timeout:300];\n"
        f"(\n"
        + "\n".join(tag_statements) + "\n"
        f");\n"
        "out geom;"
    )

    logger.debug(f"Overpass query (first 400 chars):\n{query[:400]}")

    result = client.query(query)
    geojson = _osm_to_geojson(result)

    # Deduplicate by osm_id (union query can return duplicates)
    seen: set = set()
    unique: List[Dict] = []
    for feat in geojson["features"]:
        osm_id = feat["properties"].get("osm_id")
        if osm_id not in seen:
            seen.add(osm_id)
            unique.append(feat)

    geojson["features"] = unique
    logger.info(
        f"OSM query: {len(unique)} unique infrastructure features "
        f"({'extended' if extended else 'standard'} tags)"
    )
    return geojson


def save_geojson(geojson: Dict, filepath) -> None:
    """Write a GeoJSON dict to *filepath*."""
    import pathlib
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as fh:
        json.dump(geojson, fh, indent=2)
    logger.info(f"Saved GeoJSON → {filepath}")
