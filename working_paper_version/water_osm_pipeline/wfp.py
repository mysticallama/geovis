"""
WFP Logistics Cluster data fetcher.

Downloads humanitarian infrastructure data (roads, airports, border
crossings, storage facilities) from the WFP Logistics Cluster via the
Humanitarian Data Exchange (HDX) public API.

HDX API docs:     https://data.humdata.org/api/
LogIE portal:     https://logcluster.org/tools/logie
WFP HDX org page: https://data.humdata.org/organization/wfp

Authentication:
    Most WFP/Logistics Cluster datasets on HDX are publicly available
    without authentication.  If an HDX API key is provided (HDX_API_KEY
    in .env), it is sent as a header to allow access to any non-public
    organisation datasets the key is permitted to see.

Workflow:
    1. ``search_datasets()``  — search HDX for logistics datasets by country
    2. ``download_resources()`` — download matching resources to disk
    3. ``to_geojson()``       — convert shapefiles / GeoJSONs to GeoJSON dicts
    4. ``fetch_logistics_data()`` — end-to-end convenience wrapper

Supported resource formats:
    - GeoJSON (.geojson, .json with FeatureCollection)
    - Shapefile (.shp, downloaded as a zip then extracted)
    - GeoPackage (.gpkg)

Fallback:
    If no HDX datasets are found for the specified country, the module
    falls back to OpenStreetMap Overpass queries for roads and amenities
    relevant to logistics (airports, warehouses, border crossings).
"""

import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

HDX_API_BASE = "https://data.humdata.org/api/3/action"

# Resource format extensions we can ingest
INGESTABLE_FORMATS = {".geojson", ".json", ".shp", ".gpkg", ".zip"}

# HDX search keywords for logistics cluster data
LOGISTICS_KEYWORDS = [
    "logistics cluster",
    "road network",
    "airports",
    "border crossing",
    "warehouses",
    "storage",
    "transportation",
    "wfp infrastructure",
]

# Fallback Overpass tags when HDX returns nothing
FALLBACK_OSM_LOGISTICS_TAGS = [
    {"key": "aeroway",  "value": "aerodrome"},
    {"key": "aeroway",  "value": "airport"},
    {"key": "highway",  "value": "primary"},
    {"key": "highway",  "value": "trunk"},
    {"key": "highway",  "value": "secondary"},
    {"key": "highway",  "value": "motorway"},
    {"key": "barrier",  "value": "border_control"},
    {"key": "amenity",  "value": "customs"},
    {"key": "building", "value": "warehouse"},
    {"key": "man_made", "value": "pier"},
    {"key": "landuse",  "value": "harbour"},
    {"key": "waterway", "value": "dock"},
]


@dataclass
class WFPConfig:
    """
    Configuration for WFP / HDX data retrieval.

    Attributes:
        hdx_api_key:        Optional HDX API key (HDX_API_KEY env var).
                            Required only for restricted datasets.
        country_iso3:       ISO 3166-1 alpha-3 country code (e.g. ``"SDN"``).
        country_name:       Human-readable name used in HDX searches.
        max_results:        Maximum number of HDX datasets to inspect.
        rate_limit_delay:   Seconds between HDX API calls.
        download_timeout:   Max seconds for a single resource download.
        cache_downloads:    If True, skip download if file already exists.
        fallback_to_osm:    If True, fall back to OSM when HDX returns nothing.
    """
    hdx_api_key: Optional[str] = None
    country_iso3: str = ""
    country_name: str = ""
    max_results: int = 20
    rate_limit_delay: float = 1.0
    download_timeout: int = 120
    cache_downloads: bool = True
    fallback_to_osm: bool = True
    max_features_per_layer: int = 8000  # cap per layer to keep map.html manageable

    def __post_init__(self):
        if self.hdx_api_key is None:
            self.hdx_api_key = os.environ.get("HDX_API_KEY")


class WFPLogisticsClient:
    """
    HDX-based client for WFP Logistics Cluster infrastructure data.

    Searches HDX for logistics infrastructure datasets for the configured
    country, downloads shapefiles or GeoJSON resources, and converts them
    to GeoJSON FeatureCollections keyed by layer type (roads, airports,
    crossings, storage).

    If HDX returns no results, falls back to querying OSM Overpass for
    equivalent features.

    Example::

        cfg    = WFPConfig(country_iso3="SDN", country_name="Sudan")
        client = WFPLogisticsClient(cfg)
        layers = client.fetch(aoi_geometry, output_dir=Path("output/wfp"))
        # layers = {"roads": {...}, "airports": {...}, "crossings": {...}, ...}
    """

    def __init__(self, config: Optional[WFPConfig] = None):
        self.config = config or WFPConfig()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "water-osm-pipeline/1.0"
        if self.config.hdx_api_key:
            self.session.headers["X-CKAN-API-Key"] = self.config.hdx_api_key
        self._last_request = 0.0

    # ── rate limiting ────────────────────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    # ── HDX search ──────────────────────────────────────────────────────────

    def search_datasets(self) -> List[Dict]:
        """
        Search HDX for logistics cluster datasets for the configured country.

        Tries multiple search queries (country ISO3 + logistics keywords) and
        de-duplicates results by dataset ID.

        Returns:
            List of HDX dataset dicts (may be empty).
        """
        found: Dict[str, Dict] = {}

        search_terms = [
            f"logistics {self.config.country_iso3}",
            f"logistics {self.config.country_name}",
            f"roads {self.config.country_iso3}",
            f"airports {self.config.country_iso3}",
            f"infrastructure {self.config.country_iso3}",
            f"wfp {self.config.country_iso3}",
        ]

        for term in search_terms:
            if len(found) >= self.config.max_results:
                break
            self._rate_limit()

            params = {
                "q": term,
                "rows": min(self.config.max_results, 10),
                "fq": f"groups:{self.config.country_iso3.lower()}" if self.config.country_iso3 else "",
            }

            try:
                resp = self.session.get(
                    f"{HDX_API_BASE}/package_search",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json().get("result", {}).get("results", [])
                for ds in results:
                    found[ds["id"]] = ds
                logger.debug(f"HDX search '{term}': {len(results)} datasets")
            except Exception as exc:
                logger.warning(f"HDX search failed for '{term}': {exc}")

        datasets = list(found.values())
        logger.info(f"HDX: found {len(datasets)} unique datasets for '{self.config.country_iso3}'")
        return datasets

    def _select_resources(self, datasets: List[Dict]) -> List[Dict]:
        """
        Extract downloadable resources from dataset list.

        Filters to spatial formats and de-duplicates by URL.
        """
        seen_urls: set = set()
        resources: List[Dict] = []

        logistics_keywords_lower = [k.lower() for k in LOGISTICS_KEYWORDS]

        for ds in datasets:
            ds_title = ds.get("title", "").lower()
            ds_notes = ds.get("notes", "").lower()

            for res in ds.get("resources", []):
                url    = res.get("url", "")
                fmt    = res.get("format", "").lower()
                name   = res.get("name", "").lower()
                suffix = Path(url).suffix.lower()

                if url in seen_urls:
                    continue
                if suffix not in INGESTABLE_FORMATS and fmt not in {
                    "geojson", "json", "shp", "shapefile", "gpkg", "zip",
                }:
                    continue

                # Score relevance
                score = 0
                for kw in logistics_keywords_lower:
                    if kw in name or kw in ds_title:
                        score += 2
                    if kw in ds_notes:
                        score += 1

                resources.append({**res, "_score": score, "_dataset_title": ds.get("title", "")})
                seen_urls.add(url)

        resources.sort(key=lambda r: r["_score"], reverse=True)
        logger.info(f"Selected {len(resources)} downloadable resources")
        return resources

    # ── download ────────────────────────────────────────────────────────────

    def download_resources(
        self,
        resources: List[Dict],
        output_dir: Path,
        max_downloads: int = 10,
    ) -> List[Path]:
        """
        Download spatial resources from HDX to *output_dir*.

        Args:
            resources:    Resource dicts from :meth:`_select_resources`.
            output_dir:   Directory for downloaded files.
            max_downloads: Hard cap on downloads per run.

        Returns:
            List of local file :class:`~pathlib.Path` objects.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Tuple[Path, Dict]] = []

        for res in resources[:max_downloads]:
            url  = res.get("url", "")
            name = re.sub(r"[^\w\-.]", "_", res.get("name", "resource"))
            suffix = Path(url).suffix.lower() or ".bin"
            out_path = output_dir / f"{name}{suffix}"

            if self.config.cache_downloads and out_path.exists():
                logger.info(f"  [cache] {out_path.name}")
                downloaded.append((out_path, res))
                continue

            self._rate_limit()
            try:
                resp = self.session.get(url, stream=True, timeout=self.config.download_timeout)
                resp.raise_for_status()
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                logger.info(f"  Downloaded → {out_path.name}")
                downloaded.append((out_path, res))
            except Exception as exc:
                logger.warning(f"  Failed to download {url}: {exc}")

        return downloaded

    # ── conversion ──────────────────────────────────────────────────────────

    @staticmethod
    def _file_to_geojson(path: Path) -> Optional[Dict]:
        """Convert a local spatial file to a GeoJSON FeatureCollection."""
        suffix = path.suffix.lower()

        # GeoJSON / JSON
        if suffix in {".geojson", ".json"}:
            try:
                with open(path) as fh:
                    data = json.load(fh)
                if data.get("type") == "FeatureCollection":
                    return data
                if data.get("type") == "Feature":
                    return {"type": "FeatureCollection", "features": [data]}
            except Exception as exc:
                logger.warning(f"Could not read {path.name} as GeoJSON: {exc}")
            return None

        # Shapefile / GeoPackage
        if suffix in {".shp", ".gpkg"}:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                return json.loads(gdf.to_json())
            except Exception as exc:
                logger.warning(f"Could not read {path.name} with geopandas: {exc}")
            return None

        # ZIP — extract and recurse into spatial files
        if suffix == ".zip":
            try:
                with TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(path) as zf:
                        zf.extractall(tmp)
                    tmp_path = Path(tmp)
                    for ext in (".geojson", ".json", ".shp", ".gpkg"):
                        for sp_file in tmp_path.rglob(f"*{ext}"):
                            result = WFPLogisticsClient._file_to_geojson(sp_file)
                            if result and result.get("features"):
                                return result
            except Exception as exc:
                logger.warning(f"Could not extract {path.name}: {exc}")
            return None

        logger.debug(f"Unsupported format: {path.name}")
        return None

    @staticmethod
    def _classify_layer(title: str, filename: str) -> str:
        """Assign a layer type from dataset title / filename keywords."""
        text = f"{title} {filename}".lower()
        # Check airports before roads — "aerodrome/airport" is more specific
        # than "transport" which can appear in airport dataset titles
        if any(k in text for k in ("airport", "airstrip", "aerodrome", "airfield")):
            return "airports"
        if any(k in text for k in ("road", "highway", "route", "transport")):
            return "roads"
        if any(k in text for k in ("border", "crossing", "checkpoint", "boundary")):
            return "crossings"
        if any(k in text for k in ("warehouse", "storage", "port", "depot", "hub")):
            return "storage"
        return "other"

    # ── OSM major roads ──────────────────────────────────────────────────────

    def _query_osm_major_roads(self, aoi_geometry: Dict) -> List[Dict]:
        """
        Query OSM Overpass for roads within the AOI.

        First queries major roads (motorway, trunk, primary, secondary).
        Then attempts to add residential/tertiary roads.  If the combined
        feature count would exceed ``max_features_per_layer``, the
        residential/tertiary results are discarded and only major roads
        are returned.
        """
        from .osm import OverpassClient, QueryBuilder, _osm_to_geojson

        def _line_features(fc: Dict) -> List[Dict]:
            return [
                f for f in fc.get("features", [])
                if f.get("geometry", {}).get("type") in ("LineString", "MultiLineString")
            ]

        major_tags = [
            {"key": "highway", "value": "motorway"},
            {"key": "highway", "value": "trunk"},
            {"key": "highway", "value": "primary"},
            {"key": "highway", "value": "secondary"},
        ]
        residential_tags = [
            {"key": "highway", "value": "tertiary"},
            {"key": "highway", "value": "residential"},
        ]

        # ── Major roads (required) ────────────────────────────────────────
        major_features: List[Dict] = []
        try:
            self._rate_limit()
            osm_client = OverpassClient()
            qb = QueryBuilder()
            qb.geometry(aoi_geometry).tags(major_tags)
            major_features = _line_features(_osm_to_geojson(osm_client.query(qb.build())))
            logger.info(f"OSM major roads: {len(major_features)} features")
        except Exception as exc:
            logger.warning(f"OSM major roads query failed: {exc}")
            return []

        # ── Residential / tertiary (optional) ────────────────────────────
        cap = self.config.max_features_per_layer
        try:
            self._rate_limit()
            qb2 = QueryBuilder()
            qb2.geometry(aoi_geometry).tags(residential_tags)
            res_features = _line_features(_osm_to_geojson(
                OverpassClient().query(qb2.build())
            ))
            logger.info(f"OSM residential/tertiary roads: {len(res_features)} features")

            combined = len(major_features) + len(res_features)
            if combined <= cap:
                logger.info(f"OSM roads combined: {combined} features (within cap {cap})")
                return major_features + res_features
            else:
                logger.warning(
                    f"OSM residential roads would bring total to {combined} "
                    f"(cap={cap}) — using major roads only ({len(major_features)} features)"
                )
                return major_features
        except Exception as exc:
            logger.warning(f"OSM residential roads query failed: {exc} — using major roads only")
            return major_features

    # ── OSM fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _osm_fallback(aoi_geometry: Dict) -> Dict:
        """
        Query OSM Overpass for logistics-relevant features as a fallback.

        Returns a GeoJSON FeatureCollection with road, airport, crossing,
        and warehouse features within the AOI.
        """
        logger.info("HDX returned no results — falling back to OSM Overpass for logistics data")

        from .osm import OverpassClient, QueryBuilder, Tag, _osm_to_geojson

        client = OverpassClient()
        qb = QueryBuilder()
        qb.geometry(aoi_geometry).tags(FALLBACK_OSM_LOGISTICS_TAGS)
        try:
            result = client.query(qb.build())
            fc = _osm_to_geojson(result)
            logger.info(f"OSM fallback: {len(fc.get('features', []))} features retrieved")
            return fc
        except Exception as exc:
            logger.warning(f"OSM fallback failed: {exc}")
            return {"type": "FeatureCollection", "features": []}

    # ── main API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        aoi_geometry: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """
        Fetch WFP logistics data, categorised by layer type.

        Searches HDX, downloads resources, converts to GeoJSON, and
        organises into a dict keyed by layer type.  Falls back to OSM if
        no HDX data is found and ``fallback_to_osm=True``.

        Args:
            aoi_geometry: Optional GeoJSON geometry for spatial clipping.
                          If provided, only features intersecting the AOI
                          are returned per layer.
            output_dir:   Directory for downloaded files.
                          Defaults to ``./output/wfp``.

        Returns:
            Dict mapping layer type to GeoJSON FeatureCollection:
            ``{"roads": {...}, "airports": {...}, "crossings": {...},
               "storage": {...}, "other": {...}}``
        """
        output_dir = Path(output_dir) if output_dir else Path("./output/wfp")
        output_dir.mkdir(parents=True, exist_ok=True)

        layers: Dict[str, List[Dict]] = {
            "roads": [], "airports": [], "crossings": [], "storage": [], "other": [],
            "osm_roads": [],
        }

        datasets = self.search_datasets()

        if datasets:
            resources = self._select_resources(datasets)
            dl_dir    = output_dir / "downloads"
            dl_pairs  = self.download_resources(resources, dl_dir)

            for path, res in dl_pairs:
                fc = self._file_to_geojson(path)
                if not fc or not fc.get("features"):
                    continue

                layer = self._classify_layer(
                    res.get("_dataset_title", ""),
                    path.name,
                )
                layers[layer].extend(fc["features"])
                logger.debug(
                    f"  {path.name} → layer='{layer}', "
                    f"{len(fc['features'])} features"
                )
        elif self.config.fallback_to_osm and aoi_geometry:
            fallback_fc = self._osm_fallback(aoi_geometry)
            for feat in fallback_fc.get("features", []):
                props = feat.get("properties", {})
                hw    = props.get("highway", "")
                ae    = props.get("aeroway", "")
                layer = "other"
                if hw:
                    layer = "roads"
                elif ae:
                    layer = "airports"
                elif props.get("barrier") or props.get("amenity") == "customs":
                    layer = "crossings"
                elif props.get("building") == "warehouse":
                    layer = "storage"
                layers[layer].append(feat)

        # Supplement with OSM major roads (motorway, trunk, primary, secondary)
        if aoi_geometry:
            osm_roads = self._query_osm_major_roads(aoi_geometry)
            layers["osm_roads"].extend(osm_roads)

        # Clip to AOI, normalise geometry types for Mapbox, build output
        LINE_LAYERS  = {"roads", "osm_roads"}
        POINT_LAYERS = {"airports", "crossings", "storage"}

        result_layers: Dict[str, Dict] = {}
        for lname, feats in layers.items():
            clipped = self._clip_to_aoi(feats, aoi_geometry) if aoi_geometry else feats
            # Strip null-geometry features first
            clipped = [f for f in clipped if f.get("geometry")]
            # Normalise to the geometry type the Mapbox layer expects
            if lname in LINE_LAYERS:
                clipped = self._to_line_features(clipped)
            elif lname in POINT_LAYERS:
                clipped = self._to_point_features(clipped)
            # Cap to prevent map.html from becoming unloadably large
            cap = self.config.max_features_per_layer
            if cap and len(clipped) > cap:
                logger.warning(
                    f"Layer '{lname}' has {len(clipped)} features — "
                    f"capping to {cap} for map performance"
                )
                clipped = clipped[:cap]
            result_layers[lname] = {"type": "FeatureCollection", "features": clipped}
            if clipped:
                fc_path = output_dir / f"wfp_{lname}.geojson"
                with open(fc_path, "w") as fh:
                    json.dump(result_layers[lname], fh, indent=2)
                logger.info(
                    f"WFP layer '{lname}': {len(clipped)} features → {fc_path.name}"
                )

        total = sum(len(fc.get("features", [])) for fc in result_layers.values())
        logger.info(f"WFP fetch complete: {total} features across {len(result_layers)} layers")
        return result_layers

    @staticmethod
    def _to_line_features(features: List[Dict]) -> List[Dict]:
        """
        Normalise features to LineString/MultiLineString for Mapbox line layers.

        - LineString / MultiLineString → kept as-is
        - Polygon                      → exterior ring converted to LineString
        - MultiPolygon                 → exterior rings converted to MultiLineString
        - Point / other                → dropped
        - null geometry                → dropped
        """
        result = []
        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                continue
            gtype = geom.get("type", "")
            if gtype in ("LineString", "MultiLineString"):
                result.append(feat)
            elif gtype == "Polygon":
                coords = geom.get("coordinates", [[]])
                if coords and coords[0]:
                    result.append({**feat, "geometry": {
                        "type": "LineString",
                        "coordinates": coords[0],
                    }})
            elif gtype == "MultiPolygon":
                rings = [poly[0] for poly in geom.get("coordinates", []) if poly and poly[0]]
                if rings:
                    result.append({**feat, "geometry": {
                        "type": "MultiLineString",
                        "coordinates": rings,
                    }})
        return result

    @staticmethod
    def _to_point_features(features: List[Dict]) -> List[Dict]:
        """
        Normalise features to Point for Mapbox circle layers.

        - Point        → kept as-is
        - Polygon      → centroid computed via shapely (or first coordinate fallback)
        - LineString   → midpoint used
        - null geometry → dropped
        """
        result = []
        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                continue
            gtype = geom.get("type", "")
            if gtype == "Point":
                result.append(feat)
            else:
                try:
                    from shapely.geometry import shape
                    centroid = shape(geom).centroid
                    result.append({**feat, "geometry": {
                        "type": "Point",
                        "coordinates": [centroid.x, centroid.y],
                    }})
                except Exception:
                    # Fallback: dig out the first coordinate
                    coords = geom.get("coordinates", [])
                    pt = None
                    if gtype == "LineString" and coords:
                        mid = coords[len(coords) // 2]
                        pt = mid[:2]
                    elif gtype == "Polygon" and coords and coords[0]:
                        ring = coords[0]
                        mid = ring[len(ring) // 2]
                        pt = mid[:2]
                    if pt:
                        result.append({**feat, "geometry": {
                            "type": "Point",
                            "coordinates": pt,
                        }})
        return result

    @staticmethod
    def _clip_to_aoi(features: List[Dict], aoi_geometry: Dict) -> List[Dict]:
        """Return only features that intersect the AOI."""
        if not aoi_geometry or not features:
            return features

        try:
            from shapely.geometry import shape as shp_shape
            aoi = shp_shape(aoi_geometry)
            return [
                f for f in features
                if f.get("geometry") and aoi.intersects(shp_shape(f["geometry"]))
            ]
        except Exception as exc:
            logger.warning(f"AOI clip failed: {exc} — returning all features")
            return features


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fetch_logistics_data(
    aoi_geometry: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    config: Optional[WFPConfig] = None,
) -> Dict[str, Dict]:
    """
    End-to-end WFP logistics data fetch.

    Args:
        aoi_geometry: GeoJSON geometry for spatial clipping and OSM fallback.
        output_dir:   Root directory for downloaded files and output GeoJSONs.
        config:       Optional :class:`WFPConfig`.

    Returns:
        Dict of layer name → GeoJSON FeatureCollection.
        Keys: ``"roads"``, ``"airports"``, ``"crossings"``, ``"storage"``, ``"other"``.
    """
    client = WFPLogisticsClient(config)
    return client.fetch(aoi_geometry=aoi_geometry, output_dir=output_dir)
