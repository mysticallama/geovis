"""
NASA FIRMS thermal anomaly data via GIBS (Global Imagery Browse Services).

Queries thermal anomaly data from VIIRS and MODIS satellites using NASA GIBS WMTS.
No API key required!

GIBS Documentation: https://nasa-gibs.github.io/gibs-api-docs/
Available layers: https://nasa-gibs.github.io/gibs-api-docs/available-visualizations/

Data Products:
- VIIRS NOAA-20/21 Thermal Anomalies 375m (gridded from VNP14IMG 750m L2 swath)
- VIIRS SNPP Thermal Anomalies 375m (gridded from VNP14IMG 750m L2 swath)
- MODIS Terra/Aqua Thermal Anomalies 1km

Note: The source data is "VIIRS/NPP Thermal Anomalies/Fire 6-Min L2 Swath 750m V002"
(VNP14IMG). GIBS serves this as a 375m gridded visualization layer for efficient
tile-based access. The 375m resolution uses the I-band for higher spatial detail.
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from shapely.geometry import Point, box, shape

logger = logging.getLogger(__name__)

# Try to import mapbox_vector_tile for MVT decoding
try:
    import mapbox_vector_tile
    HAS_MVT = True
except ImportError:
    HAS_MVT = False
    logger.warning(
        "mapbox_vector_tile not installed. Install with: pip install mapbox-vector-tile"
    )


# Available GIBS thermal anomaly layers
# Note: GIBS serves gridded visualization products, not raw swath data.
# Primary product: VIIRS_NOAA20_Thermal_Anomalies_375m_All (VJ114IMGT_NRT 2)
GIBS_THERMAL_LAYERS = {
    # Primary product - VIIRS NOAA-20 (VJ114IMGT_NRT 2)
    "VIIRS_NOAA20": "VIIRS_NOAA20_Thermal_Anomalies_375m_All",
    # Additional layers (available but not default)
    "VIIRS_NOAA20_Day": "VIIRS_NOAA20_Thermal_Anomalies_375m_Day",
    "VIIRS_NOAA20_Night": "VIIRS_NOAA20_Thermal_Anomalies_375m_Night",
    "VIIRS_NOAA21": "VIIRS_NOAA21_Thermal_Anomalies_375m_All",
    "VIIRS_SNPP": "VIIRS_SNPP_Thermal_Anomalies_375m_All",
    "MODIS_Terra": "MODIS_Terra_Thermal_Anomalies_All",
    "MODIS_Aqua": "MODIS_Aqua_Thermal_Anomalies_All",
}

# Tile matrix sets for different resolutions
# GIBS EPSG:4326 tile matrix zoom levels
TILE_MATRIX_SETS = {
    "250m": {"id": "250m", "zoom_levels": 6},
    "500m": {"id": "500m", "zoom_levels": 5},
    "1km": {"id": "1km", "zoom_levels": 4},
    "2km": {"id": "2km", "zoom_levels": 3},
}


@dataclass
class FIRMSConfig:
    """
    NASA GIBS configuration for thermal anomaly data.

    No API key required! GIBS is freely accessible.

    Documentation: https://nasa-gibs.github.io/gibs-api-docs/

    Attributes:
        base_url: GIBS WMTS base URL
        default_layer: Default thermal anomaly layer
        tile_matrix_set: Resolution (250m, 500m, 1km, 2km)
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries
        rate_limit_delay: Delay between requests
    """
    base_url: str = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
    default_layer: str = "VIIRS_NOAA20"
    tile_matrix_set: str = "500m"
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.2
    # Keep api_key for backward compatibility but it's not needed
    api_key: Optional[str] = None

    def __post_init__(self):
        # For backward compatibility, check env var but don't require it
        if self.api_key is None:
            self.api_key = os.environ.get("FIRMS_MAP_KEY")


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates for EPSG:4326."""
    # GIBS uses EPSG:4326 with specific tile matrix
    # At zoom 0, there are 2 tiles horizontally (-180 to 0, 0 to 180)
    n = 2 ** zoom

    # X tile (column)
    x = int((lon + 180.0) / 360.0 * n)

    # Y tile (row) - EPSG:4326 goes from 90 to -90
    y = int((90.0 - lat) / 180.0 * n)

    return max(0, min(x, n * 2 - 1)), max(0, min(y, n - 1))


def _get_tiles_for_bbox(
    west: float, south: float, east: float, north: float, zoom: int
) -> List[Tuple[int, int]]:
    """Get all tile coordinates that cover the bounding box."""
    min_col, max_row = _lat_lon_to_tile(north, west, zoom)
    max_col, min_row = _lat_lon_to_tile(south, east, zoom)

    tiles = []
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            tiles.append((row, col))

    return tiles


class FIRMSClient:
    """
    NASA GIBS client for thermal anomaly data.

    Uses WMTS vector tiles - no API key required!

    Example:
        client = FIRMSClient()
        detections = client.query(geometry, days=3)
        geojson = client.to_geojson(detections)
    """

    def __init__(self, config: Optional[FIRMSConfig] = None):
        self.config = config or FIRMSConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "water-pipeline/1.0",
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request = 0.0

        if not HAS_MVT:
            logger.warning(
                "mapbox_vector_tile not available. Install with: "
                "pip install mapbox-vector-tile"
            )

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _get_layer_name(self, source: str) -> str:
        """Get full GIBS layer name from short source name."""
        if source in GIBS_THERMAL_LAYERS:
            return GIBS_THERMAL_LAYERS[source]
        # Check if it's already a full layer name
        if source in GIBS_THERMAL_LAYERS.values():
            return source
        # Default to VIIRS NOAA-20
        return GIBS_THERMAL_LAYERS.get(
            self.config.default_layer,
            "VIIRS_NOAA20_Thermal_Anomalies_375m_All"
        )

    def _fetch_tile(
        self,
        layer: str,
        date: str,
        zoom: int,
        row: int,
        col: int
    ) -> List[Dict]:
        """Fetch a single MVT tile and decode features."""
        if not HAS_MVT:
            raise RuntimeError(
                "mapbox_vector_tile required. Install with: pip install mapbox-vector-tile"
            )

        # Build WMTS REST URL
        # Format: {base}/{layer}/default/{date}/{tilematrixset}/{zoom}/{row}/{col}.mvt
        url = (
            f"{self.config.base_url}/{layer}/default/{date}T00:00:00Z/"
            f"{self.config.tile_matrix_set}/{zoom}/{row}/{col}.mvt"
        )

        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=30)

                if response.status_code == 404:
                    # No data for this tile/date
                    return []

                if response.status_code == 400:
                    # Bad request - usually means date is outside GIBS archive
                    # GIBS typically only retains ~30-90 days of NRT data
                    return []

                if response.status_code == 429:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue

                response.raise_for_status()

                if not response.content:
                    return []

                # Decode MVT
                decoded = mapbox_vector_tile.decode(response.content)

                features = []
                for layer_name, layer_data in decoded.items():
                    for feature in layer_data.get("features", []):
                        features.append(feature)

                return features

            except requests.RequestException as e:
                logger.warning(f"Tile fetch failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue

        return []

    def _mvt_to_geojson_features(
        self,
        mvt_features: List[Dict],
        tile_row: int,
        tile_col: int,
        zoom: int,
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """Convert MVT features to GeoJSON, filtering to bbox."""
        geojson_features = []

        # MVT coordinates are in tile-local space (0-4096 typically)
        # Need to convert to lat/lon based on tile position
        extent = 4096  # Standard MVT extent

        # Calculate tile bounds in EPSG:4326
        n = 2 ** zoom
        tile_width = 360.0 / (n * 2)  # Width in degrees
        tile_height = 180.0 / n  # Height in degrees

        tile_west = -180.0 + tile_col * tile_width
        tile_north = 90.0 - tile_row * tile_height

        west, south, east, north = bbox
        bbox_geom = box(west, south, east, north)

        for feature in mvt_features:
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})

            if geom.get("type") != "Point":
                continue

            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            # Convert tile coordinates to lat/lon
            x, y = coords[0], coords[1]
            lon = tile_west + (x / extent) * tile_width
            lat = tile_north - (y / extent) * tile_height

            # Filter to bbox
            point = Point(lon, lat)
            if not bbox_geom.contains(point):
                continue

            # Build GeoJSON feature
            gj_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    **props,
                    "detection_type": "thermal_anomaly",
                    "latitude": lat,
                    "longitude": lon,
                }
            }
            geojson_features.append(gj_feature)

        return geojson_features

    def query(
        self,
        geometry: Dict,
        days: int = 3,
        source: str = None,
        date: Optional[str] = None,
        confidence_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query GIBS for thermal anomalies within geometry.

        Args:
            geometry: GeoJSON geometry (bbox will be extracted)
            days: Number of days to query (default: 3)
            source: Data source (VIIRS_NOAA20, VIIRS_SNPP, MODIS_Terra, etc.)
            date: End date in YYYY-MM-DD format (default: yesterday)
            confidence_filter: Filter by confidence (not yet implemented for GIBS)

        Returns:
            List of GeoJSON feature dicts
        """
        if not HAS_MVT:
            raise RuntimeError(
                "mapbox_vector_tile required for GIBS queries. "
                "Install with: pip install mapbox-vector-tile"
            )

        layer = self._get_layer_name(source or self.config.default_layer)
        logger.info(f"Querying GIBS layer: {layer}")

        # Get bounding box from geometry
        geom = shape(geometry)
        bbox = geom.bounds  # (west, south, east, north)

        # Determine zoom level based on tile matrix set
        tms = TILE_MATRIX_SETS.get(self.config.tile_matrix_set, TILE_MATRIX_SETS["500m"])
        zoom = tms["zoom_levels"]

        # Get tiles covering the bbox
        tiles = _get_tiles_for_bbox(*bbox, zoom)

        # Parse end date
        if date:
            end_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            # Use yesterday (today's data may not be available yet)
            end_date = datetime.utcnow() - timedelta(days=1)

        all_features = []

        # Query each day
        for day_offset in range(days):
            query_date = end_date - timedelta(days=day_offset)
            date_str = query_date.strftime("%Y-%m-%d")

            logger.debug(f"Querying {layer} for {date_str}, {len(tiles)} tiles")

            # Fetch all tiles for this day
            for row, col in tiles:
                try:
                    mvt_features = self._fetch_tile(layer, date_str, zoom, row, col)

                    if mvt_features:
                        gj_features = self._mvt_to_geojson_features(
                            mvt_features, row, col, zoom, bbox
                        )

                        # Add date to features
                        for f in gj_features:
                            f["properties"]["acq_date"] = date_str

                        all_features.extend(gj_features)

                except Exception as e:
                    logger.warning(f"Error fetching tile {row}/{col}: {e}")

        # Remove duplicates based on coordinates
        seen = set()
        unique_features = []
        for f in all_features:
            coords = tuple(f["geometry"]["coordinates"])
            date = f["properties"].get("acq_date", "")
            key = (coords, date)
            if key not in seen:
                seen.add(key)
                unique_features.append(f)

        logger.info(f"Retrieved {len(unique_features)} thermal detections from GIBS")
        return unique_features

    def query_multiple_sources(
        self,
        geometry: Dict,
        days: int = 3,
        sources: Optional[List[str]] = None,
        confidence_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query multiple VIIRS/MODIS sources and combine results.

        Args:
            geometry: GeoJSON geometry
            days: Number of days
            sources: List of sources (default: VIIRS_NOAA20, VIIRS_SNPP)
            confidence_filter: Confidence filter level

        Returns:
            Combined list of GeoJSON features
        """
        if sources is None:
            sources = ["VIIRS_NOAA20", "VIIRS_SNPP"]

        all_features = []
        for source in sources:
            try:
                features = self.query(
                    geometry,
                    days=days,
                    source=source,
                    confidence_filter=confidence_filter
                )
                for f in features:
                    f["properties"]["data_source"] = source
                all_features.extend(features)
            except Exception as e:
                logger.warning(f"Failed to query {source}: {e}")

        # Remove duplicates
        seen = set()
        unique_features = []
        for f in all_features:
            coords = tuple(f["geometry"]["coordinates"])
            date = f["properties"].get("acq_date", "")
            key = (coords, date)
            if key not in seen:
                seen.add(key)
                unique_features.append(f)

        return unique_features

    def to_geojson(self, features: List[Dict]) -> Dict:
        """
        Convert feature list to GeoJSON FeatureCollection.

        Args:
            features: List of GeoJSON feature dicts

        Returns:
            GeoJSON FeatureCollection
        """
        return {
            "type": "FeatureCollection",
            "features": features
        }


def query_thermal_anomalies(
    geometry: Dict,
    days: int = 3,
    source: str = "VIIRS_NOAA20",
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSConfig] = None
) -> Dict:
    """
    High-level function to query thermal anomalies via GIBS.

    No API key required!

    Args:
        geometry: GeoJSON geometry (polygon, bbox, etc.)
        days: Number of days to query (default: 3)
        source: GIBS layer source (VIIRS_NOAA20, VIIRS_SNPP, MODIS_Terra, etc.)
        confidence_filter: Filter by confidence level (not yet implemented)
        config: Optional FIRMSConfig

    Returns:
        GeoJSON FeatureCollection with thermal detection points
    """
    client = FIRMSClient(config)
    features = client.query(
        geometry,
        days=days,
        source=source,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(features)


def query_thermal_anomalies_multi(
    geometry: Dict,
    days: int = 3,
    sources: Optional[List[str]] = None,
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSConfig] = None
) -> Dict:
    """
    Query thermal anomalies from multiple satellite sources via GIBS.

    No API key required!

    Args:
        geometry: GeoJSON geometry
        days: Number of days to query
        sources: List of sources (default: VIIRS_NOAA20, VIIRS_SNPP)
        confidence_filter: Confidence filter level
        config: Optional FIRMSConfig

    Returns:
        GeoJSON FeatureCollection with thermal detection points
    """
    client = FIRMSClient(config)
    features = client.query_multiple_sources(
        geometry,
        days=days,
        sources=sources,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(features)


# Backward compatibility aliases
FIRMS_SOURCES = GIBS_THERMAL_LAYERS
