"""
NASA FIRMS (Fire Information for Resource Management System) API client.

Queries thermal anomaly data from VIIRS and MODIS satellites.
API Documentation: https://firms.modaps.eosdis.nasa.gov/api/

Supports:
- VIIRS S-NPP (375m resolution)
- VIIRS NOAA-20 (375m resolution)
- MODIS (1km resolution)
"""

import io
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from shapely.geometry import Point, mapping, shape

logger = logging.getLogger(__name__)


# Available FIRMS data sources
FIRMS_SOURCES = {
    "VIIRS_SNPP_NRT": "VIIRS S-NPP Near Real-Time",
    "VIIRS_SNPP_SP": "VIIRS S-NPP Standard Processing",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20 Near Real-Time",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21 Near Real-Time",
    "MODIS_NRT": "MODIS Near Real-Time",
    "MODIS_SP": "MODIS Standard Processing",
}


@dataclass
class FIRMSConfig:
    """
    NASA FIRMS API configuration.

    Get your MAP_KEY at: https://firms.modaps.eosdis.nasa.gov/api/map_key/

    Attributes:
        api_key: FIRMS MAP_KEY (32 characters). Set via FIRMS_MAP_KEY env var.
        base_url: FIRMS API base URL
        default_source: Default satellite source
        max_days: Maximum days per request (FIRMS limit is 10)
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries (seconds)
        rate_limit_delay: Delay between requests (rate limit: 5000/10min)
    """
    api_key: Optional[str] = None
    base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area"
    default_source: str = "VIIRS_SNPP_NRT"
    max_days: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.5

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("FIRMS_MAP_KEY")
        if not self.api_key:
            logger.warning(
                "FIRMS API key not set. Set FIRMS_MAP_KEY environment variable "
                "or pass api_key parameter. Get your key at: "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
            )


class FIRMSClient:
    """
    NASA FIRMS API client for thermal anomaly detection.

    Queries VIIRS and MODIS fire/thermal data within a geographic area.

    Example:
        client = FIRMSClient()
        df = client.query(geometry, days=7)
        geojson = client.to_geojson(df)
    """

    def __init__(self, config: Optional[FIRMSConfig] = None):
        self.config = config or FIRMSConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "water-pipeline/1.0"
        })
        self._last_request = 0.0

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _build_url(
        self,
        source: str,
        bbox: Tuple[float, float, float, float],
        days: int,
        date: Optional[str] = None
    ) -> str:
        """
        Build FIRMS API URL.

        API format: /api/area/csv/{MAP_KEY}/{source}/{bbox}/{days}/{date}
        """
        if not self.config.api_key:
            raise ValueError(
                "FIRMS API key required. Set FIRMS_MAP_KEY environment variable."
            )

        west, south, east, north = bbox
        bbox_str = f"{west},{south},{east},{north}"

        url = f"{self.config.base_url}/csv/{self.config.api_key}/{source}/{bbox_str}/{days}"

        if date:
            url += f"/{date}"

        return url

    def query(
        self,
        geometry: Dict,
        days: int = 7,
        source: str = None,
        date: Optional[str] = None,
        confidence_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query FIRMS for thermal anomalies within geometry.

        Args:
            geometry: GeoJSON geometry (bbox will be extracted)
            days: Number of days to query (1-10, will chunk if >10)
            source: Data source (default: VIIRS_SNPP_NRT)
            date: End date in YYYY-MM-DD format (default: today)
            confidence_filter: Filter by confidence: "low", "nominal", "high"

        Returns:
            DataFrame with columns:
                - latitude, longitude: Detection coordinates
                - brightness: Brightness temperature (Kelvin)
                - scan, track: Pixel dimensions
                - acq_date, acq_time: Acquisition datetime
                - satellite: Satellite identifier
                - confidence: Detection confidence
                - version: Algorithm version
                - bright_t31: Channel 31 brightness (MODIS)
                - frp: Fire Radiative Power (MW)
                - daynight: Day (D) or Night (N)
        """
        source = source or self.config.default_source

        # Get bounding box from geometry
        geom = shape(geometry)
        bbox = geom.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)

        # Handle requests > 10 days by chunking
        if days > self.config.max_days:
            return self._query_chunked(geometry, days, source, date, confidence_filter)

        url = self._build_url(source, bbox, days, date)

        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"FIRMS request: {source}, {days} days, bbox={bbox}")
                response = self.session.get(url, timeout=60)

                if response.status_code == 429:
                    delay = float(response.headers.get(
                        "Retry-After",
                        self.config.retry_delay * (2 ** attempt)
                    ))
                    logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue

                if response.status_code == 401:
                    raise ValueError("Invalid FIRMS API key")

                response.raise_for_status()

                # Parse CSV response
                if not response.text.strip():
                    logger.info("No thermal detections found")
                    return pd.DataFrame()

                df = pd.read_csv(io.StringIO(response.text))

                # Filter to geometry (bbox may include extra area)
                df = self._filter_to_geometry(df, geometry)

                # Apply confidence filter
                if confidence_filter and "confidence" in df.columns:
                    df = self._filter_confidence(df, confidence_filter)

                logger.info(f"Retrieved {len(df)} thermal detections")
                return df

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Max retries exceeded")

    def _query_chunked(
        self,
        geometry: Dict,
        days: int,
        source: str,
        date: Optional[str],
        confidence_filter: Optional[str]
    ) -> pd.DataFrame:
        """Query in chunks for periods > 10 days."""
        chunks = []
        remaining_days = days

        # Parse end date
        if date:
            end_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            end_date = datetime.utcnow()

        while remaining_days > 0:
            chunk_days = min(remaining_days, self.config.max_days)
            chunk_date = end_date.strftime("%Y-%m-%d")

            df = self.query(
                geometry,
                days=chunk_days,
                source=source,
                date=chunk_date,
                confidence_filter=confidence_filter
            )
            chunks.append(df)

            remaining_days -= chunk_days
            end_date -= timedelta(days=chunk_days)

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks, ignore_index=True)

        # Remove duplicates (same detection across chunk boundaries)
        if not combined.empty and "latitude" in combined.columns:
            combined = combined.drop_duplicates(
                subset=["latitude", "longitude", "acq_date", "acq_time"],
                keep="first"
            )

        return combined

    def _filter_to_geometry(self, df: pd.DataFrame, geometry: Dict) -> pd.DataFrame:
        """Filter detections to exact geometry (not just bbox)."""
        if df.empty or "latitude" not in df.columns:
            return df

        geom = shape(geometry)

        # Only filter if geometry is not a simple bbox
        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            mask = df.apply(
                lambda row: geom.contains(Point(row["longitude"], row["latitude"])),
                axis=1
            )
            return df[mask].reset_index(drop=True)

        return df

    def _filter_confidence(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        """
        Filter by confidence level.

        VIIRS confidence: "low", "nominal", "high"
        MODIS confidence: 0-100 numeric
        """
        if df.empty:
            return df

        col = "confidence"
        if col not in df.columns:
            return df

        # Handle VIIRS text confidence
        if df[col].dtype == object:
            levels = {"low": 0, "nominal": 1, "high": 2}
            min_level = levels.get(level.lower(), 0)
            df = df[df[col].str.lower().map(levels).fillna(0) >= min_level]

        # Handle MODIS numeric confidence
        else:
            thresholds = {"low": 0, "nominal": 30, "high": 80}
            min_conf = thresholds.get(level.lower(), 0)
            df = df[df[col] >= min_conf]

        return df.reset_index(drop=True)

    def query_multiple_sources(
        self,
        geometry: Dict,
        days: int = 7,
        sources: Optional[List[str]] = None,
        confidence_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query multiple VIIRS/MODIS sources and combine results.

        Args:
            geometry: GeoJSON geometry
            days: Number of days
            sources: List of sources (default: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT)
            confidence_filter: Confidence filter level

        Returns:
            Combined DataFrame with source column
        """
        if sources is None:
            sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]

        all_data = []
        for source in sources:
            try:
                df = self.query(
                    geometry,
                    days=days,
                    source=source,
                    confidence_filter=confidence_filter
                )
                if not df.empty:
                    df["data_source"] = source
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to query {source}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (same fire detected by multiple satellites)
        if not combined.empty:
            combined = combined.drop_duplicates(
                subset=["latitude", "longitude", "acq_date"],
                keep="first"
            )

        return combined

    def to_geojson(self, df: pd.DataFrame) -> Dict:
        """
        Convert DataFrame to GeoJSON FeatureCollection.

        Args:
            df: DataFrame from query()

        Returns:
            GeoJSON FeatureCollection
        """
        if df.empty:
            return {"type": "FeatureCollection", "features": []}

        features = []
        for _, row in df.iterrows():
            # Build properties from all columns except lat/lon
            properties = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in row.items()
                if k not in ["latitude", "longitude"]
            }

            # Add detection type
            properties["detection_type"] = "thermal_anomaly"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])]
                },
                "properties": properties
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}


def query_thermal_anomalies(
    geometry: Dict,
    days: int = 7,
    source: str = "VIIRS_SNPP_NRT",
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSConfig] = None
) -> Dict:
    """
    High-level function to query thermal anomalies.

    Args:
        geometry: GeoJSON geometry (polygon, bbox, etc.)
        days: Number of days to query (default: 7)
        source: FIRMS data source
        confidence_filter: Filter by confidence level
        config: Optional FIRMSConfig

    Returns:
        GeoJSON FeatureCollection with thermal detection points
    """
    client = FIRMSClient(config)
    df = client.query(
        geometry,
        days=days,
        source=source,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(df)


def query_thermal_anomalies_multi(
    geometry: Dict,
    days: int = 7,
    sources: Optional[List[str]] = None,
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSConfig] = None
) -> Dict:
    """
    Query thermal anomalies from multiple satellite sources.

    Args:
        geometry: GeoJSON geometry
        days: Number of days to query
        sources: List of FIRMS sources (default: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT)
        confidence_filter: Confidence filter level
        config: Optional FIRMSConfig

    Returns:
        GeoJSON FeatureCollection with thermal detection points
    """
    client = FIRMSClient(config)
    df = client.query_multiple_sources(
        geometry,
        days=days,
        sources=sources,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(df)
