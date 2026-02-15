"""
NASA FIRMS REST API client for thermal anomaly/fire detection data.

Queries fire data directly from NASA FIRMS using their REST API.
Requires a MAP_KEY from: https://firms.modaps.eosdis.nasa.gov/api/map_key/

API Documentation: https://firms.modaps.eosdis.nasa.gov/api/area/
Data Academy: https://firms.modaps.eosdis.nasa.gov/academy/data_api/

Data Products (Standard Processing - up to 365 days archive):
- VIIRS NOAA-20 (VJ114IMGTDL) - 375m resolution, science quality
- VIIRS SNPP (VNP14IMGTDL) - 375m resolution, science quality
- MODIS (MODIS_SP) - 1km resolution, science quality

Note: This client defaults to Standard Processing (SP) data for longer archive access.
NRT sources are also available for recent data with ~3 hour latency.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from shapely.geometry import Point, box, shape

logger = logging.getLogger(__name__)


# Available FIRMS data sources
# SP = Standard Processing (science quality, up to 365 days archive)
# NRT = Near Real-Time (~3 hour latency, ~60 days archive)
FIRMS_SOURCES = {
    # Standard processing VIIRS (375m resolution, up to 365 days)
    "VIIRS_NOAA20_SP": "VIIRS_NOAA20_SP",
    "VIIRS_SNPP_SP": "VIIRS_SNPP_SP",
    # Standard processing MODIS (1km resolution)
    "MODIS_SP": "MODIS_SP",
    # NRT sources (shorter archive, ~60 days)
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT": "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT": "VIIRS_SNPP_NRT",
    "MODIS_NRT": "MODIS_NRT",
    # Shorthand aliases (default to SP for longer archive)
    "VIIRS_NOAA20": "VIIRS_NOAA20_SP",
    "VIIRS_SNPP": "VIIRS_SNPP_SP",
    "MODIS": "MODIS_SP",
}

# Data availability windows (approximate)
DATA_AVAILABILITY = {
    # Standard Processing - up to 1 year archive
    "VIIRS_NOAA20_SP": 365,
    "VIIRS_SNPP_SP": 365,
    "MODIS_SP": 365,
    # NRT - ~60 days rolling window
    "VIIRS_NOAA20_NRT": 60,
    "VIIRS_NOAA21_NRT": 60,
    "VIIRS_SNPP_NRT": 60,
    "MODIS_NRT": 60,
}

# FIRMS API limit: max 10 days per request for area (bounding box) queries
# For longer periods, we automatically split into multiple requests
MAX_DAYS_PER_REQUEST = 10


@dataclass
class FIRMSAPIConfig:
    """
    NASA FIRMS REST API configuration.

    Get your MAP_KEY at: https://firms.modaps.eosdis.nasa.gov/api/map_key/

    Attributes:
        map_key: NASA FIRMS API key (required)
        base_url: FIRMS API base URL
        default_source: Default data source (VIIRS_NOAA20_SP for longer archive)
        output_format: Response format (csv or json)
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries (seconds)
        rate_limit_delay: Delay between requests (seconds)
    """
    map_key: Optional[str] = None
    base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area"
    default_source: str = "VIIRS_NOAA20_SP"
    output_format: str = "csv"  # csv is more reliable than json
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 0.5

    def __post_init__(self):
        if self.map_key is None:
            self.map_key = os.environ.get("FIRMS_MAP_KEY")
        if not self.map_key:
            raise ValueError(
                "FIRMS MAP_KEY required. Get one at: "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/ "
                "Then set FIRMS_MAP_KEY environment variable or pass map_key parameter."
            )


class FIRMSAPIClient:
    """
    NASA FIRMS REST API client for fire/thermal anomaly data.

    Provides access to VIIRS and MODIS active fire detections via
    NASA's FIRMS (Fire Information for Resource Management System).

    Example:
        client = FIRMSAPIClient()  # Uses FIRMS_MAP_KEY env var
        detections = client.query(geometry, days=7)
        geojson = client.to_geojson(detections)

    Example with explicit key:
        config = FIRMSAPIConfig(map_key="your-32-char-key")
        client = FIRMSAPIClient(config)
    """

    def __init__(self, config: Optional[FIRMSAPIConfig] = None):
        self.config = config or FIRMSAPIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "water-pipeline/1.0",
            "Accept": "text/csv,application/json",
        })
        self._last_request = 0.0

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _generate_date_chunks(
        self,
        total_days: int,
        end_date: Optional[str] = None
    ) -> List[Tuple[int, str]]:
        """
        Generate date chunks for queries exceeding MAX_DAYS_PER_REQUEST.

        FIRMS API limits area queries to 10 days per request. For longer
        periods, we split into multiple requests working backwards from
        the end date.

        Args:
            total_days: Total number of days to query
            end_date: End date string (YYYY-MM-DD) or None for today

        Returns:
            List of (days, end_date_str) tuples for each chunk
        """
        # Parse end date
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()

        chunks = []
        remaining_days = total_days
        current_end = end_dt

        while remaining_days > 0:
            # Determine chunk size (max 10 days)
            chunk_days = min(remaining_days, MAX_DAYS_PER_REQUEST)
            chunk_end_str = current_end.strftime("%Y-%m-%d")

            chunks.append((chunk_days, chunk_end_str))

            # Move to next chunk
            remaining_days -= chunk_days
            current_end = current_end - timedelta(days=chunk_days)

        logger.info(
            f"Split {total_days}-day query into {len(chunks)} chunks: "
            f"{[(d, e) for d, e in chunks]}"
        )

        return chunks

    def _get_source_name(self, source: str) -> str:
        """Resolve source name alias to full source identifier."""
        if source in FIRMS_SOURCES:
            return FIRMS_SOURCES[source]
        # Check if already a valid source
        if source in FIRMS_SOURCES.values():
            return source
        # Default to SP for longer archive access
        return FIRMS_SOURCES.get(
            self.config.default_source,
            "VIIRS_NOAA20_SP"
        )

    def _build_url(
        self,
        source: str,
        bbox: Tuple[float, float, float, float],
        days: int,
        date: Optional[str] = None
    ) -> str:
        """
        Build FIRMS API URL.

        URL format: {base}/{format}/{key}/{source}/{area}/{days}/{date}
        Area format: west,south,east,north
        """
        west, south, east, north = bbox
        area = f"{west},{south},{east},{north}"

        # Build URL path
        url = (
            f"{self.config.base_url}/"
            f"{self.config.output_format}/"
            f"{self.config.map_key}/"
            f"{source}/"
            f"{area}/"
            f"{days}"
        )

        # Add date if specified (for historical queries)
        if date:
            url += f"/{date}"

        return url

    def _parse_csv_response(self, csv_text: str) -> pd.DataFrame:
        """Parse CSV response into DataFrame."""
        if not csv_text or csv_text.strip() == "":
            return pd.DataFrame()

        try:
            df = pd.read_csv(StringIO(csv_text))
            return df
        except Exception as e:
            logger.warning(f"Failed to parse CSV response: {e}")
            return pd.DataFrame()

    def _fetch_data(
        self,
        source: str,
        bbox: Tuple[float, float, float, float],
        days: int,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch fire data from FIRMS API."""
        url = self._build_url(source, bbox, days, date)
        logger.debug(f"FIRMS API request: {url}")

        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=60)

                if response.status_code == 200:
                    return self._parse_csv_response(response.text)

                if response.status_code == 400:
                    # Bad request - check error message
                    logger.warning(f"FIRMS API bad request: {response.text}")
                    return pd.DataFrame()

                if response.status_code == 401:
                    raise ValueError(
                        "Invalid FIRMS MAP_KEY. Get a new one at: "
                        "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
                    )

                if response.status_code == 429:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue

                if response.status_code == 404:
                    # No data for this query
                    return pd.DataFrame()

                response.raise_for_status()

            except requests.RequestException as e:
                logger.warning(f"FIRMS request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        return pd.DataFrame()

    def query(
        self,
        geometry: Dict,
        days: int = 7,
        source: Optional[str] = None,
        date: Optional[str] = None,
        confidence_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query FIRMS for fire/thermal anomaly detections.

        Note: FIRMS API limits area queries to 10 days per request. For longer
        periods (e.g., 30 days), this method automatically splits the query
        into multiple 10-day chunks and combines the results.

        Args:
            geometry: GeoJSON geometry (bbox will be extracted)
            days: Number of days to query (up to 365 for SP sources)
            source: Data source (VIIRS_NOAA20_SP, VIIRS_SNPP_SP, MODIS_SP, etc.)
            date: End date in YYYY-MM-DD format (default: today)
            confidence_filter: Filter by confidence level:
                - "h" or "high": High confidence only
                - "n" or "nominal": Nominal and high confidence
                - "l" or "low": All detections (default)

        Returns:
            List of GeoJSON feature dicts with fire detection properties
        """
        source_name = self._get_source_name(source or self.config.default_source)
        logger.info(f"Querying FIRMS API: source={source_name}, days={days}, end_date={date}")

        # Extract bounding box from geometry
        geom = shape(geometry)
        bbox = geom.bounds  # (west, south, east, north)

        # Check if we need to split into chunks (API limit: 10 days per request)
        if days <= MAX_DAYS_PER_REQUEST:
            # Single request
            df = self._fetch_data(source_name, bbox, days, date)
        else:
            # Split into multiple requests
            logger.info(f"Query exceeds {MAX_DAYS_PER_REQUEST}-day limit, splitting into chunks...")
            chunks = self._generate_date_chunks(days, date)

            all_dfs = []
            for chunk_days, chunk_end in chunks:
                logger.info(f"  Fetching chunk: {chunk_days} days ending {chunk_end}")
                chunk_df = self._fetch_data(source_name, bbox, chunk_days, chunk_end)
                if not chunk_df.empty:
                    all_dfs.append(chunk_df)

            # Combine all chunks
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
                # Remove duplicates (in case of overlap at chunk boundaries)
                if 'latitude' in df.columns and 'longitude' in df.columns and 'acq_date' in df.columns:
                    df = df.drop_duplicates(
                        subset=['latitude', 'longitude', 'acq_date', 'acq_time'],
                        keep='first'
                    )
            else:
                df = pd.DataFrame()

        if df.empty:
            logger.info("No fire detections found")
            return []

        logger.info(f"Retrieved {len(df)} raw detections from FIRMS")

        # Apply confidence filter if specified
        if confidence_filter and "confidence" in df.columns:
            conf = confidence_filter.lower()
            if conf in ("h", "high"):
                df = df[df["confidence"] == "h"]
            elif conf in ("n", "nominal"):
                df = df[df["confidence"].isin(["h", "n"])]
            # "l" or "low" keeps all

            logger.info(f"After confidence filter: {len(df)} detections")

        # Convert to GeoJSON features
        features = self._dataframe_to_features(df, source_name, bbox)

        logger.info(f"Returning {len(features)} fire detections")
        return features

    def _dataframe_to_features(
        self,
        df: pd.DataFrame,
        source: str,
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """Convert DataFrame to GeoJSON features with bbox filtering."""
        features = []
        west, south, east, north = bbox
        bbox_geom = box(west, south, east, north)

        for _, row in df.iterrows():
            # Get coordinates
            lat = row.get("latitude")
            lon = row.get("longitude")

            if pd.isna(lat) or pd.isna(lon):
                continue

            # Verify point is within bbox (API sometimes returns nearby points)
            point = Point(lon, lat)
            if not bbox_geom.contains(point):
                continue

            # Build properties dict from all columns
            properties = {
                "detection_type": "thermal_anomaly",
                "data_source": source,
                "latitude": float(lat),
                "longitude": float(lon),
            }

            # Map standard FIRMS columns
            column_mapping = {
                "acq_date": "acq_date",
                "acq_time": "acq_time",
                "bright_ti4": "brightness_ti4",
                "bright_ti5": "brightness_ti5",
                "brightness": "brightness",
                "scan": "scan",
                "track": "track",
                "satellite": "satellite",
                "instrument": "instrument",
                "confidence": "confidence",
                "version": "version",
                "frp": "frp",  # Fire Radiative Power (MW)
                "daynight": "daynight",
                "type": "fire_type",
            }

            for csv_col, prop_name in column_mapping.items():
                if csv_col in row and pd.notna(row[csv_col]):
                    properties[prop_name] = row[csv_col]

            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "properties": properties
            }

            features.append(feature)

        return features

    def query_multiple_sources(
        self,
        geometry: Dict,
        days: int = 7,
        sources: Optional[List[str]] = None,
        date: Optional[str] = None,
        confidence_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query multiple VIIRS/MODIS sources and combine results.

        Args:
            geometry: GeoJSON geometry
            days: Number of days to query
            sources: List of sources (default: VIIRS_NOAA20_SP, VIIRS_SNPP_SP)
            date: End date
            confidence_filter: Confidence filter level

        Returns:
            Combined list of GeoJSON features (deduplicated)
        """
        if sources is None:
            sources = ["VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"]

        all_features = []
        for source in sources:
            try:
                features = self.query(
                    geometry,
                    days=days,
                    source=source,
                    date=date,
                    confidence_filter=confidence_filter
                )
                all_features.extend(features)
            except Exception as e:
                logger.warning(f"Failed to query {source}: {e}")

        # Remove duplicates based on coordinates and date
        seen = set()
        unique_features = []
        for f in all_features:
            coords = tuple(f["geometry"]["coordinates"])
            acq_date = f["properties"].get("acq_date", "")
            acq_time = f["properties"].get("acq_time", "")
            key = (coords, acq_date, acq_time)
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

    def get_data_availability(self, source: Optional[str] = None) -> Dict:
        """
        Check data availability for a source.

        Returns dict with max_days and description.
        """
        source_name = self._get_source_name(source or self.config.default_source)
        max_days = DATA_AVAILABILITY.get(source_name, 60)

        return {
            "source": source_name,
            "max_days": max_days,
            "description": f"{source_name} data available for up to {max_days} days"
        }


# Convenience functions for direct use

def query_firms_thermal_anomalies(
    geometry: Dict,
    days: int = 7,
    source: str = "VIIRS_NOAA20_SP",
    date: Optional[str] = None,
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSAPIConfig] = None
) -> Dict:
    """
    High-level function to query fire detections via FIRMS REST API.

    Requires FIRMS_MAP_KEY environment variable or config.map_key.

    Args:
        geometry: GeoJSON geometry (polygon, bbox, etc.)
        days: Number of days to query (default: 7, up to 365 for SP sources)
        source: FIRMS source (VIIRS_NOAA20_SP, VIIRS_SNPP_SP, MODIS_SP for archive)
        date: End date in YYYY-MM-DD format
        confidence_filter: Filter by confidence (h/high, n/nominal, l/low)
        config: Optional FIRMSAPIConfig

    Returns:
        GeoJSON FeatureCollection with fire detection points
    """
    client = FIRMSAPIClient(config)
    features = client.query(
        geometry,
        days=days,
        source=source,
        date=date,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(features)


def query_firms_multi_source(
    geometry: Dict,
    days: int = 7,
    sources: Optional[List[str]] = None,
    date: Optional[str] = None,
    confidence_filter: Optional[str] = None,
    config: Optional[FIRMSAPIConfig] = None
) -> Dict:
    """
    Query fire detections from multiple satellite sources.

    Args:
        geometry: GeoJSON geometry
        days: Number of days to query (up to 365 for SP sources)
        sources: List of sources (default: VIIRS_NOAA20_SP, VIIRS_SNPP_SP)
        date: End date
        confidence_filter: Confidence filter
        config: Optional FIRMSAPIConfig

    Returns:
        GeoJSON FeatureCollection with combined fire detections
    """
    client = FIRMSAPIClient(config)
    features = client.query_multiple_sources(
        geometry,
        days=days,
        sources=sources,
        date=date,
        confidence_filter=confidence_filter
    )
    return client.to_geojson(features)
