"""
ACLED (Armed Conflict Location & Event Data) API client.

Queries conflict event data including battles, protests, violence against civilians, etc.
API Documentation: https://acleddata.com/resources/general-guides/

Registration required at: https://acleddata.com/register/
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from shapely.geometry import Point, shape

logger = logging.getLogger(__name__)


# ACLED Event Types
ACLED_EVENT_TYPES = [
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Riots",
    "Protests",
    "Strategic developments",
]

# ACLED Sub-Event Types (subset)
ACLED_SUB_EVENT_TYPES = [
    "Armed clash",
    "Government regains territory",
    "Non-state actor overtakes territory",
    "Attack",
    "Sexual violence",
    "Abduction/forced disappearance",
    "Shelling/artillery/missile attack",
    "Air/drone strike",
    "Suicide bomb",
    "Remote explosive/landmine/IED",
    "Grenade",
    "Violent demonstration",
    "Mob violence",
    "Peaceful protest",
    "Protest with intervention",
    "Excessive force against protesters",
    "Headquarters or base established",
    "Agreement",
    "Arrests",
    "Change to group/activity",
    "Disrupted weapons use",
    "Looting/property destruction",
    "Non-violent transfer of territory",
    "Other",
]


@dataclass
class ACLEDConfig:
    """
    ACLED API configuration.

    Register at: https://acleddata.com/register/

    Attributes:
        email: Registered email address (from ACLED_EMAIL env var)
        api_key: ACLED API key (from ACLED_API_KEY env var)
        base_url: ACLED API base URL
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries
        default_fields: Default fields to retrieve
    """
    email: Optional[str] = None
    api_key: Optional[str] = None
    base_url: str = "https://api.acleddata.com/acled/read"
    max_retries: int = 3
    retry_delay: float = 2.0
    default_fields: List[str] = field(default_factory=lambda: [
        "event_id_cnty",
        "event_date",
        "year",
        "event_type",
        "sub_event_type",
        "actor1",
        "actor2",
        "assoc_actor_1",
        "assoc_actor_2",
        "inter1",
        "inter2",
        "interaction",
        "country",
        "admin1",
        "admin2",
        "admin3",
        "location",
        "latitude",
        "longitude",
        "geo_precision",
        "source",
        "source_scale",
        "notes",
        "fatalities",
        "timestamp",
    ])

    def __post_init__(self):
        if self.email is None:
            self.email = os.environ.get("ACLED_EMAIL")
        if self.api_key is None:
            self.api_key = os.environ.get("ACLED_API_KEY")

        if not self.email or not self.api_key:
            logger.warning(
                "ACLED credentials not set. Set ACLED_EMAIL and ACLED_API_KEY "
                "environment variables. Register at: https://acleddata.com/register/"
            )


class ACLEDClient:
    """
    ACLED API client for conflict event data.

    Example:
        client = ACLEDClient()
        df = client.query_within_aoi(geometry, days=7)
        geojson = client.to_geojson(df)
    """

    def __init__(self, config: Optional[ACLEDConfig] = None):
        self.config = config or ACLEDConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "water-pipeline/1.0"
        })

    def _build_params(
        self,
        country: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict:
        """Build API request parameters."""
        if not self.config.email or not self.config.api_key:
            raise ValueError(
                "ACLED credentials required. Set ACLED_EMAIL and ACLED_API_KEY "
                "environment variables."
            )

        params = {
            "key": self.config.api_key,
            "email": self.config.email,
        }

        # Add fields
        fields = fields or self.config.default_fields
        params["fields"] = "|".join(fields)

        # Date filters
        if start_date:
            params["event_date"] = start_date
            params["event_date_where"] = ">="
        if end_date:
            if start_date:
                # Use BETWEEN for date range
                params["event_date"] = f"{start_date}|{end_date}"
                params["event_date_where"] = "BETWEEN"
            else:
                params["event_date"] = end_date
                params["event_date_where"] = "<="

        # Country filter
        if country:
            params["country"] = country

        # Event type filter
        if event_types:
            params["event_type"] = "|".join(event_types)

        # Bounding box filter (ACLED uses latitude/longitude ranges)
        if bbox:
            west, south, east, north = bbox
            params["latitude"] = f"{south}|{north}"
            params["latitude_where"] = "BETWEEN"
            params["longitude"] = f"{west}|{east}"
            params["longitude_where"] = "BETWEEN"

        # Limit
        if limit:
            params["limit"] = limit

        return params

    def query(
        self,
        country: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        geometry: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Query ACLED for conflict events.

        Args:
            country: Country name filter
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            event_types: Filter by event types (see ACLED_EVENT_TYPES)
            fields: Columns to retrieve
            limit: Maximum number of events
            geometry: GeoJSON geometry for spatial filtering (bbox extracted)

        Returns:
            DataFrame with conflict events
        """
        # Extract bbox from geometry
        bbox = None
        if geometry:
            geom = shape(geometry)
            bbox = geom.bounds  # (west, south, east, north)

        params = self._build_params(
            country=country,
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            fields=fields,
            limit=limit,
            bbox=bbox,
        )

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"ACLED request: {params.get('country', 'all')}, "
                           f"{start_date} to {end_date}")

                response = self.session.get(
                    self.config.base_url,
                    params=params,
                    timeout=60
                )

                if response.status_code == 429:
                    delay = float(response.headers.get(
                        "Retry-After",
                        self.config.retry_delay * (2 ** attempt)
                    ))
                    logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue

                if response.status_code == 401:
                    raise ValueError("Invalid ACLED credentials")

                response.raise_for_status()

                data = response.json()

                if not data.get("success", True):
                    error = data.get("error", "Unknown error")
                    raise RuntimeError(f"ACLED API error: {error}")

                events = data.get("data", [])

                if not events:
                    logger.info("No conflict events found")
                    return pd.DataFrame()

                df = pd.DataFrame(events)

                # Filter to exact geometry if provided
                if geometry and not df.empty:
                    df = self._filter_to_geometry(df, geometry)

                logger.info(f"Retrieved {len(df)} conflict events")
                return df

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Max retries exceeded")

    def _filter_to_geometry(self, df: pd.DataFrame, geometry: Dict) -> pd.DataFrame:
        """Filter events to exact geometry bounds."""
        if df.empty or "latitude" not in df.columns:
            return df

        geom = shape(geometry)

        # Convert coordinates to numeric
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Filter to geometry
        mask = df.apply(
            lambda row: geom.contains(Point(row["longitude"], row["latitude"]))
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else False,
            axis=1
        )

        return df[mask].reset_index(drop=True)

    def query_within_aoi(
        self,
        geometry: Dict,
        days: int = 7,
        event_types: Optional[List[str]] = None,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query events within AOI for last N days.

        Args:
            geometry: GeoJSON geometry
            days: Number of days to look back (default: 7)
            event_types: Filter by event types
            country: Optional country filter

        Returns:
            DataFrame with conflict events
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        return self.query(
            geometry=geometry,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            event_types=event_types,
            country=country,
        )

    def query_violence_events(
        self,
        geometry: Dict,
        days: int = 7,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query specifically for violence-related events.

        Filters to Battles, Violence against civilians, and Explosions.

        Args:
            geometry: GeoJSON geometry
            days: Number of days to look back
            country: Optional country filter

        Returns:
            DataFrame with violence events
        """
        violence_types = [
            "Battles",
            "Violence against civilians",
            "Explosions/Remote violence",
        ]

        return self.query_within_aoi(
            geometry=geometry,
            days=days,
            event_types=violence_types,
            country=country,
        )

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
            # Skip rows without valid coordinates
            lat = row.get("latitude")
            lon = row.get("longitude")

            if pd.isna(lat) or pd.isna(lon):
                continue

            # Build properties
            properties = {
                k: (v if pd.notna(v) else None)
                for k, v in row.items()
                if k not in ["latitude", "longitude"]
            }

            # Add event metadata
            properties["event_source"] = "ACLED"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "properties": properties
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}


def query_conflict_events(
    geometry: Dict,
    days: int = 7,
    event_types: Optional[List[str]] = None,
    country: Optional[str] = None,
    config: Optional[ACLEDConfig] = None,
) -> Dict:
    """
    High-level function to query conflict events.

    Args:
        geometry: GeoJSON geometry
        days: Number of days to look back (default: 7)
        event_types: Filter by event types
        country: Optional country filter
        config: Optional ACLEDConfig

    Returns:
        GeoJSON FeatureCollection with conflict event points
    """
    client = ACLEDClient(config)
    df = client.query_within_aoi(
        geometry=geometry,
        days=days,
        event_types=event_types,
        country=country,
    )
    return client.to_geojson(df)


def query_violence_events(
    geometry: Dict,
    days: int = 7,
    country: Optional[str] = None,
    config: Optional[ACLEDConfig] = None,
) -> Dict:
    """
    Query violence-related events (battles, civilian violence, explosions).

    Args:
        geometry: GeoJSON geometry
        days: Number of days to look back
        country: Optional country filter
        config: Optional ACLEDConfig

    Returns:
        GeoJSON FeatureCollection
    """
    client = ACLEDClient(config)
    df = client.query_violence_events(
        geometry=geometry,
        days=days,
        country=country,
    )
    return client.to_geojson(df)
