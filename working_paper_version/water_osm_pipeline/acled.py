"""
ACLED (Armed Conflict Location & Event Data) API client.

Queries conflict events for a user-defined AOI and explicit date window.
Authentication uses OAuth (email + password → access token) or a pre-obtained
access token.  Tokens are automatically refreshed when they expire.

Registration: https://acleddata.com/register/
API docs:     https://acleddata.com/api-documentation/getting-started
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from shapely.geometry import Point, shape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

ACLED_EVENT_TYPES: List[str] = [
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Riots",
    "Protests",
    "Strategic developments",
]

# Sub-event types most likely to cause physical infrastructure damage
DEFAULT_VIOLENCE_SUB_EVENTS: List[str] = [
    "Attack",
    "Shelling/artillery/missile attack",
    "Air/drone strike",
    "Mob violence",
    "Remote explosive/landmine/IED",
    "Looting/property destruction",
    "Grenade",
    "Suicide bomb",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ACLEDConfig:
    """
    ACLED API configuration.

    Credentials are read from environment variables if not provided directly.

    Environment variables:
        ACLED_EMAIL       — registered e-mail address
        ACLED_PASSWORD    — account password
        ACLED_ACCESS_TOKEN (optional) — pre-obtained Bearer token
    """
    email: Optional[str] = None
    password: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    base_url: str = "https://acleddata.com/api/acled/read"
    token_url: str = "https://acleddata.com/oauth/token"
    max_retries: int = 3
    retry_delay: float = 2.0
    default_fields: List[str] = field(default_factory=lambda: [
        "event_id_cnty", "event_date", "year",
        "event_type", "sub_event_type",
        "actor1", "actor2",
        "country", "admin1", "admin2", "admin3", "location",
        "latitude", "longitude", "geo_precision",
        "source", "source_scale", "notes", "fatalities", "timestamp",
    ])

    def __post_init__(self):
        if self.email is None:
            self.email = os.environ.get("ACLED_EMAIL")
        if self.password is None:
            self.password = os.environ.get("ACLED_PASSWORD")
        if self.access_token is None:
            self.access_token = os.environ.get("ACLED_ACCESS_TOKEN")

        if not self.email and not self.access_token:
            logger.warning(
                "ACLED credentials not configured. "
                "Set ACLED_EMAIL + ACLED_PASSWORD (or ACLED_ACCESS_TOKEN) in .env"
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ACLEDClient:
    """
    ACLED OAuth API client.

    Accepts an explicit date range (``start_date``, ``end_date``) so that
    the pipeline can work with any custom time window, not just "last N days".

    Example::

        client = ACLEDClient()
        df = client.query(
            geometry=aoi,
            start_date="2023-10-01",
            end_date="2024-02-01",
            filter_violence=True,
        )
        geojson = client.to_geojson(df)
    """

    def __init__(self, config: Optional[ACLEDConfig] = None):
        self.config = config or ACLEDConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "water-osm-pipeline/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self._access_token: Optional[str] = self.config.access_token
        self._refresh_token: Optional[str] = self.config.refresh_token
        self._token_expires_at: Optional[datetime] = None

        if self._access_token:
            # Conservative: assume token expires in 23 h
            self._token_expires_at = datetime.utcnow() + timedelta(hours=23)

    # ── auth ────────────────────────────────────────────────────────────────

    def _authenticate(self) -> str:
        if not self.config.email or not self.config.password:
            raise ValueError(
                "ACLED credentials required. "
                "Set ACLED_EMAIL and ACLED_PASSWORD in .env"
            )
        data = {
            "username": self.config.email,
            "password": self.config.password,
            "grant_type": "password",
            "client_id": "acled",
        }
        resp = self.session.post(self.config.token_url, data=data, timeout=30)
        if resp.status_code == 401:
            raise ValueError("Invalid ACLED credentials")
        resp.raise_for_status()

        token_data = resp.json()
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 86400)
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 3600)

        logger.info("ACLED: authenticated successfully")
        return self._access_token

    def _refresh(self) -> str:
        if not self._refresh_token:
            return self._authenticate()

        data = {
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
            "client_id": "acled",
        }
        try:
            resp = self.session.post(self.config.token_url, data=data, timeout=30)
            if resp.status_code == 401:
                return self._authenticate()
            resp.raise_for_status()
            token_data = resp.json()
            self._access_token = token_data["access_token"]
            if token_data.get("refresh_token"):
                self._refresh_token = token_data["refresh_token"]
            expires_in = token_data.get("expires_in", 86400)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 3600)
            logger.debug("ACLED: token refreshed")
            return self._access_token
        except requests.RequestException:
            return self._authenticate()

    def _get_token(self) -> str:
        if self._access_token and self._token_expires_at:
            if datetime.utcnow() < self._token_expires_at:
                return self._access_token
            return self._refresh()
        if self._refresh_token:
            return self._refresh()
        return self._authenticate()

    # ── query ───────────────────────────────────────────────────────────────

    def query(
        self,
        geometry: Optional[Dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        sub_event_types: Optional[List[str]] = None,
        filter_violence: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query ACLED conflict events.

        Args:
            geometry:       GeoJSON geometry for AOI filtering (bbox extracted).
            start_date:     Inclusive start date, ``YYYY-MM-DD``.
            end_date:       Inclusive end date, ``YYYY-MM-DD``.
            country:        Optional country name filter (e.g. ``"Sudan"``).
            event_types:    Restrict to specific event types.
            sub_event_types: Restrict to specific sub-event types.
            filter_violence: If True, apply :data:`DEFAULT_VIOLENCE_SUB_EVENTS`.
            limit:          Maximum row count returned by the API.

        Returns:
            :class:`pandas.DataFrame` with one row per conflict event,
            or an empty DataFrame if no events are found.
        """
        if filter_violence and sub_event_types is None:
            sub_event_types = DEFAULT_VIOLENCE_SUB_EVENTS

        bbox: Optional[Tuple] = None
        if geometry:
            bbox = shape(geometry).bounds   # (west, south, east, north)

        params = self._build_params(
            start_date=start_date,
            end_date=end_date,
            country=country,
            event_types=event_types,
            sub_event_types=sub_event_types,
            limit=limit,
            bbox=bbox,
        )

        for attempt in range(self.config.max_retries):
            try:
                token = self._get_token()
                resp = self.session.get(
                    self.config.base_url,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=60,
                )

                if resp.status_code == 429:
                    delay = float(resp.headers.get(
                        "Retry-After",
                        self.config.retry_delay * (2 ** attempt),
                    ))
                    logger.warning(f"ACLED rate limited — waiting {delay:.0f}s")
                    time.sleep(delay)
                    continue

                if resp.status_code == 401:
                    logger.warning("ACLED 401 — refreshing token")
                    self._access_token = None
                    self._token_expires_at = None
                    if attempt < self.config.max_retries - 1:
                        continue
                    raise ValueError("ACLED authentication failed")

                resp.raise_for_status()
                data = resp.json()

                if not data.get("success", True):
                    raise RuntimeError(f"ACLED API error: {data.get('error')}")

                events = data.get("data", [])
                if not events:
                    logger.info("ACLED: no conflict events found for this query")
                    return pd.DataFrame()

                df = pd.DataFrame(events)

                # Precise geometry filter (API bbox may be coarser)
                if geometry and not df.empty:
                    df = self._filter_to_geometry(df, geometry)

                logger.info(f"ACLED: retrieved {len(df)} conflict events")
                return df

            except requests.RequestException as exc:
                logger.warning(f"ACLED request failed (attempt {attempt+1}): {exc}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("ACLED: max retries exceeded")

    def _build_params(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        country: Optional[str],
        event_types: Optional[List[str]],
        sub_event_types: Optional[List[str]],
        limit: Optional[int],
        bbox: Optional[Tuple],
    ) -> Dict:
        params: Dict = {"fields": "|".join(self.config.default_fields)}

        if start_date and end_date:
            params["event_date"] = f"{start_date}|{end_date}"
            params["event_date_where"] = "BETWEEN"
        elif start_date:
            params["event_date"] = start_date
            params["event_date_where"] = ">="
        elif end_date:
            params["event_date"] = end_date
            params["event_date_where"] = "<="

        if country:
            params["country"] = country

        if event_types:
            params["event_type"] = "|".join(event_types)

        if sub_event_types:
            params["sub_event_type"] = ":OR:".join(sub_event_types)

        if bbox:
            west, south, east, north = bbox
            params["latitude"] = f"{south}|{north}"
            params["latitude_where"] = "BETWEEN"
            params["longitude"] = f"{west}|{east}"
            params["longitude_where"] = "BETWEEN"

        if limit:
            params["limit"] = limit

        return params

    @staticmethod
    def _filter_to_geometry(df: pd.DataFrame, geometry: Dict) -> pd.DataFrame:
        """Filter events to the precise geometry boundary."""
        if df.empty or "latitude" not in df.columns:
            return df

        geom = shape(geometry)
        df = df.copy()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        mask = df.apply(
            lambda r: geom.contains(Point(r["longitude"], r["latitude"]))
            if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude"))
            else False,
            axis=1,
        )
        return df[mask].reset_index(drop=True)

    # ── GeoJSON export ───────────────────────────────────────────────────────

    def to_geojson(self, df: pd.DataFrame) -> Dict:
        """Convert a query result DataFrame to a GeoJSON FeatureCollection."""
        if df.empty:
            return {"type": "FeatureCollection", "features": []}

        features = []
        for _, row in df.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue

            props = {
                k: (v if pd.notna(v) else None)
                for k, v in row.items()
                if k not in ("latitude", "longitude")
            }
            props["event_source"] = "ACLED"

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            })

        return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def query_conflict_events(
    geometry: Dict,
    start_date: str,
    end_date: str,
    event_types: Optional[List[str]] = None,
    country: Optional[str] = None,
    filter_violence: bool = True,
    config: Optional[ACLEDConfig] = None,
) -> Dict:
    """
    Query ACLED and return a GeoJSON FeatureCollection.

    Args:
        geometry:       AOI as a GeoJSON geometry dict.
        start_date:     Inclusive start date ``YYYY-MM-DD``.
        end_date:       Inclusive end date ``YYYY-MM-DD``.
        event_types:    Optional event type filter.
        country:        Optional country name filter.
        filter_violence: If True, restrict to :data:`DEFAULT_VIOLENCE_SUB_EVENTS`.
        config:         Optional :class:`ACLEDConfig`.

    Returns:
        GeoJSON FeatureCollection of conflict event points.
    """
    client = ACLEDClient(config)
    df = client.query(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        country=country,
        filter_violence=filter_violence,
    )
    return client.to_geojson(df)
