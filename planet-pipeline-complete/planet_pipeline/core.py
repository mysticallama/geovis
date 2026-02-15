"""
Core module for Planet Labs imagery pipeline.

Handles API queries, downloads, and storage in a unified interface.
"""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlanetConfig:
    """Configuration for Planet API client."""
    api_key: Optional[str] = None
    base_url: str = "https://api.planet.com/data/v1"
    max_retries: int = 3
    retry_delay: float = 2.0
    download_workers: int = 2
    chunk_size: int = 8192
    activation_timeout: int = 600

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("PL_API_KEY")
        if not self.api_key:
            raise ValueError("Planet API key required. Set PL_API_KEY or pass api_key.")


# =============================================================================
# Storage Manager
# =============================================================================

class Storage:
    """
    Manages organized storage for imagery and derived products.

    Structure:
        base_dir/
        ├── aois/{name}/geometry.geojson, metadata.json
        ├── imagery/{aoi}/{date}/{item_id}/{asset}.tif
        ├── processed/{aoi}/
        ├── indices/{aoi}/
        ├── labels/{aoi}/
        └── exports/
    """

    def __init__(self, base_dir: Union[str, Path] = "./planet_data"):
        self.base_dir = Path(base_dir)
        self._setup()

    def _setup(self) -> None:
        for d in ["aois", "imagery", "processed", "indices", "labels", "exports"]:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)

    def save_aoi(self, name: str, geometry: Dict, metadata: Optional[Dict] = None) -> Path:
        """Save AOI geometry and metadata."""
        aoi_dir = self.base_dir / "aois" / name
        aoi_dir.mkdir(parents=True, exist_ok=True)

        # Save geometry
        geojson_path = aoi_dir / "geometry.geojson"
        with open(geojson_path, 'w') as f:
            json.dump({
                "type": "Feature",
                "properties": {"name": name, **(metadata or {})},
                "geometry": geometry
            }, f, indent=2)

        # Save metadata
        with open(aoi_dir / "metadata.json", 'w') as f:
            json.dump({
                "name": name,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }, f, indent=2)

        return geojson_path

    def load_aoi(self, name: str) -> Optional[Dict]:
        """Load AOI geometry and metadata."""
        geojson_path = self.base_dir / "aois" / name / "geometry.geojson"
        if not geojson_path.exists():
            return None
        with open(geojson_path) as f:
            data = json.load(f)
        return {"geometry": data["geometry"], "properties": data.get("properties", {})}

    def list_aois(self) -> List[str]:
        """List all stored AOIs."""
        aois_dir = self.base_dir / "aois"
        return [d.name for d in aois_dir.iterdir() if d.is_dir()] if aois_dir.exists() else []

    def get_imagery_dir(self, aoi_name: str, date: Optional[str] = None, item_id: Optional[str] = None) -> Path:
        """Get imagery directory path."""
        path = self.base_dir / "imagery" / aoi_name
        if date:
            path = path / date
        if item_id:
            path = path / item_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_imagery(self, src_path: Path, aoi_name: str, item_id: str,
                     asset_type: str, metadata: Dict) -> Path:
        """Save downloaded imagery with metadata."""
        acquired = metadata.get("properties", {}).get("acquired", "")
        date = acquired[:10] if acquired else "unknown"

        dest_dir = self.get_imagery_dir(aoi_name, date, item_id)
        dest_file = dest_dir / f"{asset_type}.tif"
        shutil.copy2(src_path, dest_file)

        with open(dest_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return dest_file

    def get_imagery_files(self, aoi_name: str, asset_type: Optional[str] = None) -> List[Path]:
        """Get all imagery files for an AOI."""
        imagery_dir = self.base_dir / "imagery" / aoi_name
        if not imagery_dir.exists():
            return []

        pattern = f"**/{asset_type}.tif" if asset_type else "**/*.tif"
        return sorted(imagery_dir.glob(pattern))

    def get_dir(self, category: str, aoi_name: str) -> Path:
        """Get directory for a category (processed, indices, labels, exports)."""
        path = self.base_dir / category / aoi_name
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# Planet API Client
# =============================================================================

class PlanetClient:
    """Planet Labs API client with retry logic and rate limiting."""

    def __init__(self, config: Optional[PlanetConfig] = None):
        self.config = config or PlanetConfig()
        self.session = requests.Session()
        self.session.auth = (self.config.api_key, "")
        self.session.headers.update({"Content-Type": "application/json"})

    def search(
        self,
        geometry: Dict,
        start_date: str,
        end_date: str,
        item_types: List[str] = None,
        cloud_cover_max: float = 0.1,
        min_gsd: Optional[float] = None,
        max_gsd: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search Planet API for imagery.

        Args:
            geometry: GeoJSON geometry (Polygon or Point with buffer)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            item_types: Planet item types (default: ["PSScene"])
            cloud_cover_max: Maximum cloud cover (0.0-1.0)
            min_gsd: Minimum ground sample distance (meters)
            max_gsd: Maximum ground sample distance (meters)
            limit: Maximum results to return

        Returns:
            List of scene features
        """
        item_types = item_types or ["PSScene"]

        # Build filters
        filters = [
            {"type": "GeometryFilter", "field_name": "geometry", "config": geometry},
            {"type": "DateRangeFilter", "field_name": "acquired", "config": {
                "gte": f"{start_date}T00:00:00.000Z",
                "lte": f"{end_date}T23:59:59.999Z"
            }},
            {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": cloud_cover_max}}
        ]

        if min_gsd is not None:
            filters.append({"type": "RangeFilter", "field_name": "gsd", "config": {"gte": min_gsd}})
        if max_gsd is not None:
            filters.append({"type": "RangeFilter", "field_name": "gsd", "config": {"lte": max_gsd}})

        request_body = {
            "item_types": item_types,
            "filter": {"type": "AndFilter", "config": filters}
        }

        # Execute search with pagination
        all_features = []
        url = f"{self.config.base_url}/quick-search"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(url, json=request_body)
                response.raise_for_status()
                data = response.json()
                all_features.extend(data.get("features", []))
                next_url = data.get("_links", {}).get("_next")

                # Paginate
                while next_url and (limit is None or len(all_features) < limit):
                    response = self.session.get(next_url)
                    response.raise_for_status()
                    data = response.json()
                    all_features.extend(data.get("features", []))
                    next_url = data.get("_links", {}).get("_next")

                break
            except requests.HTTPError as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        if limit:
            all_features = all_features[:limit]

        logger.info(f"Found {len(all_features)} scenes")
        return all_features

    def get_assets(self, item_type: str, item_id: str) -> Dict:
        """Get available assets for an item."""
        url = f"{self.config.base_url}/item-types/{item_type}/items/{item_id}/assets"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    delay = float(e.response.headers.get('Retry-After', self.config.retry_delay * (2 ** attempt)))
                    time.sleep(delay)
                    continue
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                raise
        return {}

    def activate_and_download(self, item_type: str, item_id: str, asset_type: str) -> Optional[str]:
        """Activate asset and get download URL."""
        assets = self.get_assets(item_type, item_id)

        if asset_type not in assets:
            logger.warning(f"Asset {asset_type} not available for {item_id}")
            return None

        asset = assets[asset_type]
        status = asset.get("status")

        if status == "active":
            return asset.get("location")

        # Activate
        activate_url = asset.get("_links", {}).get("activate")
        if not activate_url:
            logger.error(f"No activation URL for {item_id}/{asset_type}")
            return None

        try:
            self.session.post(activate_url).raise_for_status()
        except requests.HTTPError:
            return None

        # Poll for activation
        asset_url = asset.get("_links", {}).get("_self")
        start_time = time.time()

        while time.time() - start_time < self.config.activation_timeout:
            try:
                response = self.session.get(asset_url)
                response.raise_for_status()
                asset_data = response.json()

                if asset_data.get("status") == "active":
                    return asset_data.get("location")
                elif asset_data.get("status") == "failed":
                    logger.error(f"Activation failed for {item_id}/{asset_type}")
                    return None

                time.sleep(5)
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    time.sleep(float(e.response.headers.get('Retry-After', 10)))
                    continue
                raise

        logger.error(f"Activation timeout for {item_id}/{asset_type}")
        return None


# =============================================================================
# Downloader
# =============================================================================

class Downloader:
    """Download Planet imagery with parallel processing."""

    def __init__(self, client: PlanetClient, storage: Storage, config: Optional[PlanetConfig] = None):
        self.client = client
        self.storage = storage
        self.config = config or PlanetConfig()

    def download_scenes(
        self,
        scenes: List[Dict],
        aoi_name: str,
        asset_types: List[str] = None,
        skip_existing: bool = True
    ) -> List[Path]:
        """
        Download multiple scenes.

        Args:
            scenes: List of scene features from search
            aoi_name: AOI identifier
            asset_types: Asset types to download (default: ["ortho_analytic_4b"])
            skip_existing: Skip already downloaded files

        Returns:
            List of downloaded file paths
        """
        asset_types = asset_types or ["ortho_analytic_4b"]
        downloaded = []

        with ThreadPoolExecutor(max_workers=self.config.download_workers) as executor:
            futures = []

            for scene in scenes:
                item_type = scene.get("properties", {}).get("item_type")
                item_id = scene.get("id")

                if not item_type or not item_id:
                    continue

                # Check existing
                if skip_existing:
                    existing = self.storage.get_imagery_files(aoi_name)
                    if any(item_id in str(f) for f in existing):
                        continue

                for asset_type in asset_types:
                    futures.append(executor.submit(
                        self._download_single,
                        item_type, item_id, asset_type, aoi_name, scene
                    ))

            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    try:
                        path = future.result()
                        if path:
                            downloaded.append(path)
                    except Exception as e:
                        logger.error(f"Download failed: {e}")
                    pbar.update(1)

        logger.info(f"Downloaded {len(downloaded)} files for {aoi_name}")
        return downloaded

    def _download_single(
        self,
        item_type: str,
        item_id: str,
        asset_type: str,
        aoi_name: str,
        metadata: Dict
    ) -> Optional[Path]:
        """Download a single asset."""
        download_url = self.client.activate_and_download(item_type, item_id, asset_type)

        if not download_url:
            return None

        # Download to temp file
        temp_dir = Path("/tmp/planet_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"{item_id}_{asset_type}.tif"

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.session.get(download_url, stream=True)
                response.raise_for_status()

                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)

                # Save to storage
                final_path = self.storage.save_imagery(temp_file, aoi_name, item_id, asset_type, metadata)
                temp_file.unlink()
                return final_path

            except requests.HTTPError as e:
                if e.response.status_code == 429 and attempt < self.config.max_retries - 1:
                    delay = float(e.response.headers.get('Retry-After', self.config.retry_delay * (2 ** attempt)))
                    time.sleep(delay)
                    continue
                raise

        return None


# =============================================================================
# High-Level Pipeline Interface
# =============================================================================

class PlanetPipeline:
    """
    High-level interface for Planet imagery workflows.

    Example:
        pipeline = PlanetPipeline(storage_dir="./data")
        pipeline.add_aoi("site1", geometry)
        scenes = pipeline.search("site1", "2024-01-01", "2024-03-31")
        pipeline.download("site1", scenes, limit=10)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_dir: Union[str, Path] = "./planet_data"
    ):
        self.config = PlanetConfig(api_key=api_key)
        self.storage = Storage(storage_dir)
        self.client = PlanetClient(self.config)
        self.downloader = Downloader(self.client, self.storage, self.config)
        self.aois: Dict[str, Dict] = {}

        logger.info(f"Pipeline initialized: {storage_dir}")

    def add_aoi(
        self,
        name: str,
        geometry: Optional[Dict] = None,
        geometry_file: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add an Area of Interest.

        Args:
            name: Unique AOI identifier
            geometry: GeoJSON geometry dict
            geometry_file: Path to GeoJSON file (alternative to geometry)
            metadata: Optional metadata dict
        """
        if geometry_file:
            with open(geometry_file) as f:
                data = json.load(f)
            if data.get("type") == "FeatureCollection":
                geometry = data["features"][0]["geometry"]
            elif data.get("type") == "Feature":
                geometry = data["geometry"]
            else:
                geometry = data

        if not geometry:
            raise ValueError("Must provide geometry or geometry_file")

        self.aois[name] = {"geometry": geometry, "metadata": metadata or {}}
        self.storage.save_aoi(name, geometry, metadata)
        logger.info(f"Added AOI: {name}")

    def add_aois_from_file(self, filepath: str) -> None:
        """Load multiple AOIs from GeoJSON FeatureCollection or JSON dict."""
        with open(filepath) as f:
            data = json.load(f)

        if data.get("type") == "FeatureCollection":
            for feature in data["features"]:
                name = feature.get("properties", {}).get("name")
                if name:
                    self.add_aoi(name, feature["geometry"], metadata=feature.get("properties"))
        else:
            for name, aoi_data in data.items():
                self.add_aoi(name, aoi_data.get("geometry"), metadata=aoi_data.get("metadata"))

    def search(
        self,
        aoi_name: str,
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 0.1,
        max_gsd: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for imagery in an AOI.

        Args:
            aoi_name: AOI identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover_max: Maximum cloud cover (0-1)
            max_gsd: Maximum resolution in meters
            limit: Maximum results

        Returns:
            List of scene features
        """
        if aoi_name not in self.aois:
            raise ValueError(f"Unknown AOI: {aoi_name}")

        return self.client.search(
            geometry=self.aois[aoi_name]["geometry"],
            start_date=start_date,
            end_date=end_date,
            cloud_cover_max=cloud_cover_max,
            max_gsd=max_gsd,
            limit=limit
        )

    def search_all(
        self,
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 0.1,
        max_gsd: Optional[float] = None
    ) -> Dict[str, List[Dict]]:
        """Search all registered AOIs."""
        results = {}
        for aoi_name in self.aois:
            results[aoi_name] = self.search(aoi_name, start_date, end_date, cloud_cover_max, max_gsd)
        return results

    def download(
        self,
        aoi_name: str,
        scenes: List[Dict],
        asset_types: List[str] = None,
        limit: Optional[int] = None
    ) -> List[Path]:
        """
        Download scenes for an AOI.

        Args:
            aoi_name: AOI identifier
            scenes: Scene features from search
            asset_types: Asset types (default: ["ortho_analytic_4b"])
            limit: Maximum scenes to download

        Returns:
            List of downloaded file paths
        """
        if limit:
            scenes = scenes[:limit]
        return self.downloader.download_scenes(scenes, aoi_name, asset_types)

    def get_imagery(self, aoi_name: str, asset_type: Optional[str] = None) -> List[Path]:
        """Get downloaded imagery files for an AOI."""
        return self.storage.get_imagery_files(aoi_name, asset_type)
