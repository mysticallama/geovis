"""
Planet API Query Module

Enhanced querying capabilities with multi-AOI support, caching, and error recovery.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from pathlib import Path


class PlanetAPIClient:
    """Enhanced Planet API client with caching and retry logic."""
    
    BASE_URL = "https://api.planet.com/data/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize Planet API client.
        
        Args:
            api_key: Planet API key
            cache_dir: Directory for caching responses
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.environ.get("PL_API_KEY")
        if not self.api_key:
            raise ValueError("Planet API key required")
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.session = requests.Session()
        self.session.auth = (self.api_key, "")
        self.session.headers.update({"Content-Type": "application/json"})
    
    def build_search_request(
        self,
        geometry: Dict,
        start_date: str,
        end_date: str,
        item_types: List[str],
        cloud_cover_max: float = 0.1,
        min_gsd: Optional[float] = None,
        max_gsd: Optional[float] = None,
        quality_category: Optional[str] = None
    ) -> Dict:
        """
        Build Planet API search request with filters.
        
        Args:
            geometry: GeoJSON geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            item_types: List of Planet item types
            cloud_cover_max: Maximum cloud cover (0.0-1.0)
            min_gsd: Minimum GSD in meters
            max_gsd: Maximum GSD in meters
            quality_category: Quality filter ("standard", "test", etc.)
        
        Returns:
            Search request payload
        """
        filters = [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": geometry
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": f"{start_date}T00:00:00.000Z",
                    "lte": f"{end_date}T23:59:59.999Z"
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": cloud_cover_max}
            }
        ]
        
        if min_gsd is not None:
            filters.append({
                "type": "RangeFilter",
                "field_name": "gsd",
                "config": {"gte": min_gsd}
            })
        
        if max_gsd is not None:
            filters.append({
                "type": "RangeFilter",
                "field_name": "gsd",
                "config": {"lte": max_gsd}
            })
        
        if quality_category:
            filters.append({
                "type": "StringInFilter",
                "field_name": "quality_category",
                "config": [quality_category]
            })
        
        return {
            "item_types": item_types,
            "filter": {
                "type": "AndFilter",
                "config": filters
            }
        }
    
    def quick_search(
        self,
        search_request: Dict,
        limit: int = 250
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Execute quick search with retry logic.
        
        Args:
            search_request: Search request payload
            limit: Max results per page
        
        Returns:
            Tuple of (features, next_page_url)
        """
        url = f"{self.BASE_URL}/quick-search"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=search_request)
                response.raise_for_status()
                
                data = response.json()
                return data.get("features", []), data.get("_links", {}).get("_next")
                
            except requests.HTTPError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
        
        return [], None
    
    def get_all_pages(
        self,
        search_request: Dict,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """
        Get all pages of search results.
        
        Args:
            search_request: Search request payload
            max_results: Maximum total results to retrieve
        
        Returns:
            List of all features
        """
        all_features = []
        features, next_url = self.quick_search(search_request)
        all_features.extend(features)
        
        while next_url and (max_results is None or len(all_features) < max_results):
            response = self.session.get(next_url)
            response.raise_for_status()
            data = response.json()
            
            features = data.get("features", [])
            all_features.extend(features)
            next_url = data.get("_links", {}).get("_next")
            
            if max_results and len(all_features) >= max_results:
                all_features = all_features[:max_results]
                break
        
        return all_features
    
    def get_item_assets(self, item_type: str, item_id: str) -> Dict:
        """
        Get available assets for an item.

        Args:
            item_type: Planet item type
            item_id: Item ID

        Returns:
            Dict of available assets
        """
        url = f"{self.BASE_URL}/item-types/{item_type}/items/{item_id}/assets"

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                if e.response.status_code == 429 and attempt < self.max_retries - 1:
                    # Handle rate limiting with exponential backoff
                    retry_after = e.response.headers.get('Retry-After')
                    delay = float(retry_after) if retry_after else self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise

        return {}
    
    def activate_asset(self, asset_url: str) -> bool:
        """
        Activate an asset for download with retry logic.

        Args:
            asset_url: Asset activation URL (the activate endpoint)

        Returns:
            True if activation request successful
        """
        for attempt in range(self.max_retries):
            try:
                # IMPORTANT: Must use POST to activate, not GET
                response = self.session.post(asset_url)
                response.raise_for_status()
                return True
            except requests.HTTPError as e:
                if e.response.status_code == 429 and attempt < self.max_retries - 1:
                    retry_after = e.response.headers.get('Retry-After')
                    delay = float(retry_after) if retry_after else self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise

        return False
    
    def get_download_url(self, asset_url: str, timeout: int = 300) -> Optional[str]:
        """
        Get download URL for an activated asset with rate limit handling.

        Args:
            asset_url: Asset URL (the _self link to check status)
            timeout: Maximum wait time in seconds

        Returns:
            Download URL or None if timeout
        """
        import logging
        logger = logging.getLogger(__name__)

        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(asset_url)
                response.raise_for_status()
                asset_data = response.json()

                status = asset_data.get("status")

                # Log status changes
                if status != last_status:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"Asset status: {status} (after {elapsed}s)")
                    last_status = status

                if status == "active":
                    return asset_data.get("location")
                elif status == "failed":
                    raise RuntimeError(f"Asset activation failed: {asset_data.get('error', 'Unknown error')}")
                elif status == "inactive":
                    # Asset needs activation but wasn't activated properly
                    raise RuntimeError("Asset is inactive - activation may have failed")

                # Still activating, wait before checking again
                time.sleep(5)

            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get('Retry-After')
                    delay = float(retry_after) if retry_after else 10
                    logger.warning(f"Rate limited while checking activation status, waiting {delay}s")
                    time.sleep(delay)
                    continue
                elif e.response.status_code == 403:
                    # Permission denied - might not have access to this asset
                    raise RuntimeError(f"Access denied to asset - may require subscription or permissions")
                raise

        # Timeout reached
        logger.error(f"Asset activation timed out after {timeout}s. Last status: {last_status}")
        return None
