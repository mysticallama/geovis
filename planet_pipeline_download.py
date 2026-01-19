"""
Imagery Download Module

Handles downloading Planet imagery with activation, progress tracking, and error recovery.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .query import PlanetAPIClient
from .storage import ImageryStorage

logger = logging.getLogger(__name__)


class ImageryDownloader:
    """Download Planet imagery with activation and progress tracking."""
    
    def __init__(
        self,
        api_key: str,
        storage: ImageryStorage,
        max_workers: int = 4,
        chunk_size: int = 8192
    ):
        """
        Initialize downloader.
        
        Args:
            api_key: Planet API key
            storage: Storage manager instance
            max_workers: Number of parallel downloads
            chunk_size: Download chunk size in bytes
        """
        self.client = PlanetAPIClient(api_key=api_key)
        self.storage = storage
        self.max_workers = max_workers
        self.chunk_size = chunk_size
    
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
            scenes: List of scene features from Planet API
            aoi_name: AOI identifier
            asset_types: Asset types to download
            skip_existing: Skip already downloaded files
        
        Returns:
            List of downloaded file paths
        """
        asset_types = asset_types or ["ortho_analytic_4b"]
        downloaded_files = []
        
        logger.info(f"Starting download of {len(scenes)} scenes for {aoi_name}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for scene in scenes:
                item_type = scene.get("properties", {}).get("item_type")
                item_id = scene.get("id")
                
                if not item_type or not item_id:
                    logger.warning("Scene missing item_type or id, skipping")
                    continue
                
                for asset_type in asset_types:
                    future = executor.submit(
                        self._download_single_asset,
                        item_type=item_type,
                        item_id=item_id,
                        asset_type=asset_type,
                        aoi_name=aoi_name,
                        scene_metadata=scene,
                        skip_existing=skip_existing
                    )
                    futures.append(future)
            
            # Process completed downloads with progress bar
            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    try:
                        filepath = future.result()
                        if filepath:
                            downloaded_files.append(filepath)
                    except Exception as e:
                        logger.error(f"Download failed: {e}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"Downloaded {len(downloaded_files)} files for {aoi_name}")
        return downloaded_files
    
    def _download_single_asset(
        self,
        item_type: str,
        item_id: str,
        asset_type: str,
        aoi_name: str,
        scene_metadata: Dict,
        skip_existing: bool = True
    ) -> Optional[Path]:
        """
        Download a single asset.
        
        Args:
            item_type: Planet item type
            item_id: Item ID
            asset_type: Asset type to download
            aoi_name: AOI identifier
            scene_metadata: Scene metadata
            skip_existing: Skip if already exists
        
        Returns:
            Path to downloaded file or None
        """
        # Check if already downloaded
        existing_files = self.storage.get_imagery_files(aoi_name, asset_type=asset_type)
        if skip_existing and any(item_id in str(f) for f in existing_files):
            logger.debug(f"Skipping existing file: {item_id}/{asset_type}")
            return None
        
        try:
            # Get asset info
            assets = self.client.get_item_assets(item_type, item_id)
            
            if asset_type not in assets:
                logger.warning(f"Asset type {asset_type} not available for {item_id}")
                return None
            
            asset = assets[asset_type]
            asset_url = asset.get("_links", {}).get("_self")
            
            if not asset_url:
                logger.error(f"No asset URL for {item_id}/{asset_type}")
                return None
            
            # Activate asset if needed
            status = asset.get("status")
            if status != "active":
                logger.debug(f"Activating asset: {item_id}/{asset_type}")
                activate_url = asset.get("_links", {}).get("activate")
                if activate_url:
                    self.client.activate_asset(activate_url)
                
                # Wait for activation
                download_url = self.client.get_download_url(asset_url, timeout=300)
            else:
                download_url = asset.get("location")
            
            if not download_url:
                logger.error(f"Could not get download URL for {item_id}/{asset_type}")
                return None
            
            # Download file
            temp_file = self._download_file(download_url, item_id, asset_type)
            
            # Save to storage
            final_path = self.storage.save_imagery(
                filepath=temp_file,
                aoi_name=aoi_name,
                item_id=item_id,
                asset_type=asset_type,
                metadata=scene_metadata
            )
            
            # Clean up temp file
            temp_file.unlink()
            
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to download {item_id}/{asset_type}: {e}")
            return None
    
    def _download_file(
        self,
        url: str,
        item_id: str,
        asset_type: str
    ) -> Path:
        """
        Download file from URL.
        
        Args:
            url: Download URL
            item_id: Item ID
            asset_type: Asset type
        
        Returns:
            Path to temporary downloaded file
        """
        # Create temp directory
        temp_dir = Path("/tmp/planet_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / f"{item_id}_{asset_type}.tif"
        
        # Download with progress
        response = self.client.session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_file, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
        
        return temp_file
    
    def download_by_ids(
        self,
        item_ids: List[str],
        item_type: str,
        aoi_name: str,
        asset_types: List[str] = None
    ) -> List[Path]:
        """
        Download specific items by ID.
        
        Args:
            item_ids: List of Planet item IDs
            item_type: Planet item type
            aoi_name: AOI identifier
            asset_types: Asset types to download
        
        Returns:
            List of downloaded file paths
        """
        asset_types = asset_types or ["ortho_analytic_4b"]
        downloaded_files = []
        
        for item_id in item_ids:
            for asset_type in asset_types:
                filepath = self._download_single_asset(
                    item_type=item_type,
                    item_id=item_id,
                    asset_type=asset_type,
                    aoi_name=aoi_name,
                    scene_metadata={"id": item_id, "properties": {"item_type": item_type}},
                    skip_existing=False
                )
                
                if filepath:
                    downloaded_files.append(filepath)
        
        return downloaded_files
    
    def estimate_download_size(self, scenes: List[Dict], asset_types: List[str]) -> int:
        """
        Estimate total download size.
        
        Args:
            scenes: List of scenes
            asset_types: Asset types
        
        Returns:
            Estimated size in bytes
        """
        # Rough estimates per asset type (in MB)
        size_estimates = {
            "ortho_analytic_4b": 100,
            "ortho_analytic_8b": 200,
            "ortho_visual": 75,
            "ortho_analytic_sr": 100,
            "basic_analytic": 50,
            "basic_analytic_dn": 50
        }
        
        total_mb = 0
        for scene in scenes:
            for asset_type in asset_types:
                total_mb += size_estimates.get(asset_type, 100)
        
        return total_mb * 1024 * 1024  # Convert to bytes
