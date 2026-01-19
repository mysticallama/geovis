"""
Imagery Storage Module

Manages organized storage of Planet imagery, metadata, and derived products.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ImageryStorage:
    """
    Manages structured storage for Planet imagery and derived products.
    
    Directory structure:
        base_dir/
            aois/
                {aoi_name}/
                    metadata.json
                    geometry.geojson
            imagery/
                {aoi_name}/
                    {date}/
                        {item_id}/
                            {asset_type}.tif
                            metadata.json
            processed/
                {aoi_name}/
                    {processing_type}/
                        {item_id}.tif
            indices/
                {aoi_name}/
                    {index_name}/
                        {item_id}.tif
            ml_datasets/
                {dataset_name}/
                    train/
                    val/
                    test/
            cache/
                search_results/
                api_responses/
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize storage manager.
        
        Args:
            base_dir: Root directory for all storage
        """
        self.base_dir = Path(base_dir)
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create directory structure."""
        dirs = [
            "aois",
            "imagery",
            "processed",
            "indices",
            "ml_datasets",
            "cache/search_results",
            "cache/api_responses"
        ]
        
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
    
    def save_aoi(self, name: str, aoi_data: Dict) -> None:
        """
        Save AOI metadata and geometry.
        
        Args:
            name: AOI identifier
            aoi_data: AOI data including geometry and metadata
        """
        aoi_dir = self.base_dir / "aois" / name
        aoi_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(aoi_dir / "metadata.json", 'w') as f:
            json.dump({
                "name": name,
                "metadata": aoi_data.get("metadata", {}),
                "added_at": aoi_data.get("added_at")
            }, f, indent=2)
        
        # Save geometry as GeoJSON
        with open(aoi_dir / "geometry.geojson", 'w') as f:
            json.dump({
                "type": "Feature",
                "properties": {"name": name},
                "geometry": aoi_data["geometry"]
            }, f, indent=2)
    
    def get_aoi(self, name: str) -> Optional[Dict]:
        """
        Load AOI data.
        
        Args:
            name: AOI identifier
        
        Returns:
            AOI data or None if not found
        """
        aoi_dir = self.base_dir / "aois" / name
        if not aoi_dir.exists():
            return None
        
        with open(aoi_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        with open(aoi_dir / "geometry.geojson", 'r') as f:
            geojson = json.load(f)
        
        return {
            "geometry": geojson["geometry"],
            "metadata": metadata.get("metadata", {}),
            "added_at": metadata.get("added_at")
        }
    
    def list_aois(self) -> List[str]:
        """
        List all stored AOIs.
        
        Returns:
            List of AOI names
        """
        aois_dir = self.base_dir / "aois"
        if not aois_dir.exists():
            return []
        
        return [d.name for d in aois_dir.iterdir() if d.is_dir()]
    
    def save_search_results(self, results: Dict[str, List[Dict]], params: Dict) -> None:
        """
        Save search results with parameters.
        
        Args:
            results: Search results by AOI
            params: Search parameters
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"search_{timestamp}.json"
        filepath = self.base_dir / "cache" / "search_results" / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "parameters": params,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Search results saved to {filepath}")
    
    def get_imagery_dir(self, aoi_name: str, date: Optional[str] = None) -> Path:
        """
        Get imagery directory for AOI and optional date.
        
        Args:
            aoi_name: AOI identifier
            date: Optional date (YYYY-MM-DD)
        
        Returns:
            Path to imagery directory
        """
        if date:
            path = self.base_dir / "imagery" / aoi_name / date
        else:
            path = self.base_dir / "imagery" / aoi_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_imagery(
        self,
        filepath: Path,
        aoi_name: str,
        item_id: str,
        asset_type: str,
        metadata: Dict
    ) -> Path:
        """
        Save downloaded imagery with metadata.
        
        Args:
            filepath: Source file path
            aoi_name: AOI identifier
            item_id: Planet item ID
            asset_type: Asset type
            metadata: Scene metadata
        
        Returns:
            Destination path
        """
        # Extract date from metadata
        acquired = metadata.get("properties", {}).get("acquired", "")
        date = acquired[:10] if acquired else "unknown"
        
        # Create destination directory
        dest_dir = self.get_imagery_dir(aoi_name, date) / item_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy imagery file
        dest_file = dest_dir / f"{asset_type}.tif"
        shutil.copy2(filepath, dest_file)
        
        # Save metadata
        metadata_file = dest_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved imagery to {dest_file}")
        return dest_file
    
    def get_imagery_files(
        self,
        aoi_name: str,
        asset_type: Optional[str] = None
    ) -> List[Path]:
        """
        Get all imagery files for an AOI.
        
        Args:
            aoi_name: AOI identifier
            asset_type: Optional filter by asset type
        
        Returns:
            List of imagery file paths
        """
        imagery_dir = self.base_dir / "imagery" / aoi_name
        if not imagery_dir.exists():
            return []
        
        files = []
        pattern = f"*/{asset_type}.tif" if asset_type else "*/*.tif"
        
        for date_dir in imagery_dir.iterdir():
            if date_dir.is_dir():
                files.extend(date_dir.glob(pattern))
        
        return sorted(files)
    
    def get_processed_dir(self, aoi_name: str) -> Path:
        """Get processed imagery directory."""
        path = self.base_dir / "processed" / aoi_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_indices_dir(self, aoi_name: str) -> Path:
        """Get spectral indices directory."""
        path = self.base_dir / "indices" / aoi_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_ml_dataset_dir(self, dataset_name: str) -> Path:
        """Get ML dataset directory."""
        path = self.base_dir / "ml_datasets" / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        
        # Create train/val/test splits
        for split in ["train", "val", "test"]:
            (path / split).mkdir(exist_ok=True)
        
        return path
    
    def get_storage_usage(self) -> Dict[str, int]:
        """
        Calculate storage usage by category.
        
        Returns:
            Dict of storage usage in bytes
        """
        def get_dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        return {
            "imagery": get_dir_size(self.base_dir / "imagery"),
            "processed": get_dir_size(self.base_dir / "processed"),
            "indices": get_dir_size(self.base_dir / "indices"),
            "ml_datasets": get_dir_size(self.base_dir / "ml_datasets"),
            "cache": get_dir_size(self.base_dir / "cache"),
            "total": get_dir_size(self.base_dir)
        }
    
    def get_structure(self) -> Dict:
        """
        Get storage structure summary.
        
        Returns:
            Dict describing storage structure
        """
        return {
            "aois": self.list_aois(),
            "num_aois": len(self.list_aois()),
            "storage_usage_mb": {
                k: v / (1024 * 1024) for k, v in self.get_storage_usage().items()
            }
        }
    
    def cleanup_cache(self, older_than_days: int = 30) -> None:
        """
        Remove old cache files.
        
        Args:
            older_than_days: Remove files older than this many days
        """
        import time
        
        cache_dir = self.base_dir / "cache"
        cutoff_time = time.time() - (older_than_days * 86400)
        
        for cache_file in cache_dir.rglob('*'):
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                logger.info(f"Removed cached file: {cache_file}")
