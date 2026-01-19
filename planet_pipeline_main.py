"""
Planet Labs Imagery Processing Pipeline

A modular, production-ready pipeline for querying, downloading, storing, and processing
Planet Labs satellite imagery for machine learning workflows.

Usage:
    from planet_pipeline import PlanetPipeline
    
    pipeline = PlanetPipeline(api_key="your_key", storage_dir="./data")
    pipeline.add_aoi("san_francisco", geometry_file="sf_aoi.geojson")
    pipeline.query_all_aois(start_date="2024-01-01", end_date="2024-01-31")
    pipeline.download_imagery(max_cloud_cover=0.1)
    pipeline.calculate_indices(indices=["ndvi", "ndwi", "evi"])
    pipeline.prepare_for_ml(model_type="pytorch")
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

from .query import PlanetAPIClient
from .download import ImageryDownloader
from .storage import ImageryStorage
from .preprocessing import ImagePreprocessor
from .indices import SpectralIndices
from .ml_prep import MLDataPrep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlanetPipeline:
    """
    Main pipeline orchestrator for Planet Labs imagery processing.
    
    This class provides a high-level interface for managing the entire workflow
    from querying to ML-ready data preparation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_dir: Union[str, Path] = "./planet_data",
        cache_enabled: bool = True
    ):
        """
        Initialize the Planet Labs processing pipeline.
        
        Args:
            api_key: Planet API key (reads from PL_API_KEY if not provided)
            storage_dir: Root directory for storing all pipeline data
            cache_enabled: Whether to cache API responses and intermediate results
        """
        self.api_key = api_key or os.environ.get("PL_API_KEY")
        if not self.api_key:
            raise ValueError("Planet API key required. Set PL_API_KEY or pass api_key parameter.")
        
        self.storage_dir = Path(storage_dir)
        self.cache_enabled = cache_enabled
        
        # Initialize components
        self.client = PlanetAPIClient(api_key=self.api_key)
        self.storage = ImageryStorage(base_dir=self.storage_dir)
        self.downloader = ImageryDownloader(
            api_key=self.api_key,
            storage=self.storage
        )
        self.preprocessor = ImagePreprocessor()
        self.indices = SpectralIndices()
        self.ml_prep = MLDataPrep(storage_dir=self.storage_dir)
        
        # Track AOIs and search results
        self.aois: Dict[str, Dict] = {}
        self.search_results: Dict[str, List[Dict]] = {}
        
        logger.info(f"Pipeline initialized with storage at: {self.storage_dir}")
    
    def add_aoi(
        self,
        name: str,
        geometry: Optional[Dict] = None,
        geometry_file: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add an Area of Interest (AOI) to the pipeline.
        
        Args:
            name: Unique identifier for this AOI
            geometry: GeoJSON geometry dict
            geometry_file: Path to GeoJSON file
            metadata: Optional metadata to associate with this AOI
        
        Raises:
            ValueError: If neither geometry nor geometry_file provided
        """
        if name in self.aois:
            logger.warning(f"AOI '{name}' already exists. Overwriting.")
        
        # Load geometry
        if geometry_file:
            with open(geometry_file, 'r') as f:
                data = json.load(f)
                if data.get("type") == "FeatureCollection":
                    geometry = data["features"][0]["geometry"]
                elif data.get("type") == "Feature":
                    geometry = data["geometry"]
                else:
                    geometry = data
        elif geometry is None:
            raise ValueError("Must provide either geometry or geometry_file")
        
        self.aois[name] = {
            "geometry": geometry,
            "metadata": metadata or {},
            "added_at": datetime.utcnow().isoformat()
        }
        
        # Save AOI to storage
        self.storage.save_aoi(name, self.aois[name])
        logger.info(f"Added AOI: {name}")
    
    def add_aois_from_file(self, filepath: str) -> None:
        """
        Load multiple AOIs from a JSON or GeoJSON file.
        
        Expected format:
        {
            "aoi_name_1": {"geometry": {...}, "metadata": {...}},
            "aoi_name_2": {"geometry": {...}, "metadata": {...}}
        }
        
        Or a FeatureCollection where properties.name is used as AOI name.
        
        Args:
            filepath: Path to JSON/GeoJSON file containing multiple AOIs
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data.get("type") == "FeatureCollection":
            # Handle GeoJSON FeatureCollection
            for feature in data["features"]:
                name = feature.get("properties", {}).get("name")
                if not name:
                    logger.warning("Feature without 'name' property skipped")
                    continue
                
                self.add_aoi(
                    name=name,
                    geometry=feature["geometry"],
                    metadata=feature.get("properties", {})
                )
        else:
            # Handle custom JSON format
            for name, aoi_data in data.items():
                self.add_aoi(
                    name=name,
                    geometry=aoi_data.get("geometry"),
                    metadata=aoi_data.get("metadata", {})
                )
        
        logger.info(f"Loaded {len(self.aois)} AOIs from {filepath}")
    
    def query_all_aois(
        self,
        start_date: str,
        end_date: str,
        item_types: List[str] = None,
        cloud_cover_max: float = 0.1,
        min_gsd: Optional[float] = None,
        max_gsd: Optional[float] = None
    ) -> Dict[str, List[Dict]]:
        """
        Query Planet API for all registered AOIs.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            item_types: List of Planet item types (default: ["PSScene"])
            cloud_cover_max: Maximum cloud cover (0.0-1.0)
            min_gsd: Minimum resolution in meters
            max_gsd: Maximum resolution in meters
        
        Returns:
            Dict mapping AOI names to lists of search results
        """
        if not self.aois:
            raise ValueError("No AOIs registered. Add AOIs first using add_aoi()")
        
        item_types = item_types or ["PSScene"]
        results = {}
        
        logger.info(f"Querying {len(self.aois)} AOIs from {start_date} to {end_date}")
        
        for aoi_name, aoi_data in self.aois.items():
            logger.info(f"Querying AOI: {aoi_name}")
            
            search_request = self.client.build_search_request(
                geometry=aoi_data["geometry"],
                start_date=start_date,
                end_date=end_date,
                item_types=item_types,
                cloud_cover_max=cloud_cover_max,
                min_gsd=min_gsd,
                max_gsd=max_gsd
            )
            
            features, _ = self.client.quick_search(search_request)
            results[aoi_name] = features
            
            logger.info(f"Found {len(features)} scenes for {aoi_name}")
        
        self.search_results = results
        
        # Save search results
        self.storage.save_search_results(results, {
            "start_date": start_date,
            "end_date": end_date,
            "item_types": item_types,
            "cloud_cover_max": cloud_cover_max,
            "min_gsd": min_gsd,
            "max_gsd": max_gsd
        })
        
        return results
    
    def download_imagery(
        self,
        asset_types: List[str] = None,
        aoi_filter: Optional[List[str]] = None,
        max_cloud_cover: Optional[float] = None,
        limit_per_aoi: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Download imagery for queried scenes.
        
        Args:
            asset_types: Asset types to download (default: ["ortho_analytic_4b"])
            aoi_filter: Only download for specific AOIs (default: all)
            max_cloud_cover: Further filter by cloud cover
            limit_per_aoi: Maximum scenes to download per AOI
        
        Returns:
            Dict mapping AOI names to lists of downloaded file paths
        """
        if not self.search_results:
            raise ValueError("No search results available. Run query_all_aois() first.")
        
        asset_types = asset_types or ["ortho_analytic_4b"]
        aoi_filter = aoi_filter or list(self.search_results.keys())
        
        downloaded_files = {}
        
        for aoi_name in aoi_filter:
            if aoi_name not in self.search_results:
                logger.warning(f"AOI '{aoi_name}' not in search results. Skipping.")
                continue
            
            scenes = self.search_results[aoi_name]
            
            # Apply additional filtering
            if max_cloud_cover is not None:
                scenes = [
                    s for s in scenes
                    if s.get("properties", {}).get("cloud_cover", 1.0) <= max_cloud_cover
                ]
            
            # Apply limit
            if limit_per_aoi:
                scenes = scenes[:limit_per_aoi]
            
            logger.info(f"Downloading {len(scenes)} scenes for {aoi_name}")
            
            files = self.downloader.download_scenes(
                scenes=scenes,
                aoi_name=aoi_name,
                asset_types=asset_types
            )
            
            downloaded_files[aoi_name] = files
        
        return downloaded_files
    
    def calculate_indices(
        self,
        indices: List[str] = None,
        aoi_filter: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Calculate spectral indices for downloaded imagery.
        
        Args:
            indices: List of indices to calculate (default: ["ndvi", "ndwi", "evi"])
                Options: ndvi, ndwi, evi, savi, msavi, ndbi, bai, nbr, gndvi, etc.
            aoi_filter: Only process specific AOIs (default: all)
        
        Returns:
            Dict mapping AOI names to dicts of index names to output file paths
        """
        indices = indices or ["ndvi", "ndwi", "evi"]
        aoi_filter = aoi_filter or self.storage.list_aois()
        
        results = {}
        
        for aoi_name in aoi_filter:
            logger.info(f"Calculating indices for {aoi_name}")
            
            imagery_files = self.storage.get_imagery_files(aoi_name)
            aoi_results = {}
            
            for idx_name in indices:
                idx_files = []
                
                for img_file in imagery_files:
                    try:
                        output_path = self.indices.calculate(
                            imagery_path=img_file,
                            index_name=idx_name,
                            output_dir=self.storage.get_indices_dir(aoi_name)
                        )
                        idx_files.append(output_path)
                    except Exception as e:
                        logger.error(f"Failed to calculate {idx_name} for {img_file}: {e}")
                
                aoi_results[idx_name] = idx_files
                logger.info(f"Calculated {idx_name} for {len(idx_files)} images")
            
            results[aoi_name] = aoi_results
        
        return results
    
    def prepare_for_ml(
        self,
        model_type: str = "pytorch",
        output_format: str = "chips",
        chip_size: int = 256,
        overlap: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        normalize: bool = True,
        augment: bool = False,
        label_file: Optional[str] = None
    ) -> str:
        """
        Prepare imagery for machine learning model training.
        
        Args:
            model_type: Target ML framework ("pytorch", "tensorflow", "sklearn")
            output_format: "chips" (tiled), "patches" (random samples), or "full" (entire images)
            chip_size: Size of image chips/patches in pixels
            overlap: Overlap between chips in pixels
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            normalize: Apply normalization to imagery
            augment: Apply data augmentation
            label_file: Path to labels file (CSV or GeoJSON)
        
        Returns:
            Path to prepared dataset directory
        """
        logger.info(f"Preparing ML dataset for {model_type}")
        
        dataset_path = self.ml_prep.prepare_dataset(
            aois=list(self.aois.keys()),
            storage=self.storage,
            model_type=model_type,
            output_format=output_format,
            chip_size=chip_size,
            overlap=overlap,
            splits=(train_split, val_split, test_split),
            normalize=normalize,
            augment=augment,
            label_file=label_file
        )
        
        logger.info(f"ML dataset prepared at: {dataset_path}")
        return dataset_path
    
    def preprocess_imagery(
        self,
        operations: List[str] = None,
        aoi_filter: Optional[List[str]] = None
    ) -> None:
        """
        Apply preprocessing operations to imagery.
        
        Args:
            operations: List of preprocessing operations
                Options: "atmospheric_correction", "cloud_mask", "pansharpening",
                        "radiometric_calibration", "orthorectification"
            aoi_filter: Only process specific AOIs (default: all)
        """
        operations = operations or ["cloud_mask"]
        aoi_filter = aoi_filter or self.storage.list_aois()
        
        for aoi_name in aoi_filter:
            logger.info(f"Preprocessing imagery for {aoi_name}")
            imagery_files = self.storage.get_imagery_files(aoi_name)
            
            for img_file in imagery_files:
                self.preprocessor.process(
                    input_path=img_file,
                    operations=operations,
                    output_dir=self.storage.get_processed_dir(aoi_name)
                )
    
    def export_metadata(self, output_file: str) -> None:
        """
        Export complete pipeline metadata to JSON.
        
        Args:
            output_file: Path to output JSON file
        """
        metadata = {
            "aois": self.aois,
            "search_results_summary": {
                aoi: len(results)
                for aoi, results in self.search_results.items()
            },
            "storage_structure": self.storage.get_structure(),
            "pipeline_config": {
                "storage_dir": str(self.storage_dir),
                "cache_enabled": self.cache_enabled
            },
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata exported to: {output_file}")
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the pipeline state.
        
        Returns:
            Dict containing pipeline statistics and status
        """
        return {
            "num_aois": len(self.aois),
            "aoi_names": list(self.aois.keys()),
            "search_results": {
                aoi: len(results)
                for aoi, results in self.search_results.items()
            },
            "storage_usage": self.storage.get_storage_usage(),
            "available_indices": self.indices.list_available_indices()
        }


def main():
    """Example usage of the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Planet Labs Imagery Processing Pipeline")
    parser.add_argument("--storage-dir", default="./planet_data", help="Storage directory")
    parser.add_argument("--aoi-file", help="JSON/GeoJSON file with AOIs")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--download", action="store_true", help="Download imagery")
    parser.add_argument("--indices", nargs="+", help="Calculate spectral indices")
    parser.add_argument("--ml-prep", choices=["pytorch", "tensorflow", "sklearn"],
                       help="Prepare data for ML framework")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PlanetPipeline(storage_dir=args.storage_dir)
    
    # Load AOIs
    if args.aoi_file:
        pipeline.add_aois_from_file(args.aoi_file)
    else:
        logger.error("No AOI file provided. Use --aoi-file")
        return
    
    # Query imagery
    pipeline.query_all_aois(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Download if requested
    if args.download:
        pipeline.download_imagery()
    
    # Calculate indices if requested
    if args.indices:
        pipeline.calculate_indices(indices=args.indices)
    
    # Prepare for ML if requested
    if args.ml_prep:
        pipeline.prepare_for_ml(model_type=args.ml_prep)
    
    # Print summary
    summary = pipeline.get_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
