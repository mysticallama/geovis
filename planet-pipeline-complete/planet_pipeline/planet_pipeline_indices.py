"""
Spectral Indices Module

Calculate common spectral indices for satellite imagery analysis.
Supports Planet Labs 4-band and 8-band imagery.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SpectralIndices:
    """Calculate spectral indices from multispectral imagery."""
    
    # Band indices for Planet PSScene 4-band (Blue, Green, Red, NIR)
    PLANET_4B_BANDS = {
        "blue": 0,
        "green": 1,
        "red": 2,
        "nir": 3
    }
    
    # Band indices for Planet PSScene 8-band
    PLANET_8B_BANDS = {
        "coastal_blue": 0,
        "blue": 1,
        "green_i": 2,
        "green": 3,
        "yellow": 4,
        "red": 5,
        "red_edge": 6,
        "nir": 7
    }
    
    def __init__(self):
        """Initialize spectral indices calculator."""
        self.indices_registry = self._build_indices_registry()
    
    def _build_indices_registry(self) -> Dict[str, Callable]:
        """
        Build registry of available indices.
        
        Returns:
            Dict mapping index names to calculation functions
        """
        return {
            "ndvi": self._calculate_ndvi,
            "ndwi": self._calculate_ndwi,
            "evi": self._calculate_evi,
            "savi": self._calculate_savi,
            "msavi": self._calculate_msavi,
            "ndbi": self._calculate_ndbi,
            "bai": self._calculate_bai,
            "nbr": self._calculate_nbr,
            "gndvi": self._calculate_gndvi,
            "ndmi": self._calculate_ndmi,
            "arvi": self._calculate_arvi,
            "gci": self._calculate_gci,
            "sipi": self._calculate_sipi,
            "vari": self._calculate_vari
        }
    
    def list_available_indices(self) -> List[str]:
        """
        Get list of available indices.
        
        Returns:
            List of index names
        """
        return list(self.indices_registry.keys())
    
    def calculate(
        self,
        imagery_path: Path,
        index_name: str,
        output_dir: Optional[Path] = None,
        nodata_value: float = -9999
    ) -> Path:
        """
        Calculate spectral index.
        
        Args:
            imagery_path: Path to input imagery (GeoTIFF)
            index_name: Name of index to calculate
            output_dir: Output directory (default: same as input)
            nodata_value: NoData value for output
        
        Returns:
            Path to output file
        
        Raises:
            ValueError: If index name not recognized
        """
        if index_name not in self.indices_registry:
            raise ValueError(
                f"Unknown index: {index_name}. "
                f"Available: {', '.join(self.list_available_indices())}"
            )
        
        logger.info(f"Calculating {index_name} for {imagery_path.name}")
        
        # Read imagery
        with rasterio.open(imagery_path) as src:
            bands = src.read()
            profile = src.profile.copy()
            num_bands = src.count
        
        # Select band mapping based on number of bands
        if num_bands == 4:
            band_map = self.PLANET_4B_BANDS
        elif num_bands == 8:
            band_map = self.PLANET_8B_BANDS
        else:
            logger.warning(f"Unexpected band count: {num_bands}. Assuming 4-band.")
            band_map = self.PLANET_4B_BANDS
        
        # Calculate index
        calc_func = self.indices_registry[index_name]
        index_array = calc_func(bands, band_map)
        
        # Handle invalid values
        index_array[~np.isfinite(index_array)] = nodata_value
        
        # Prepare output
        if output_dir is None:
            output_dir = imagery_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{imagery_path.stem}_{index_name}.tif"
        output_path = output_dir / output_filename
        
        # Update profile for single-band output
        profile.update({
            "count": 1,
            "dtype": rasterio.float32,
            "nodata": nodata_value
        })
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(index_array.astype(rasterio.float32), 1)
        
        logger.info(f"Saved {index_name} to {output_path}")
        return output_path
    
    def calculate_multiple(
        self,
        imagery_path: Path,
        index_names: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Calculate multiple indices for single image.
        
        Args:
            imagery_path: Path to input imagery
            index_names: List of index names
            output_dir: Output directory
        
        Returns:
            Dict mapping index names to output paths
        """
        results = {}
        
        for index_name in index_names:
            try:
                output_path = self.calculate(
                    imagery_path=imagery_path,
                    index_name=index_name,
                    output_dir=output_dir
                )
                results[index_name] = output_path
            except Exception as e:
                logger.error(f"Failed to calculate {index_name}: {e}")
        
        return results
    
    # Index calculation methods
    
    def _calculate_ndvi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Normalized Difference Vegetation Index.
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1 (vegetation typically 0.2 to 0.8)
        """
        nir = bands[band_map["nir"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        
        return (nir - red) / (nir + red + 1e-10)
    
    def _calculate_ndwi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Normalized Difference Water Index.
        NDWI = (Green - NIR) / (Green + NIR)
        Range: -1 to 1 (water typically > 0.3)
        """
        green = bands[band_map["green"]].astype(float)
        nir = bands[band_map["nir"]].astype(float)
        
        return (green - nir) / (green + nir + 1e-10)
    
    def _calculate_evi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Enhanced Vegetation Index.
        EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        Range: -1 to 1
        """
        nir = bands[band_map["nir"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        blue = bands[band_map["blue"]].astype(float)
        
        return 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-10))
    
    def _calculate_savi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Soil-Adjusted Vegetation Index.
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        L = 0.5 (default soil brightness correction factor)
        """
        nir = bands[band_map["nir"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        L = 0.5
        
        return ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)
    
    def _calculate_msavi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Modified Soil-Adjusted Vegetation Index.
        MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
        """
        nir = bands[band_map["nir"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        
        return (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red))) / 2
    
    def _calculate_ndbi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Normalized Difference Built-up Index.
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        Note: Approximated using Red band as SWIR substitute for 4-band imagery
        """
        # For 4-band, approximate with Red
        # For proper NDBI, SWIR band is needed
        swir_proxy = bands[band_map["red"]].astype(float)
        nir = bands[band_map["nir"]].astype(float)
        
        return (swir_proxy - nir) / (swir_proxy + nir + 1e-10)
    
    def _calculate_bai(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Burned Area Index.
        BAI = 1 / ((0.1 - Red)^2 + (0.06 - NIR)^2)
        """
        red = bands[band_map["red"]].astype(float)
        nir = bands[band_map["nir"]].astype(float)
        
        return 1 / ((0.1 - red)**2 + (0.06 - nir)**2 + 1e-10)
    
    def _calculate_nbr(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Normalized Burn Ratio.
        NBR = (NIR - SWIR) / (NIR + SWIR)
        Note: Approximated using Red band as SWIR substitute
        """
        nir = bands[band_map["nir"]].astype(float)
        swir_proxy = bands[band_map["red"]].astype(float)
        
        return (nir - swir_proxy) / (nir + swir_proxy + 1e-10)
    
    def _calculate_gndvi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Green Normalized Difference Vegetation Index.
        GNDVI = (NIR - Green) / (NIR + Green)
        """
        nir = bands[band_map["nir"]].astype(float)
        green = bands[band_map["green"]].astype(float)
        
        return (nir - green) / (nir + green + 1e-10)
    
    def _calculate_ndmi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Normalized Difference Moisture Index.
        NDMI = (NIR - SWIR) / (NIR + SWIR)
        Note: Approximated using Red band as SWIR substitute
        """
        nir = bands[band_map["nir"]].astype(float)
        swir_proxy = bands[band_map["red"]].astype(float)
        
        return (nir - swir_proxy) / (nir + swir_proxy + 1e-10)
    
    def _calculate_arvi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Atmospherically Resistant Vegetation Index.
        ARVI = (NIR - (2*Red - Blue)) / (NIR + (2*Red - Blue))
        """
        nir = bands[band_map["nir"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        blue = bands[band_map["blue"]].astype(float)
        
        rb = 2*red - blue
        return (nir - rb) / (nir + rb + 1e-10)
    
    def _calculate_gci(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Green Chlorophyll Index.
        GCI = (NIR / Green) - 1
        """
        nir = bands[band_map["nir"]].astype(float)
        green = bands[band_map["green"]].astype(float)
        
        return (nir / (green + 1e-10)) - 1
    
    def _calculate_sipi(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Structure Insensitive Pigment Index.
        SIPI = (NIR - Blue) / (NIR - Red)
        """
        nir = bands[band_map["nir"]].astype(float)
        blue = bands[band_map["blue"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        
        return (nir - blue) / (nir - red + 1e-10)
    
    def _calculate_vari(self, bands: np.ndarray, band_map: Dict) -> np.ndarray:
        """
        Visible Atmospherically Resistant Index.
        VARI = (Green - Red) / (Green + Red - Blue)
        """
        green = bands[band_map["green"]].astype(float)
        red = bands[band_map["red"]].astype(float)
        blue = bands[band_map["blue"]].astype(float)
        
        return (green - red) / (green + red - blue + 1e-10)
