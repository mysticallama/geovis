"""
Image Preprocessing Module

Preprocessing operations for Planet imagery including cloud masking,
atmospheric correction, and radiometric calibration.
"""

import numpy as np
import rasterio
from rasterio.mask import mask
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocessing operations for satellite imagery."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.operations = {
            "cloud_mask": self._apply_cloud_mask,
            "radiometric_calibration": self._radiometric_calibration,
            "atmospheric_correction": self._atmospheric_correction,
            "normalization": self._normalize,
            "histogram_equalization": self._histogram_equalization
        }
    
    def process(
        self,
        input_path: Path,
        operations: List[str],
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Apply preprocessing operations.
        
        Args:
            input_path: Input imagery path
            operations: List of operations to apply
            output_dir: Output directory
            **kwargs: Additional parameters for operations
        
        Returns:
            Path to processed imagery
        """
        logger.info(f"Preprocessing {input_path.name}")
        
        # Read input
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
        
        # Apply operations sequentially
        for operation in operations:
            if operation not in self.operations:
                logger.warning(f"Unknown operation: {operation}")
                continue
            
            logger.debug(f"Applying {operation}")
            data = self.operations[operation](data, **kwargs)
        
        # Save output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_processed.tif"
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        
        logger.info(f"Saved processed imagery to {output_path}")
        return output_path
    
    def _apply_cloud_mask(
        self,
        data: np.ndarray,
        cloud_threshold: float = 0.3,
        **kwargs
    ) -> np.ndarray:
        """
        Apply simple cloud mask based on brightness threshold.
        
        Args:
            data: Input imagery array
            cloud_threshold: Brightness threshold for cloud detection
        
        Returns:
            Masked imagery array
        """
        # Simple cloud detection: high reflectance in visible bands
        if data.shape[0] >= 3:
            # Use mean of RGB bands
            rgb_mean = np.mean(data[:3], axis=0)
            cloud_mask = rgb_mean > cloud_threshold
            
            # Apply mask to all bands
            data_masked = data.copy()
            data_masked[:, cloud_mask] = 0
            
            logger.info(f"Masked {np.sum(cloud_mask) / cloud_mask.size * 100:.2f}% cloudy pixels")
            return data_masked
        
        return data
    
    def _radiometric_calibration(
        self,
        data: np.ndarray,
        calibration_coeffs: Optional[Dict] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply radiometric calibration to convert DN to reflectance.
        
        Args:
            data: Input imagery array (DN values)
            calibration_coeffs: Dict of calibration coefficients per band
        
        Returns:
            Calibrated imagery in reflectance units
        """
        # Planet imagery often comes pre-calibrated
        # This is a placeholder for custom calibration
        
        if calibration_coeffs is None:
            # Use default Planet reflectance scaling
            # DN values in Planet ortho products are typically scaled reflectance
            return data.astype(np.float32) / 10000.0
        
        calibrated = np.zeros_like(data, dtype=np.float32)
        for band_idx in range(data.shape[0]):
            coeff = calibration_coeffs.get(band_idx, 1.0)
            calibrated[band_idx] = data[band_idx] * coeff
        
        return calibrated
    
    def _atmospheric_correction(
        self,
        data: np.ndarray,
        method: str = "dos",
        **kwargs
    ) -> np.ndarray:
        """
        Apply atmospheric correction.
        
        Args:
            data: Input imagery array
            method: Correction method ("dos" for Dark Object Subtraction)
        
        Returns:
            Atmospherically corrected imagery
        """
        if method == "dos":
            # Dark Object Subtraction - simple atmospheric correction
            corrected = np.zeros_like(data, dtype=np.float32)
            
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx].astype(np.float32)
                
                # Find dark object value (1st percentile)
                dark_value = np.percentile(band_data[band_data > 0], 1)
                
                # Subtract dark value
                corrected[band_idx] = np.maximum(band_data - dark_value, 0)
            
            logger.info("Applied Dark Object Subtraction atmospheric correction")
            return corrected
        
        return data
    
    def _normalize(
        self,
        data: np.ndarray,
        method: str = "minmax",
        **kwargs
    ) -> np.ndarray:
        """
        Normalize imagery values.
        
        Args:
            data: Input imagery array
            method: Normalization method ("minmax", "zscore", "percentile")
        
        Returns:
            Normalized imagery
        """
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for band_idx in range(data.shape[0]):
            band_data = data[band_idx].astype(np.float32)
            
            if method == "minmax":
                # Min-max normalization to [0, 1]
                min_val = np.min(band_data)
                max_val = np.max(band_data)
                if max_val > min_val:
                    normalized[band_idx] = (band_data - min_val) / (max_val - min_val)
                else:
                    normalized[band_idx] = band_data
            
            elif method == "zscore":
                # Z-score normalization
                mean = np.mean(band_data)
                std = np.std(band_data)
                if std > 0:
                    normalized[band_idx] = (band_data - mean) / std
                else:
                    normalized[band_idx] = band_data
            
            elif method == "percentile":
                # Percentile-based normalization (2-98 percentile)
                p2 = np.percentile(band_data, 2)
                p98 = np.percentile(band_data, 98)
                if p98 > p2:
                    normalized[band_idx] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                else:
                    normalized[band_idx] = band_data
        
        logger.info(f"Applied {method} normalization")
        return normalized
    
    def _histogram_equalization(
        self,
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            data: Input imagery array
        
        Returns:
            Equalized imagery
        """
        equalized = np.zeros_like(data, dtype=np.float32)
        
        for band_idx in range(data.shape[0]):
            band_data = data[band_idx].flatten()
            
            # Compute histogram
            hist, bins = np.histogram(band_data, bins=256, range=(0, 1))
            
            # Compute cumulative distribution
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            
            # Interpolate to get equalized values
            equalized[band_idx] = np.interp(
                data[band_idx].flatten(),
                bins[:-1],
                cdf_normalized
            ).reshape(data[band_idx].shape)
        
        logger.info("Applied histogram equalization")
        return equalized
    
    def clip_to_aoi(
        self,
        input_path: Path,
        geometry: Dict,
        output_path: Path
    ) -> Path:
        """
        Clip imagery to AOI geometry.
        
        Args:
            input_path: Input imagery path
            geometry: GeoJSON geometry
            output_path: Output path
        
        Returns:
            Path to clipped imagery
        """
        with rasterio.open(input_path) as src:
            # Clip to geometry
            clipped, transform = mask(src, [geometry], crop=True)
            
            # Update metadata
            profile = src.profile.copy()
            profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform
            })
        
        # Write clipped imagery
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(clipped)
        
        logger.info(f"Clipped imagery to AOI: {output_path}")
        return output_path
    
    def create_rgb_composite(
        self,
        input_path: Path,
        output_path: Path,
        red_band: int = 2,
        green_band: int = 1,
        blue_band: int = 0,
        stretch: bool = True
    ) -> Path:
        """
        Create RGB composite for visualization.
        
        Args:
            input_path: Input multispectral imagery
            output_path: Output RGB path
            red_band: Red band index
            green_band: Green band index
            blue_band: Blue band index
            stretch: Apply contrast stretch
        
        Returns:
            Path to RGB composite
        """
        with rasterio.open(input_path) as src:
            # Read RGB bands
            red = src.read(red_band + 1).astype(np.float32)
            green = src.read(green_band + 1).astype(np.float32)
            blue = src.read(blue_band + 1).astype(np.float32)
            
            profile = src.profile.copy()
        
        # Stack bands
        rgb = np.stack([red, green, blue], axis=0)
        
        # Apply stretch if requested
        if stretch:
            for i in range(3):
                band = rgb[i]
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                rgb[i] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
        
        # Convert to 8-bit
        rgb_8bit = (rgb * 255).astype(np.uint8)
        
        # Update profile
        profile.update({
            "count": 3,
            "dtype": rasterio.uint8
        })
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rgb_8bit)
        
        logger.info(f"Created RGB composite: {output_path}")
        return output_path
