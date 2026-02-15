"""
Image processing module for Planet imagery.

Handles preprocessing, spectral indices, and raster operations.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window

logger = logging.getLogger(__name__)


# =============================================================================
# Band Mappings
# =============================================================================

PLANET_4B = {"blue": 0, "green": 1, "red": 2, "nir": 3}
PLANET_8B = {
    "coastal_blue": 0, "blue": 1, "green_i": 2, "green": 3,
    "yellow": 4, "red": 5, "red_edge": 6, "nir": 7
}


# =============================================================================
# Spectral Indices
# =============================================================================

def _safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safe division avoiding divide by zero."""
    return a / (b + eps)


def ndvi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)"""
    nir, red = bands[band_map["nir"]].astype(float), bands[band_map["red"]].astype(float)
    return _safe_divide(nir - red, nir + red)


def ndwi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)"""
    green, nir = bands[band_map["green"]].astype(float), bands[band_map["nir"]].astype(float)
    return _safe_divide(green - nir, green + nir)


def evi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Enhanced Vegetation Index: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
    nir = bands[band_map["nir"]].astype(float)
    red = bands[band_map["red"]].astype(float)
    blue = bands[band_map["blue"]].astype(float)
    return 2.5 * _safe_divide(nir - red, nir + 6*red - 7.5*blue + 1)


def savi(bands: np.ndarray, band_map: Dict, L: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index: ((NIR - Red) / (NIR + Red + L)) * (1 + L)"""
    nir, red = bands[band_map["nir"]].astype(float), bands[band_map["red"]].astype(float)
    return _safe_divide(nir - red, nir + red + L) * (1 + L)


def msavi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Modified Soil-Adjusted Vegetation Index."""
    nir, red = bands[band_map["nir"]].astype(float), bands[band_map["red"]].astype(float)
    return (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red))) / 2


def gndvi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Green NDVI: (NIR - Green) / (NIR + Green)"""
    nir, green = bands[band_map["nir"]].astype(float), bands[band_map["green"]].astype(float)
    return _safe_divide(nir - green, nir + green)


def arvi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Atmospherically Resistant Vegetation Index."""
    nir = bands[band_map["nir"]].astype(float)
    red = bands[band_map["red"]].astype(float)
    blue = bands[band_map["blue"]].astype(float)
    rb = 2*red - blue
    return _safe_divide(nir - rb, nir + rb)


def vari(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Visible Atmospherically Resistant Index: (Green - Red) / (Green + Red - Blue)"""
    green = bands[band_map["green"]].astype(float)
    red = bands[band_map["red"]].astype(float)
    blue = bands[band_map["blue"]].astype(float)
    return _safe_divide(green - red, green + red - blue)


def bai(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Burned Area Index: 1 / ((0.1 - Red)^2 + (0.06 - NIR)^2)"""
    red, nir = bands[band_map["red"]].astype(float), bands[band_map["nir"]].astype(float)
    return 1 / ((0.1 - red)**2 + (0.06 - nir)**2 + 1e-10)


def gci(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Green Chlorophyll Index: (NIR / Green) - 1"""
    nir, green = bands[band_map["nir"]].astype(float), bands[band_map["green"]].astype(float)
    return _safe_divide(nir, green) - 1


def sipi(bands: np.ndarray, band_map: Dict) -> np.ndarray:
    """Structure Insensitive Pigment Index: (NIR - Blue) / (NIR - Red)"""
    nir = bands[band_map["nir"]].astype(float)
    blue = bands[band_map["blue"]].astype(float)
    red = bands[band_map["red"]].astype(float)
    return _safe_divide(nir - blue, nir - red)


# Registry of all indices
INDICES: Dict[str, Callable] = {
    "ndvi": ndvi, "ndwi": ndwi, "evi": evi, "savi": savi, "msavi": msavi,
    "gndvi": gndvi, "arvi": arvi, "vari": vari, "bai": bai, "gci": gci, "sipi": sipi
}


# =============================================================================
# Image Processor
# =============================================================================

class ImageProcessor:
    """Process satellite imagery: indices, normalization, composites."""

    @staticmethod
    def calculate_index(
        imagery_path: Union[str, Path],
        index_name: str,
        output_path: Optional[Union[str, Path]] = None,
        nodata: float = -9999
    ) -> Path:
        """
        Calculate a spectral index.

        Args:
            imagery_path: Path to input GeoTIFF
            index_name: Index name (ndvi, ndwi, evi, savi, etc.)
            output_path: Output path (default: {input}_{index}.tif)
            nodata: NoData value

        Returns:
            Path to output file
        """
        if index_name not in INDICES:
            raise ValueError(f"Unknown index: {index_name}. Available: {list(INDICES.keys())}")

        imagery_path = Path(imagery_path)
        output_path = Path(output_path) if output_path else imagery_path.parent / f"{imagery_path.stem}_{index_name}.tif"

        with rasterio.open(imagery_path) as src:
            bands = src.read()
            profile = src.profile.copy()
            band_map = PLANET_4B if src.count == 4 else PLANET_8B if src.count == 8 else PLANET_4B

        result = INDICES[index_name](bands, band_map)
        result[~np.isfinite(result)] = nodata

        profile.update(count=1, dtype=rasterio.float32, nodata=nodata)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result.astype(np.float32), 1)

        logger.info(f"Calculated {index_name}: {output_path}")
        return output_path

    @staticmethod
    def calculate_indices(
        imagery_path: Union[str, Path],
        indices: List[str],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """Calculate multiple indices for an image."""
        results = {}
        for idx in indices:
            try:
                out_path = None
                if output_dir:
                    out_path = Path(output_dir) / f"{Path(imagery_path).stem}_{idx}.tif"
                results[idx] = ImageProcessor.calculate_index(imagery_path, idx, out_path)
            except Exception as e:
                logger.error(f"Failed {idx}: {e}")
        return results

    @staticmethod
    def normalize(
        data: np.ndarray,
        method: str = "percentile",
        per_band: bool = True
    ) -> np.ndarray:
        """
        Normalize imagery.

        Args:
            data: Input array (bands, height, width)
            method: "percentile" (2-98), "minmax", or "zscore"
            per_band: Normalize each band independently

        Returns:
            Normalized array
        """
        result = np.zeros_like(data, dtype=np.float32)

        for i in range(data.shape[0]) if per_band else [slice(None)]:
            band = data[i].astype(np.float32)
            valid = band[band > 0] if np.any(band > 0) else band.flatten()

            if method == "percentile":
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    result[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            elif method == "minmax":
                vmin, vmax = valid.min(), valid.max()
                if vmax > vmin:
                    result[i] = (band - vmin) / (vmax - vmin)
            elif method == "zscore":
                mean, std = valid.mean(), valid.std()
                if std > 0:
                    result[i] = (band - mean) / std

        return result

    @staticmethod
    def clip_to_geometry(
        imagery_path: Union[str, Path],
        geometry: Dict,
        output_path: Union[str, Path]
    ) -> Path:
        """Clip imagery to a GeoJSON geometry."""
        with rasterio.open(imagery_path) as src:
            clipped, transform = rio_mask(src, [geometry], crop=True)
            profile = src.profile.copy()
            profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(clipped)

        return output_path

    @staticmethod
    def create_rgb(
        imagery_path: Union[str, Path],
        output_path: Union[str, Path],
        bands: tuple = (2, 1, 0),
        stretch: bool = True
    ) -> Path:
        """Create RGB composite for visualization."""
        with rasterio.open(imagery_path) as src:
            rgb = np.stack([src.read(b + 1).astype(np.float32) for b in bands])
            profile = src.profile.copy()

        if stretch:
            for i in range(3):
                p2, p98 = np.percentile(rgb[i], [2, 98])
                rgb[i] = np.clip((rgb[i] - p2) / (p98 - p2 + 1e-10), 0, 1)

        rgb_8bit = (rgb * 255).astype(np.uint8)
        profile.update(count=3, dtype=rasterio.uint8)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rgb_8bit)

        return output_path

    @staticmethod
    def apply_cloud_mask(
        data: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """Simple cloud mask based on brightness."""
        if data.shape[0] >= 3:
            brightness = np.mean(data[:3], axis=0)
            mask = brightness > threshold
            data = data.copy()
            data[:, mask] = 0
        return data

    @staticmethod
    def atmospheric_correction(
        data: np.ndarray,
        method: str = "dos"
    ) -> np.ndarray:
        """Dark Object Subtraction atmospheric correction."""
        result = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            band = data[i].astype(np.float32)
            dark = np.percentile(band[band > 0], 1) if np.any(band > 0) else 0
            result[i] = np.maximum(band - dark, 0)
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def list_indices() -> List[str]:
    """List available spectral indices."""
    return list(INDICES.keys())


def process_imagery(
    imagery_path: Union[str, Path],
    output_dir: Union[str, Path],
    indices: List[str] = None,
    normalize: bool = False,
    cloud_mask: bool = False
) -> Dict[str, Path]:
    """
    Process imagery with multiple operations.

    Args:
        imagery_path: Input imagery path
        output_dir: Output directory
        indices: List of indices to calculate
        normalize: Apply normalization
        cloud_mask: Apply cloud masking

    Returns:
        Dict of output paths
    """
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate indices
    if indices:
        for idx in indices:
            out_path = output_dir / f"{Path(imagery_path).stem}_{idx}.tif"
            results[idx] = ImageProcessor.calculate_index(imagery_path, idx, out_path)

    # Process full image if requested
    if normalize or cloud_mask:
        with rasterio.open(imagery_path) as src:
            data = src.read()
            profile = src.profile.copy()

        if cloud_mask:
            data = ImageProcessor.apply_cloud_mask(data)
        if normalize:
            data = ImageProcessor.normalize(data)

        out_path = output_dir / f"{Path(imagery_path).stem}_processed.tif"
        profile.update(dtype=rasterio.float32)

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data.astype(np.float32))

        results["processed"] = out_path

    return results
