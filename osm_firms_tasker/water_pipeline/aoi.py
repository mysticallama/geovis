"""
Area of Interest (AOI) input handler module.

Supports loading AOIs from:
- Raw coordinates (polygon, point, bounding box)
- GeoJSON files
- Shapefiles
- WKT strings
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    box,
    mapping,
    shape,
)
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class AOIConfig:
    """AOI handling configuration."""
    default_crs: str = "EPSG:4326"
    simplify_tolerance: float = 0.0001  # degrees (~11m at equator)
    buffer_meters: float = 0.0


class AOIHandler:
    """
    Unified AOI input handler.

    Supports multiple input formats and normalizes to GeoJSON geometry.

    Examples:
        # From bounding box
        aoi = AOIHandler.from_bbox(-122.5, 37.7, -122.3, 37.9)

        # From coordinates
        aoi = AOIHandler.from_coordinates([
            [-122.5, 37.7], [-122.3, 37.7], [-122.3, 37.9],
            [-122.5, 37.9], [-122.5, 37.7]
        ])

        # From file (auto-detect format)
        aoi = AOIHandler.load("region.geojson")
        aoi = AOIHandler.load("region.shp")

        # From GeoJSON dict
        aoi = AOIHandler.load({"type": "Polygon", "coordinates": [...]})
    """

    def __init__(self, config: Optional[AOIConfig] = None):
        self.config = config or AOIConfig()

    @staticmethod
    def from_coordinates(
        coords: Union[List, Tuple],
        coord_type: str = "polygon"
    ) -> Dict:
        """
        Create GeoJSON geometry from coordinates.

        Args:
            coords: Coordinates in format:
                - polygon: [[lon, lat], [lon, lat], ...] or [[[lon, lat], ...]]
                - point: [lon, lat]
                - bbox: [west, south, east, north]
            coord_type: "polygon", "point", or "bbox"

        Returns:
            GeoJSON geometry dict
        """
        if coord_type == "point":
            if len(coords) != 2:
                raise ValueError("Point requires [longitude, latitude]")
            return {"type": "Point", "coordinates": list(coords)}

        elif coord_type == "bbox":
            if len(coords) != 4:
                raise ValueError("Bbox requires [west, south, east, north]")
            west, south, east, north = coords
            return AOIHandler.from_bbox(west, south, east, north)

        elif coord_type == "polygon":
            # Handle nested vs flat coordinate arrays
            if coords and isinstance(coords[0][0], (list, tuple)):
                # Already in polygon format [[ring1], [ring2], ...]
                polygon_coords = coords
            else:
                # Flat list of coordinates
                polygon_coords = [coords]

            # Ensure ring is closed
            for ring in polygon_coords:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])

            return {"type": "Polygon", "coordinates": polygon_coords}

        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    @staticmethod
    def from_bbox(
        west: float,
        south: float,
        east: float,
        north: float
    ) -> Dict:
        """
        Create polygon geometry from bounding box.

        Args:
            west: Western longitude
            south: Southern latitude
            east: Eastern longitude
            north: Northern latitude

        Returns:
            GeoJSON Polygon geometry
        """
        return {
            "type": "Polygon",
            "coordinates": [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south]
            ]]
        }

    @staticmethod
    def from_geojson_file(filepath: Union[str, Path]) -> Dict:
        """
        Load AOI from GeoJSON file.

        Handles Feature, FeatureCollection, and raw Geometry objects.
        For FeatureCollections, returns the union of all geometries.

        Args:
            filepath: Path to GeoJSON file

        Returns:
            GeoJSON geometry dict
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        return AOIHandler._extract_geometry(data)

    @staticmethod
    def from_shapefile(filepath: Union[str, Path], layer: int = 0) -> Dict:
        """
        Load AOI from shapefile.

        Requires geopandas to be installed.
        For multi-feature shapefiles, returns the union of all geometries.

        Args:
            filepath: Path to shapefile (.shp)
            layer: Layer index for multi-layer files

        Returns:
            GeoJSON geometry dict
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for shapefile support. "
                "Install with: pip install geopandas"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Shapefile not found: {filepath}")

        gdf = gpd.read_file(filepath, layer=layer)

        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        # Union all geometries
        unified = unary_union(gdf.geometry)

        return mapping(unified)

    @staticmethod
    def from_wkt(wkt_string: str) -> Dict:
        """
        Load AOI from WKT string.

        Args:
            wkt_string: Well-Known Text geometry string

        Returns:
            GeoJSON geometry dict
        """
        from shapely import wkt
        geom = wkt.loads(wkt_string)
        return mapping(geom)

    @classmethod
    def load(cls, source: Union[str, Path, Dict, List, Tuple]) -> Dict:
        """
        Auto-detect input type and return GeoJSON geometry.

        Supports:
            - File path (str/Path): .geojson, .json, .shp
            - Dict: GeoJSON Feature, FeatureCollection, or Geometry
            - List/Tuple: Coordinates (auto-detect polygon vs bbox)

        Args:
            source: AOI source in any supported format

        Returns:
            GeoJSON geometry dict
        """
        # Handle file paths
        if isinstance(source, (str, Path)):
            path = Path(source)

            # Check if it's a WKT string (no file extension and contains geometry keywords)
            if isinstance(source, str) and not path.suffix:
                wkt_keywords = ["POLYGON", "POINT", "LINESTRING", "MULTIPOLYGON"]
                if any(kw in source.upper() for kw in wkt_keywords):
                    return cls.from_wkt(source)

            # File-based loading
            if path.suffix.lower() in [".geojson", ".json"]:
                return cls.from_geojson_file(path)
            elif path.suffix.lower() == ".shp":
                return cls.from_shapefile(path)
            else:
                # Try GeoJSON first, then shapefile
                try:
                    return cls.from_geojson_file(path)
                except (json.JSONDecodeError, FileNotFoundError):
                    return cls.from_shapefile(path)

        # Handle dict (GeoJSON)
        elif isinstance(source, dict):
            return cls._extract_geometry(source)

        # Handle coordinates
        elif isinstance(source, (list, tuple)):
            # Detect if it's a bbox or polygon coordinates
            if len(source) == 4 and all(isinstance(x, (int, float)) for x in source):
                return cls.from_coordinates(source, coord_type="bbox")
            elif len(source) == 2 and all(isinstance(x, (int, float)) for x in source):
                return cls.from_coordinates(source, coord_type="point")
            else:
                return cls.from_coordinates(source, coord_type="polygon")

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def _extract_geometry(data: Dict) -> Dict:
        """Extract geometry from GeoJSON dict."""
        geom_type = data.get("type")

        if geom_type == "FeatureCollection":
            # Union all feature geometries
            geometries = [
                shape(f["geometry"])
                for f in data.get("features", [])
                if f.get("geometry")
            ]
            if not geometries:
                raise ValueError("FeatureCollection contains no valid geometries")
            unified = unary_union(geometries)
            return mapping(unified)

        elif geom_type == "Feature":
            if not data.get("geometry"):
                raise ValueError("Feature has no geometry")
            return data["geometry"]

        elif geom_type in ["Polygon", "MultiPolygon", "Point", "LineString",
                          "MultiPoint", "MultiLineString", "GeometryCollection"]:
            return data

        else:
            raise ValueError(f"Unknown GeoJSON type: {geom_type}")

    @staticmethod
    def get_bounds(geometry: Dict) -> Tuple[float, float, float, float]:
        """
        Extract bounding box from geometry.

        Args:
            geometry: GeoJSON geometry dict

        Returns:
            Tuple of (west, south, east, north)
        """
        geom = shape(geometry)
        return geom.bounds  # (minx, miny, maxx, maxy)

    @staticmethod
    def get_centroid(geometry: Dict) -> Tuple[float, float]:
        """
        Get centroid of geometry.

        Args:
            geometry: GeoJSON geometry dict

        Returns:
            Tuple of (longitude, latitude)
        """
        geom = shape(geometry)
        centroid = geom.centroid
        return (centroid.x, centroid.y)

    @staticmethod
    def buffer(geometry: Dict, meters: float) -> Dict:
        """
        Buffer geometry by specified distance in meters.

        Uses approximate degree conversion (accurate near equator,
        less accurate at high latitudes).

        Args:
            geometry: GeoJSON geometry dict
            meters: Buffer distance in meters

        Returns:
            Buffered GeoJSON geometry
        """
        # Approximate degree conversion (1 degree ~ 111km at equator)
        degrees = meters / 111000

        geom = shape(geometry)
        buffered = geom.buffer(degrees)

        return mapping(buffered)

    @staticmethod
    def buffer_precise(geometry: Dict, meters: float) -> Dict:
        """
        Buffer geometry with precise meter distance using projection.

        Projects to local UTM zone, buffers, and projects back.
        Requires pyproj.

        Args:
            geometry: GeoJSON geometry dict
            meters: Buffer distance in meters

        Returns:
            Buffered GeoJSON geometry
        """
        try:
            import pyproj
            from shapely.ops import transform
        except ImportError:
            raise ImportError(
                "pyproj is required for precise buffering. "
                "Install with: pip install pyproj"
            )

        geom = shape(geometry)
        centroid = geom.centroid

        # Determine UTM zone
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"

        # Create transformers
        project_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", utm_crs, always_xy=True
        ).transform
        project_to_wgs = pyproj.Transformer.from_crs(
            utm_crs, "EPSG:4326", always_xy=True
        ).transform

        # Transform, buffer, transform back
        geom_utm = transform(project_to_utm, geom)
        buffered_utm = geom_utm.buffer(meters)
        buffered_wgs = transform(project_to_wgs, buffered_utm)

        return mapping(buffered_wgs)

    @staticmethod
    def simplify(geometry: Dict, tolerance: float = 0.0001) -> Dict:
        """
        Simplify geometry to reduce complexity.

        Args:
            geometry: GeoJSON geometry dict
            tolerance: Simplification tolerance in degrees

        Returns:
            Simplified GeoJSON geometry
        """
        geom = shape(geometry)
        simplified = geom.simplify(tolerance, preserve_topology=True)
        return mapping(simplified)

    @staticmethod
    def to_wkt(geometry: Dict) -> str:
        """
        Convert geometry to WKT string.

        Args:
            geometry: GeoJSON geometry dict

        Returns:
            WKT string representation
        """
        geom = shape(geometry)
        return geom.wkt

    @staticmethod
    def area_sq_km(geometry: Dict) -> float:
        """
        Calculate approximate area in square kilometers.

        Uses spherical approximation, accurate for small areas.

        Args:
            geometry: GeoJSON geometry dict

        Returns:
            Area in square kilometers
        """
        try:
            import pyproj
            from shapely.ops import transform
        except ImportError:
            # Fallback to rough approximation
            geom = shape(geometry)
            # 1 degree lat ~ 111km, 1 degree lon varies
            centroid = geom.centroid
            import math
            lon_scale = math.cos(math.radians(centroid.y))
            # Very rough approximation
            bounds = geom.bounds
            width_km = (bounds[2] - bounds[0]) * 111 * lon_scale
            height_km = (bounds[3] - bounds[1]) * 111
            return width_km * height_km * 0.8  # Rough polygon area factor

        geom = shape(geometry)
        centroid = geom.centroid

        # Project to equal-area projection
        aea_crs = f"+proj=aea +lat_1={centroid.y-1} +lat_2={centroid.y+1} +lat_0={centroid.y} +lon_0={centroid.x}"

        project = pyproj.Transformer.from_crs(
            "EPSG:4326", aea_crs, always_xy=True
        ).transform

        geom_projected = transform(project, geom)
        area_sq_m = geom_projected.area

        return area_sq_m / 1_000_000


def load_aoi(source: Union[str, Path, Dict, List, Tuple]) -> Dict:
    """
    Convenience function to load AOI from any supported format.

    Args:
        source: AOI in any supported format

    Returns:
        GeoJSON geometry dict
    """
    return AOIHandler.load(source)
