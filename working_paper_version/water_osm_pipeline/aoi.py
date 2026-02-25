"""
Area of Interest (AOI) input handler.

Supports bounding boxes, GeoJSON files, shapefiles, coordinate lists, and WKT.
All outputs are normalised to a GeoJSON geometry dict (EPSG:4326).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class AOIConfig:
    """AOI handling configuration."""
    default_crs: str = "EPSG:4326"
    simplify_tolerance: float = 0.0001   # degrees (~11 m at equator)


class AOIHandler:
    """
    Unified AOI input handler.

    Normalises any supported input to a GeoJSON geometry dict.

    Examples::

        aoi = AOIHandler.from_bbox(34.5, 31.2, 35.5, 32.1)
        aoi = AOIHandler.load("region.geojson")
        aoi = AOIHandler.load([34.5, 31.2, 35.5, 32.1])   # bbox list
    """

    def __init__(self, config: Optional[AOIConfig] = None):
        self.config = config or AOIConfig()

    # ── constructors ────────────────────────────────────────────────────────

    @staticmethod
    def from_bbox(west: float, south: float, east: float, north: float) -> Dict:
        """Return a GeoJSON Polygon geometry for the given bounding box."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [west, south], [east, south],
                [east, north], [west, north],
                [west, south],
            ]],
        }

    @staticmethod
    def from_coordinates(
        coords: Union[List, Tuple],
        coord_type: str = "polygon",
    ) -> Dict:
        """
        Create a GeoJSON geometry from a coordinate list.

        Args:
            coords: ``[lon, lat]`` pairs for polygons/points, or
                    ``[west, south, east, north]`` for bbox.
            coord_type: ``"polygon"``, ``"point"``, or ``"bbox"``.
        """
        if coord_type == "point":
            if len(coords) != 2:
                raise ValueError("Point requires [longitude, latitude]")
            return {"type": "Point", "coordinates": list(coords)}

        if coord_type == "bbox":
            if len(coords) != 4:
                raise ValueError("Bbox requires [west, south, east, north]")
            west, south, east, north = coords
            return AOIHandler.from_bbox(west, south, east, north)

        # polygon
        if coords and isinstance(coords[0][0], (list, tuple)):
            polygon_coords = list(coords)          # already nested
        else:
            polygon_coords = [list(coords)]        # flat ring

        for ring in polygon_coords:
            if list(ring[0]) != list(ring[-1]):
                ring.append(ring[0])

        return {"type": "Polygon", "coordinates": polygon_coords}

    @staticmethod
    def from_geojson_file(filepath: Union[str, Path]) -> Dict:
        """Load AOI geometry from a GeoJSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")
        with open(filepath) as fh:
            data = json.load(fh)
        return AOIHandler._extract_geometry(data)

    @staticmethod
    def from_shapefile(filepath: Union[str, Path]) -> Dict:
        """Load AOI geometry from a shapefile (requires geopandas)."""
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas required: pip install geopandas")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Shapefile not found: {filepath}")

        gdf = gpd.read_file(filepath)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return mapping(unary_union(gdf.geometry))

    @staticmethod
    def from_wkt(wkt_string: str) -> Dict:
        """Load AOI geometry from a WKT string."""
        from shapely import wkt
        return mapping(wkt.loads(wkt_string))

    @classmethod
    def load(cls, source: Union[str, Path, Dict, List, Tuple]) -> Dict:
        """
        Auto-detect input type and return a GeoJSON geometry dict.

        Supported types:
            - ``str`` / ``Path``: file path (.geojson, .json, .shp) or WKT string
            - ``dict``: GeoJSON Feature, FeatureCollection, or Geometry
            - ``list`` / ``tuple``: coordinate list or ``[W, S, E, N]`` bbox
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if isinstance(source, str) and not path.suffix:
                keywords = ["POLYGON", "POINT", "LINESTRING", "MULTIPOLYGON"]
                if any(kw in source.upper() for kw in keywords):
                    return cls.from_wkt(source)

            if path.suffix.lower() in {".geojson", ".json"}:
                return cls.from_geojson_file(path)
            if path.suffix.lower() == ".shp":
                return cls.from_shapefile(path)
            try:
                return cls.from_geojson_file(path)
            except (json.JSONDecodeError, FileNotFoundError):
                return cls.from_shapefile(path)

        if isinstance(source, dict):
            return cls._extract_geometry(source)

        if isinstance(source, (list, tuple)):
            if len(source) == 4 and all(isinstance(x, (int, float)) for x in source):
                return cls.from_coordinates(source, coord_type="bbox")
            if len(source) == 2 and all(isinstance(x, (int, float)) for x in source):
                return cls.from_coordinates(source, coord_type="point")
            return cls.from_coordinates(source, coord_type="polygon")

        raise TypeError(f"Unsupported source type: {type(source)}")

    # ── geometry helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_geometry(data: Dict) -> Dict:
        """Extract a GeoJSON geometry dict from any GeoJSON type."""
        geom_type = data.get("type")

        if geom_type == "FeatureCollection":
            geometries = [
                shape(f["geometry"])
                for f in data.get("features", [])
                if f.get("geometry")
            ]
            if not geometries:
                raise ValueError("FeatureCollection contains no valid geometries")
            return mapping(unary_union(geometries))

        if geom_type == "Feature":
            if not data.get("geometry"):
                raise ValueError("Feature has no geometry")
            return data["geometry"]

        valid_types = {
            "Polygon", "MultiPolygon", "Point", "LineString",
            "MultiPoint", "MultiLineString", "GeometryCollection",
        }
        if geom_type in valid_types:
            return data

        raise ValueError(f"Unknown GeoJSON type: {geom_type!r}")

    @staticmethod
    def get_bounds(geometry: Dict) -> Tuple[float, float, float, float]:
        """Return ``(west, south, east, north)`` bounding box."""
        return shape(geometry).bounds   # (minx, miny, maxx, maxy)

    @staticmethod
    def get_centroid(geometry: Dict) -> Tuple[float, float]:
        """Return ``(longitude, latitude)`` centroid."""
        c = shape(geometry).centroid
        return (c.x, c.y)

    @staticmethod
    def buffer_precise(geometry: Dict, meters: float) -> Dict:
        """
        Buffer geometry by *meters* using an AEQD local projection.

        More accurate than a simple degree-based buffer.
        Requires pyproj.
        """
        try:
            import pyproj
            from shapely.ops import transform
        except ImportError:
            raise ImportError("pyproj required: pip install pyproj")

        geom = shape(geometry)
        c = geom.centroid
        proj_str = f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m +datum=WGS84"

        to_local = pyproj.Transformer.from_crs(
            "EPSG:4326", proj_str, always_xy=True
        ).transform
        to_wgs84 = pyproj.Transformer.from_crs(
            proj_str, "EPSG:4326", always_xy=True
        ).transform

        from shapely.ops import transform as shp_transform
        return mapping(shp_transform(to_wgs84, shp_transform(to_local, geom).buffer(meters)))

    @staticmethod
    def simplify(geometry: Dict, tolerance: float = 0.0001) -> Dict:
        """Simplify geometry while preserving topology."""
        return mapping(shape(geometry).simplify(tolerance, preserve_topology=True))

    @staticmethod
    def area_sq_km(geometry: Dict) -> float:
        """Approximate area in square kilometres (AEQD projection)."""
        try:
            import pyproj
            from shapely.ops import transform
        except ImportError:
            # Very rough fallback
            import math
            geom = shape(geometry)
            c = geom.centroid
            b = geom.bounds
            return (b[2] - b[0]) * 111 * math.cos(math.radians(c.y)) * (b[3] - b[1]) * 111

        geom = shape(geometry)
        c = geom.centroid
        proj_str = (
            f"+proj=aea +lat_1={c.y-1} +lat_2={c.y+1} "
            f"+lat_0={c.y} +lon_0={c.x}"
        )
        project = pyproj.Transformer.from_crs("EPSG:4326", proj_str, always_xy=True).transform
        from shapely.ops import transform
        return transform(project, geom).area / 1_000_000


def load_aoi(source) -> Dict:
    """Convenience wrapper around :meth:`AOIHandler.load`."""
    return AOIHandler.load(source)
