"""
Export module for satellite APIs and ML models.

Exports GeoJSONs in formats suitable for:
- Planet Labs satellite tasking API
- Maxar Discovery API (STAC-based)
- Generic satellite imagery APIs
- YOLOv8 object detection
- DBSCAN/HDBSCAN clustering
- Siamese network training
- Segmentation masks
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon, mapping, shape

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Export configuration."""
    coordinate_precision: int = 6
    include_timestamps: bool = True
    yolo_normalize: bool = True  # Normalize YOLO coordinates to 0-1


class APIExporter:
    """Export GeoJSONs for satellite imagery APIs."""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    @staticmethod
    def for_planet_tasking(
        features: Dict,
        output_path: Union[str, Path],
        priority: str = "standard",
        imagery_type: str = "PSScene",
        additional_params: Optional[Dict] = None,
    ) -> Path:
        """
        Export for Planet Labs tasking/ordering API.

        Creates GeoJSON with proper format for Planet API and metadata file.

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path (without extension)
            priority: Order priority (standard, high)
            imagery_type: Item type (PSScene, SkySatCollect, etc.)
            additional_params: Extra parameters for order

        Returns:
            Path to exported GeoJSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract geometries and create AOI
        feature_list = features.get("features", [])

        if not feature_list:
            logger.warning("No features to export for Planet")
            return output_path.with_suffix(".geojson")

        # Create unified AOI from all feature geometries
        from shapely.ops import unary_union

        geoms = [shape(f["geometry"]) for f in feature_list if f.get("geometry")]
        unified = unary_union(geoms)

        # Buffer slightly to ensure coverage
        buffered = unified.buffer(0.001)  # ~100m buffer

        # Create Planet-compatible GeoJSON
        planet_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(buffered),
                "properties": {
                    "name": "water_infrastructure_aoi",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "detection_count": len(feature_list),
                }
            }]
        }

        # Save GeoJSON
        geojson_path = output_path.with_suffix(".geojson")
        with open(geojson_path, "w") as f:
            json.dump(planet_geojson, f, indent=2)

        # Save order parameters
        order_params = {
            "item_type": imagery_type,
            "priority": priority,
            "aoi_file": str(geojson_path),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "detection_count": len(feature_list),
            **(additional_params or {})
        }

        params_path = output_path.with_suffix(".params.json")
        with open(params_path, "w") as f:
            json.dump(order_params, f, indent=2)

        logger.info(f"Exported Planet tasking GeoJSON to {geojson_path}")
        return geojson_path

    @staticmethod
    def for_maxar(
        features: Dict,
        output_path: Union[str, Path],
        collections: Optional[List[str]] = None,
        datetime_range: Optional[str] = None,
        cloud_cover_max: Optional[int] = None,
        additional_filters: Optional[Dict] = None,
    ) -> Path:
        """
        Export for Maxar Discovery API (STAC-based).

        Creates a STAC-compatible search request body with 'intersects' geometry
        for use with Maxar's Discovery API.

        API Docs: https://developers.maxar.com/docs/discovery/

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path
            collections: Maxar collections to search (e.g., ["wv02", "wv03"])
                        Default: ["wv01", "wv02", "wv03"]
            datetime_range: ISO 8601 datetime range (e.g., "2024-01-01/2024-01-31")
            cloud_cover_max: Maximum cloud cover percentage (0-100)
            additional_filters: Additional STAC filter parameters

        Returns:
            Path to exported search request JSON
        """
        from shapely.ops import unary_union

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        feature_list = features.get("features", [])

        if not feature_list:
            logger.warning("No features to export for Maxar")
            return output_path.with_suffix(".json")

        # Create unified AOI geometry from all features
        geoms = [shape(f["geometry"]) for f in feature_list if f.get("geometry")]
        unified = unary_union(geoms)

        # Buffer slightly to ensure coverage (~100m)
        buffered = unified.buffer(0.001)

        # Get the geometry in GeoJSON format (WGS84)
        aoi_geometry = mapping(buffered)

        # Simplify if geometry is complex (Maxar has query timeout limits)
        if buffered.geom_type == "MultiPolygon" or len(str(aoi_geometry)) > 5000:
            simplified = buffered.simplify(0.0001, preserve_topology=True)
            aoi_geometry = mapping(simplified)

        # Default collections (WorldView satellites)
        if collections is None:
            collections = ["wv01", "wv02", "wv03"]

        # Build STAC search request body
        # Format: https://developers.maxar.com/docs/discovery/guides/discovery-guide
        search_request = {
            "intersects": aoi_geometry,
            "collections": collections,
        }

        # Add datetime filter if provided
        if datetime_range:
            search_request["datetime"] = datetime_range

        # Build filter for cloud cover and other properties
        filters = []
        if cloud_cover_max is not None:
            filters.append({
                "op": "<",
                "args": [{"property": "eo:cloud_cover"}, cloud_cover_max]
            })

        if additional_filters:
            for prop, value in additional_filters.items():
                if isinstance(value, dict) and "op" in value:
                    filters.append(value)
                else:
                    filters.append({
                        "op": "=",
                        "args": [{"property": prop}, value]
                    })

        if filters:
            if len(filters) == 1:
                search_request["filter"] = filters[0]
            else:
                search_request["filter"] = {
                    "op": "and",
                    "args": filters
                }

        # Save search request
        request_path = output_path.with_suffix(".search.json")
        with open(request_path, "w") as f:
            json.dump(search_request, f, indent=2)

        # Also save bbox for GET request alternative
        bounds = buffered.bounds  # (minx, miny, maxx, maxy)
        bbox_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

        # Save metadata with API usage info
        metadata = {
            "api": "maxar_discovery",
            "api_base_url": "https://api.maxar.com/discovery/v1",
            "search_endpoint": "https://api.maxar.com/discovery/v1/search",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "detection_count": len(feature_list),
            "collections": collections,
            "bbox": bbox_str,
            "datetime": datetime_range,
            "cloud_cover_max": cloud_cover_max,
            "usage": {
                "post_request": f"POST to search_endpoint with body from {request_path.name}",
                "get_request": f"GET search_endpoint?collections={','.join(collections)}&bbox={bbox_str}"
                               + (f"&datetime={datetime_range}" if datetime_range else ""),
            },
            "notes": [
                "Coordinates are in WGS84 (EPSG:4326)",
                "bbox format: west,south,east,north",
                "datetime format: start-date/end-date (ISO 8601)",
                "Query timeout is 30 seconds",
            ]
        }

        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save the AOI geometry separately for reference
        aoi_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": aoi_geometry,
                "properties": {
                    "name": "water_infrastructure_aoi",
                    "detection_count": len(feature_list),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
            }]
        }

        geojson_path = output_path.with_suffix(".geojson")
        with open(geojson_path, "w") as f:
            json.dump(aoi_geojson, f, indent=2)

        logger.info(f"Exported Maxar Discovery search request to {request_path}")
        return request_path

    @staticmethod
    def for_vantor(
        features: Dict,
        output_path: Union[str, Path],
        parameters: Optional[Dict] = None,
    ) -> Path:
        """
        Export for Vantor API format (alias for Maxar).

        Vantor uses Maxar's platform, so this delegates to for_maxar().

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path
            parameters: Additional parameters (passed as additional_filters)

        Returns:
            Path to exported file
        """
        return APIExporter.for_maxar(
            features,
            output_path,
            additional_filters=parameters,
        )

    @staticmethod
    def for_generic_satellite(
        features: Dict,
        output_path: Union[str, Path],
        api_name: str,
        custom_properties: Optional[Dict] = None,
    ) -> Path:
        """
        Export for generic satellite API with custom formatting.

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path
            api_name: Name of the satellite API
            custom_properties: Custom properties to add

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "type": "FeatureCollection",
            "metadata": {
                "api": api_name,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "feature_count": len(features.get("features", [])),
                **(custom_properties or {})
            },
            "features": features.get("features", [])
        }

        geojson_path = output_path.with_suffix(".geojson")
        with open(geojson_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {api_name} GeoJSON to {geojson_path}")
        return geojson_path


class MLExporter:
    """Export for ML model consumption."""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def for_yolov8(
        self,
        features: Dict,
        output_dir: Union[str, Path],
        image_bounds: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        class_map: Dict[str, int],
        include_geojson: bool = True,
    ) -> Path:
        """
        Export YOLO format annotations.

        Creates .txt files with YOLO format bounding boxes:
        <class_id> <x_center> <y_center> <width> <height>

        All values normalized to 0-1 if yolo_normalize is True.

        Args:
            features: GeoJSON FeatureCollection
            output_dir: Output directory
            image_bounds: Image bounds (west, south, east, north)
            image_size: Image size (width, height) in pixels
            class_map: Map of classification/type to class ID
            include_geojson: Also save GeoJSON for reference

        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        west, south, east, north = image_bounds
        img_width, img_height = image_size

        # Calculate pixel scaling
        lon_scale = img_width / (east - west)
        lat_scale = img_height / (north - south)

        annotations = []
        feature_list = features.get("features", [])

        for feature in feature_list:
            if not feature.get("geometry"):
                continue

            props = feature.get("properties", {})
            geom = shape(feature["geometry"])

            # Determine class from classification or infrastructure_type
            class_name = (
                props.get("classification") or
                props.get("infrastructure_type") or
                props.get("detection_type") or
                "unknown"
            )

            class_id = class_map.get(class_name, class_map.get("unknown", 0))

            # Get bounding box
            bounds = geom.bounds  # (minx, miny, maxx, maxy)

            # Convert to pixel coordinates
            x_min = (bounds[0] - west) * lon_scale
            y_min = (north - bounds[3]) * lat_scale  # Flip Y
            x_max = (bounds[2] - west) * lon_scale
            y_max = (north - bounds[1]) * lat_scale

            # Calculate center and dimensions
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Normalize if configured
            if self.config.yolo_normalize:
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height

            # Ensure minimum bounding box size for point features
            min_size = 0.01 if self.config.yolo_normalize else 10
            width = max(width, min_size)
            height = max(height, min_size)

            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save annotations
        labels_path = output_dir / "labels.txt"
        with open(labels_path, "w") as f:
            f.write("\n".join(annotations))

        # Save class map
        classes_path = output_dir / "classes.txt"
        with open(classes_path, "w") as f:
            sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
            f.write("\n".join(name for name, _ in sorted_classes))

        # Save data.yaml for YOLOv8
        data_yaml = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {v: k for k, v in class_map.items()}
        }

        import yaml
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Save GeoJSON for reference
        if include_geojson:
            geojson_path = output_dir / "features.geojson"
            with open(geojson_path, "w") as f:
                json.dump(features, f, indent=2)

        # Save metadata
        metadata = {
            "image_bounds": image_bounds,
            "image_size": image_size,
            "class_map": class_map,
            "annotation_count": len(annotations),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported {len(annotations)} YOLO annotations to {output_dir}")
        return output_dir

    @staticmethod
    def for_dbscan(
        features: Dict,
        output_path: Union[str, Path],
        format: str = "npy",
        include_properties: bool = True,
    ) -> Path:
        """
        Export point coordinates for DBSCAN/HDBSCAN clustering.

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path
            format: Output format ("npy", "csv", "json")
            include_properties: Include feature properties in output

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        points = []
        properties_list = []

        for feature in features.get("features", []):
            if not feature.get("geometry"):
                continue

            geom = shape(feature["geometry"])
            centroid = geom.centroid

            points.append([centroid.x, centroid.y])

            if include_properties:
                properties_list.append(feature.get("properties", {}))

        if not points:
            logger.warning("No points to export for clustering")
            points = [[0, 0]]  # Placeholder

        coords = np.array(points)

        if format == "npy":
            file_path = output_path.with_suffix(".npy")
            np.save(file_path, coords)

            if include_properties:
                props_path = output_path.with_suffix(".properties.json")
                with open(props_path, "w") as f:
                    json.dump(properties_list, f, indent=2)

        elif format == "csv":
            file_path = output_path.with_suffix(".csv")
            header = "longitude,latitude"

            if include_properties and properties_list:
                # Add property columns
                prop_keys = list(properties_list[0].keys())
                header += "," + ",".join(prop_keys)

            with open(file_path, "w") as f:
                f.write(header + "\n")
                for i, point in enumerate(points):
                    row = f"{point[0]},{point[1]}"
                    if include_properties and i < len(properties_list):
                        for key in prop_keys:
                            val = properties_list[i].get(key, "")
                            # Escape commas in string values
                            if isinstance(val, str) and "," in val:
                                val = f'"{val}"'
                            row += f",{val}"
                    f.write(row + "\n")

        else:  # json
            file_path = output_path.with_suffix(".json")
            data = {
                "coordinates": points,
                "properties": properties_list if include_properties else None,
                "count": len(points),
            }
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

        logger.info(f"Exported {len(points)} points for clustering to {file_path}")
        return file_path

    @staticmethod
    def for_siamese(
        features: Dict,
        output_path: Union[str, Path],
        class_key: str = "classification",
        pair_same_class: bool = True,
    ) -> Path:
        """
        Export feature pairs for Siamese network training.

        Creates pairs of features with same/different labels for
        similarity learning.

        Args:
            features: GeoJSON FeatureCollection
            output_path: Output file path
            class_key: Property key for classification
            pair_same_class: Create same-class pairs (for similarity learning)

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        feature_list = features.get("features", [])

        if len(feature_list) < 2:
            logger.warning("Need at least 2 features for Siamese pairs")
            return output_path.with_suffix(".json")

        # Group features by class
        class_groups = {}
        for i, f in enumerate(feature_list):
            class_name = f.get("properties", {}).get(class_key, "unknown")
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(i)

        pairs = []

        # Create same-class pairs (positive examples)
        for class_name, indices in class_groups.items():
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i + 1, min(i + 5, len(indices))):  # Limit pairs
                        pairs.append({
                            "idx1": indices[i],
                            "idx2": indices[j],
                            "same_class": True,
                            "class": class_name,
                        })

        # Create different-class pairs (negative examples)
        class_names = list(class_groups.keys())
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                indices1 = class_groups[class_names[i]]
                indices2 = class_groups[class_names[j]]

                # Create limited pairs between classes
                for idx1 in indices1[:3]:
                    for idx2 in indices2[:3]:
                        pairs.append({
                            "idx1": idx1,
                            "idx2": idx2,
                            "same_class": False,
                            "class1": class_names[i],
                            "class2": class_names[j],
                        })

        # Save pairs
        output_data = {
            "pairs": pairs,
            "features": feature_list,
            "class_groups": {k: v for k, v in class_groups.items()},
            "total_pairs": len(pairs),
            "positive_pairs": sum(1 for p in pairs if p["same_class"]),
            "negative_pairs": sum(1 for p in pairs if not p["same_class"]),
        }

        file_path = output_path.with_suffix(".json")
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Exported {len(pairs)} Siamese pairs to {file_path}")
        return file_path

    @staticmethod
    def for_segmentation(
        features: Dict,
        bounds: Tuple[float, float, float, float],
        size: Tuple[int, int],
        class_map: Dict[str, int],
        output_path: Union[str, Path],
        background_value: int = 0,
    ) -> Path:
        """
        Generate segmentation mask from features.

        Args:
            features: GeoJSON FeatureCollection
            bounds: Geographic bounds (west, south, east, north)
            size: Mask size (width, height) in pixels
            class_map: Map of class names to mask values
            output_path: Output file path
            background_value: Value for background pixels

        Returns:
            Path to exported mask
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        west, south, east, north = bounds
        width, height = size

        # Create empty mask
        mask = np.full((height, width), background_value, dtype=np.uint8)

        # Calculate scaling
        lon_scale = width / (east - west)
        lat_scale = height / (north - south)

        for feature in features.get("features", []):
            if not feature.get("geometry"):
                continue

            props = feature.get("properties", {})
            geom = shape(feature["geometry"])

            # Determine class
            class_name = (
                props.get("classification") or
                props.get("infrastructure_type") or
                "unknown"
            )
            class_value = class_map.get(class_name, background_value)

            if class_value == background_value:
                continue

            # Rasterize geometry
            if geom.geom_type == "Point":
                x = int((geom.x - west) * lon_scale)
                y = int((north - geom.y) * lat_scale)

                # Draw small circle for points
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dx*dx + dy*dy <= 9:  # Circle radius 3
                            px, py = x + dx, y + dy
                            if 0 <= px < width and 0 <= py < height:
                                mask[py, px] = class_value

            elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                # Simple polygon rasterization
                if geom.geom_type == "MultiPolygon":
                    polygons = list(geom.geoms)
                else:
                    polygons = [geom]

                for poly in polygons:
                    minx, miny, maxx, maxy = poly.bounds

                    # Convert to pixel bounds
                    px_min = max(0, int((minx - west) * lon_scale))
                    px_max = min(width - 1, int((maxx - west) * lon_scale))
                    py_min = max(0, int((north - maxy) * lat_scale))
                    py_max = min(height - 1, int((north - miny) * lat_scale))

                    for py in range(py_min, py_max + 1):
                        for px in range(px_min, px_max + 1):
                            # Convert pixel to geo coordinate
                            lon = west + (px + 0.5) / lon_scale
                            lat = north - (py + 0.5) / lat_scale

                            if poly.contains(Point(lon, lat)):
                                mask[py, px] = class_value

        # Save mask
        mask_path = output_path.with_suffix(".npy")
        np.save(mask_path, mask)

        # Save PNG visualization
        try:
            from PIL import Image

            # Create colored visualization
            colors = [
                [0, 0, 0],  # Background
                [255, 0, 0],  # Class 1 (red)
                [0, 255, 0],  # Class 2 (green)
                [0, 0, 255],  # Class 3 (blue)
                [255, 255, 0],  # Class 4 (yellow)
                [255, 0, 255],  # Class 5 (magenta)
            ]

            vis = np.zeros((height, width, 3), dtype=np.uint8)
            for class_val, color in enumerate(colors):
                vis[mask == class_val] = color

            img = Image.fromarray(vis)
            png_path = output_path.with_suffix(".png")
            img.save(png_path)
            logger.info(f"Saved mask visualization to {png_path}")

        except ImportError:
            logger.debug("PIL not available for mask visualization")

        # Save metadata
        metadata = {
            "bounds": bounds,
            "size": size,
            "class_map": class_map,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported segmentation mask to {mask_path}")
        return mask_path


def export_for_satellite_api(
    features: Dict,
    output_dir: Union[str, Path],
    api: str = "planet",
    **kwargs,
) -> Dict[str, Path]:
    """
    High-level export for satellite APIs.

    Args:
        features: GeoJSON FeatureCollection
        output_dir: Output directory
        api: API name ("planet", "maxar", "vantor", or custom)
        **kwargs: Additional parameters for specific API
            - planet: priority, imagery_type, additional_params
            - maxar: collections, datetime_range, cloud_cover_max, additional_filters

    Returns:
        Dict mapping output type to file path
    """
    output_dir = Path(output_dir)
    exporter = APIExporter()

    outputs = {}

    if api == "planet":
        path = exporter.for_planet_tasking(
            features,
            output_dir / "planet_tasking",
            **kwargs
        )
        outputs["planet_geojson"] = path
        outputs["planet_params"] = path.with_suffix(".params.json")

    elif api == "maxar":
        path = exporter.for_maxar(
            features,
            output_dir / "maxar_discovery",
            collections=kwargs.get("collections"),
            datetime_range=kwargs.get("datetime_range"),
            cloud_cover_max=kwargs.get("cloud_cover_max"),
            additional_filters=kwargs.get("additional_filters"),
        )
        outputs["maxar_search"] = path
        outputs["maxar_geojson"] = path.with_suffix(".geojson")
        outputs["maxar_meta"] = path.with_suffix(".meta.json")

    elif api == "vantor":
        # Vantor uses Maxar platform
        path = exporter.for_vantor(
            features,
            output_dir / "vantor_export",
            parameters=kwargs.get("parameters")
        )
        outputs["vantor_search"] = path
        outputs["vantor_geojson"] = path.with_suffix(".geojson")

    else:
        path = exporter.for_generic_satellite(
            features,
            output_dir / f"{api}_export",
            api_name=api,
            custom_properties=kwargs.get("custom_properties")
        )
        outputs[f"{api}_geojson"] = path

    return outputs


def export_for_ml(
    features: Dict,
    output_dir: Union[str, Path],
    formats: List[str] = None,
    image_bounds: Optional[Tuple[float, float, float, float]] = None,
    image_size: Tuple[int, int] = (1024, 1024),
    class_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Path]:
    """
    High-level export for ML models.

    Args:
        features: GeoJSON FeatureCollection
        output_dir: Output directory
        formats: List of formats ("yolo", "clustering", "siamese", "segmentation")
        image_bounds: Geographic bounds for pixel conversion
        image_size: Image size in pixels
        class_map: Classification to class ID mapping

    Returns:
        Dict mapping format to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = formats or ["yolo", "clustering"]
    exporter = MLExporter()

    # Default class map
    if class_map is None:
        class_map = {
            "confirmed_incident": 0,
            "potential_incident": 1,
            "known_event": 2,
            "unclassified": 3,
            "unknown": 4,
        }

    # Get bounds from features if not provided
    if image_bounds is None:
        from shapely.ops import unary_union

        geoms = [shape(f["geometry"]) for f in features.get("features", [])
                if f.get("geometry")]
        if geoms:
            unified = unary_union(geoms)
            image_bounds = unified.buffer(0.01).bounds  # Add small buffer

    outputs = {}

    for fmt in formats:
        if fmt == "yolo":
            if image_bounds:
                path = exporter.for_yolov8(
                    features,
                    output_dir / "yolo",
                    image_bounds,
                    image_size,
                    class_map,
                )
                outputs["yolo"] = path

        elif fmt == "clustering":
            path = exporter.for_dbscan(
                features,
                output_dir / "clustering",
                format="npy",
            )
            outputs["clustering"] = path

        elif fmt == "siamese":
            path = exporter.for_siamese(
                features,
                output_dir / "siamese",
            )
            outputs["siamese"] = path

        elif fmt == "segmentation":
            if image_bounds:
                path = exporter.for_segmentation(
                    features,
                    image_bounds,
                    image_size,
                    class_map,
                    output_dir / "segmentation_mask",
                )
                outputs["segmentation"] = path

    # Always save raw GeoJSON
    geojson_path = output_dir / "features.geojson"
    with open(geojson_path, "w") as f:
        json.dump(features, f, indent=2)
    outputs["geojson"] = geojson_path

    return outputs
