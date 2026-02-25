"""
xView / YOLOv8 change-detection module for Planet satellite tile analysis.

Primary model: xView (DigitalGlobe WorldView, ~0.3 m native GSD)
--------------------------------------------------------------------
Planet PSScene imagery is at ~3.0 m/px.  Because xView was trained on
imagery 10× finer, objects in Planet tiles appear 10× smaller in pixel
space than the model expects.

GSD correction
--------------
Each tile is presented to the model at an *effective* inference resolution:

    scale_factor   = inference_gsd_m / training_gsd_m = 3.0 / 0.3 = 10
    effective_imgsz = tile_size × scale_factor         = 640 × 10  = 6400

Ultralytics resizes the input to ``effective_imgsz`` before running the
backbone, so objects appear at the correct pixel scale.

xView classes of interest
--------------------------
NOTE: IDs are 0-indexed YOLO class numbers, remapped from the original
sparse xView type_ids (11–94) sorted in ascending order.

    YOLO ID   xView type_id   Label
    -------   -------------   -----------------------------------------------
     15           29          Truck w/Liquid
     21           37          Tank Car
     33           53          Engineering Vehicle
     40           61          Haul Truck
     50           76          Damaged Building
     55           86          Storage Tank
     56           89          Shipping Container Lot
     57           91          Shipping Container
     59           94          Tower

Outputs
-------
For each tile set (before / after):
    - Annotated PNG: bounding boxes with class label + confidence drawn
      on the original tile image using Pillow
    - Detection CSV: one row per detection with geo-coordinates,
      class, confidence, change status, and nearest OSM feature info

Workflow
--------
    1. ``detect_tiles()``        — run GSD-corrected inference on one date's tiles
    2. ``compare_tile_sets()``   — classify changes: new / persistent / disappeared
    3. ``annotate_tiles()``      — draw boxes on tile PNGs, save annotated copies
    4. ``save_detection_csv()``  — write per-date CSV tables
    5. ``osm_proximity()``       — cross-reference detections with OSM GeoJSON
    6. ``detect_and_annotate()`` — end-to-end convenience wrapper
"""

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# xView class configuration
# ---------------------------------------------------------------------------

XVIEW_CLASSES_OF_INTEREST: List[int] = [15, 21, 33, 40, 50, 55, 56, 57, 59]

XVIEW_CLASS_NAMES: Dict[int, str] = {
    15: "truck_w_liquid",
    21: "tank_car",
    33: "engineering_vehicle",
    40: "haul_truck",
    50: "damaged_building",
    55: "storage_tank",
    56: "shipping_container_lot",
    57: "shipping_container",
    59: "tower",
}

# Colour palette per class (RGB) for annotation boxes
XVIEW_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    15: (0,   150, 255), # blue      — truck w/liquid (water)
    21: (100, 100, 200), # steel     — tank car
    33: (255, 165, 0),   # orange    — engineering vehicle
    40: (255, 210, 0),   # yellow    — haul truck
    50: (255, 50,  50),  # red       — damaged building
    55: (50,  50,  255), # blue      — storage tank
    56: (150, 50,  200), # purple    — shipping container lot
    57: (50,  200, 200), # cyan      — shipping container
    59: (180, 180, 180), # grey      — tower
}

XVIEW_TRAINING_GSD_M: float = 0.3
PLANET_INFERENCE_GSD_M: float = 3.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class YOLOv8Config:
    """
    Configuration for xView-based YOLOv8 inference and change detection.

    The default GSD correction ratio is 10× (3.0 m ÷ 0.3 m), giving an
    effective inference image size of 6400 px for 640-px tiles.

    Attributes:
        model_path:                   Path to xView ``.pt`` weights.
        training_gsd_m:               Training imagery GSD (xView = 0.3 m).
        inference_gsd_m:              Inference tile GSD (Planet = 3.0 m).
        confidence_threshold:         Minimum detection confidence.
        iou_threshold:                NMS IoU threshold.
        device:                       Inference device ("auto", "cpu", "cuda", "mps").
        imgsz:                        Base tile size in pixels (640).
        half:                         Use fp16 on GPU.
        classes_of_interest:          xView class IDs to keep.
        class_names:                  Class ID → label mapping.
        change_overlap_iou:           Minimum IoU to cluster detections across dates.
        min_detections_for_persistent: Minimum dates for a detection to be "persistent".
        osm_proximity_radius_m:       Radius (m) for OSM feature cross-reference.
        save_annotated_tiles:         Write annotated PNGs alongside originals.
        annotation_box_width:         Bounding-box border width (pixels).
    """
    model_path: str = "xview.pt"
    training_gsd_m: float = XVIEW_TRAINING_GSD_M
    inference_gsd_m: float = PLANET_INFERENCE_GSD_M
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    imgsz: int = 640
    half: bool = False
    classes_of_interest: Optional[List[int]] = field(
        default_factory=lambda: list(XVIEW_CLASSES_OF_INTEREST)
    )
    class_names: Optional[Dict[int, str]] = field(
        default_factory=lambda: dict(XVIEW_CLASS_NAMES)
    )
    change_overlap_iou: float = 0.30
    min_detections_for_persistent: int = 2
    osm_proximity_radius_m: float = 500.0
    save_annotated_tiles: bool = True
    annotation_box_width: int = 3

    @property
    def scale_factor(self) -> float:
        """GSD correction ratio: inference_gsd_m / training_gsd_m (default 10×)."""
        return self.inference_gsd_m / self.training_gsd_m

    @property
    def effective_imgsz(self) -> int:
        """
        Actual inference size passed to ultralytics after GSD correction.
        Rounded to nearest multiple of 32 (required by YOLOv8 backbone stride).
        """
        raw = round(self.imgsz * self.scale_factor)
        return (raw // 32) * 32


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """
    xView YOLOv8 detector with GSD correction, change analysis,
    annotated image output, and CSV reporting.

    Example::

        cfg = YOLOv8Config(model_path="xview_best.pt", device="cuda")
        det = YOLOv8Detector(cfg)
        results = detect_and_annotate(
            tile_metadata_path="output/tiles_metadata.json",
            osm_geojson=infra_geojson,
            output_dir=Path("output/yolo"),
            config=cfg,
        )
    """

    def __init__(self, config: Optional[YOLOv8Config] = None):
        self.config = config or YOLOv8Config()
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics required: pip install ultralytics") from exc

        logger.info(
            f"Loading xView model: {self.config.model_path}  |  "
            f"GSD {self.config.training_gsd_m} m → {self.config.inference_gsd_m} m  "
            f"(scale={self.config.scale_factor:.1f}×, "
            f"effective_imgsz={self.config.effective_imgsz} px)"
        )
        self.model = YOLO(self.config.model_path)

        if hasattr(self.model, "names") and self.config.class_names is None:
            self.config.class_names = self.model.names
        elif hasattr(self.model, "names"):
            for cls_id, label in self.model.names.items():
                if cls_id not in self.config.class_names:
                    self.config.class_names[cls_id] = label

    # ── geo helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _box_to_geo(norm_box: List[float], geo_bounds: Dict) -> List[float]:
        """
        Map a normalised [x1,y1,x2,y2] box to WGS-84 [W,S,E,N].

        norm_box values are in [0, 1] relative to the effective_imgsz space.
        """
        west, east = geo_bounds["west"], geo_bounds["east"]
        south, north = geo_bounds["south"], geo_bounds["north"]
        lon_span = east - west
        lat_span = north - south

        x1, y1, x2, y2 = norm_box
        geo_west  = west  + x1 * lon_span
        geo_east  = west  + x2 * lon_span
        geo_north = north - y1 * lat_span
        geo_south = north - y2 * lat_span

        return [
            min(geo_west, geo_east),
            min(geo_north, geo_south),
            max(geo_west, geo_east),
            max(geo_north, geo_south),
        ]

    # ── detection ────────────────────────────────────────────────────────────

    def detect_tiles(self, tile_metadata: List[Dict]) -> List[Dict]:
        """
        Run GSD-corrected xView inference on all tiles in *tile_metadata*.

        Args:
            tile_metadata: Tile metadata list from :mod:`imagery`.

        Returns:
            List of detection dicts::

                {tile_path, image_date, label, geo_bounds, class_id,
                 class_name, confidence, box_pixels, box_geo,
                 scale_factor, effective_imgsz}
        """
        self._load_model()
        eff = self.config.effective_imgsz
        sf  = self.config.scale_factor
        all_det = []

        for meta in tile_metadata:
            tile_path  = meta["path"]
            geo_bounds = meta["geo_bounds"]
            img_date   = meta.get("image_date", "unknown")
            label      = meta.get("label", "scene")

            try:
                results = self.model.predict(
                    source=tile_path,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=eff,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning(f"Inference failed on {tile_path}: {exc}")
                continue

            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if (
                        self.config.classes_of_interest is not None
                        and cls_id not in self.config.classes_of_interest
                    ):
                        continue

                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    norm_box = [c / eff for c in xyxy]
                    box_geo  = self._box_to_geo(norm_box, geo_bounds)
                    names    = self.config.class_names or {}

                    all_det.append({
                        "tile_path":       tile_path,
                        "image_date":      img_date,
                        "label":           label,
                        "geo_bounds":      geo_bounds,
                        "class_id":        cls_id,
                        "class_name":      names.get(cls_id, str(cls_id)),
                        "confidence":      round(conf, 4),
                        "box_pixels":      xyxy,
                        "box_geo":         box_geo,
                        "scale_factor":    sf,
                        "effective_imgsz": eff,
                    })

        logger.info(
            f"Detected {len(all_det)} xView objects across {len(tile_metadata)} tiles"
        )
        return all_det

    # ── change detection ─────────────────────────────────────────────────────

    @staticmethod
    def _box_iou(a: List[float], b: List[float]) -> float:
        """IoU between two [W, S, E, N] geographic boxes."""
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def compare_tile_sets(self, detections_by_date: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Classify change status across weekly or paired image sets.

        Clusters spatially overlapping same-class detections and assigns:
            - ``"new"``          — appears only in the *latest* image
            - ``"persistent"``   — appears in ≥ ``min_detections_for_persistent`` images
            - ``"disappeared"``  — present in earlier images, absent in latest

        Args:
            detections_by_date: Mapping of date string → detection list.

        Returns:
            List of change-comparison result dicts with ``change_status`` field.
        """
        if not detections_by_date:
            return []

        sorted_dates = sorted(detections_by_date.keys())
        latest_date  = sorted_dates[-1]

        flat_dets = [
            {**d, "image_date": date}
            for date, dets in detections_by_date.items()
            for d in dets
        ]

        used: set = set()
        clusters: List[List[Dict]] = []

        for i, di in enumerate(flat_dets):
            if i in used:
                continue
            cluster = [di]
            used.add(i)
            for j, dj in enumerate(flat_dets):
                if j in used or j == i:
                    continue
                if di["class_id"] != dj["class_id"]:
                    continue
                if self._box_iou(di["box_geo"], dj["box_geo"]) >= self.config.change_overlap_iou:
                    cluster.append(dj)
                    used.add(j)
            clusters.append(cluster)

        results = []
        for cluster in clusters:
            dates = sorted({d["image_date"] for d in cluster})
            n     = len(dates)
            best  = max(cluster, key=lambda x: x["confidence"])
            in_latest = latest_date in dates

            if n >= self.config.min_detections_for_persistent:
                status = "persistent"
            elif in_latest and n == 1:
                status = "new"
            elif not in_latest:
                status = "disappeared"
            else:
                status = "new"

            results.append({
                "class_id":            best["class_id"],
                "class_name":          best["class_name"],
                "confidence":          best["confidence"],
                "box_geo":             best["box_geo"],
                "tile_path":           best["tile_path"],
                "label":               best.get("label", "scene"),
                "dates_detected":      dates,
                "change_status":       status,
                "change_confidence":   round(n / len(sorted_dates), 3),
                "representative_date": best["image_date"],
                # OSM proximity will be filled in by osm_proximity()
                "near_osm_feature":    False,
                "nearest_osm_type":    None,
                "nearest_osm_dist_m":  None,
            })

        n_new  = sum(1 for r in results if r["change_status"] == "new")
        n_per  = sum(1 for r in results if r["change_status"] == "persistent")
        n_dis  = sum(1 for r in results if r["change_status"] == "disappeared")
        logger.info(
            f"Change analysis: {len(results)} clusters — "
            f"new={n_new}, persistent={n_per}, disappeared={n_dis}"
        )
        return results

    # ── OSM proximity ────────────────────────────────────────────────────────

    def osm_proximity(
        self,
        results: List[Dict],
        osm_geojson: Dict,
    ) -> List[Dict]:
        """
        Cross-reference detections with OSM infrastructure features.

        For each detection, computes the distance to the nearest OSM feature
        centroid and records whether it falls within ``osm_proximity_radius_m``.

        Args:
            results:     Change-detection results from :meth:`compare_tile_sets`.
            osm_geojson: GeoJSON FeatureCollection of OSM features.

        Returns:
            Updated *results* list with ``near_osm_feature``,
            ``nearest_osm_type``, and ``nearest_osm_dist_m`` populated.
        """
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union

        features = osm_geojson.get("features", [])
        if not features or not results:
            return results

        # Build list of (centroid, properties) for each OSM feature
        osm_pts = []
        for feat in features:
            geom = feat.get("geometry")
            props = feat.get("properties", {})
            if not geom:
                continue
            try:
                centroid = shape(geom).centroid
                osm_pts.append((centroid, props))
            except Exception:
                continue

        if not osm_pts:
            return results

        # Reference lat for degree→metre conversion
        avg_lat = sum(pt.y for pt, _ in osm_pts) / len(osm_pts)
        m_per_deg = 111320 * math.cos(math.radians(avg_lat))
        radius_deg = self.config.osm_proximity_radius_m / m_per_deg

        for res in results:
            west, south, east, north = res["box_geo"]
            det_centroid = Point((west + east) / 2, (south + north) / 2)

            min_dist_deg = float("inf")
            nearest_type = None

            for osm_pt, osm_props in osm_pts:
                d = det_centroid.distance(osm_pt)
                if d < min_dist_deg:
                    min_dist_deg = d
                    nearest_type = (
                        osm_props.get("infra_category")
                        or osm_props.get("water")
                        or osm_props.get("waterway")
                        or osm_props.get("man_made")
                        or "unknown"
                    )

            dist_m = min_dist_deg * m_per_deg
            res["near_osm_feature"]   = dist_m <= self.config.osm_proximity_radius_m
            res["nearest_osm_type"]   = nearest_type
            res["nearest_osm_dist_m"] = round(dist_m, 1)

        n_near = sum(1 for r in results if r["near_osm_feature"])
        logger.info(
            f"OSM proximity: {n_near}/{len(results)} detections within "
            f"{self.config.osm_proximity_radius_m:.0f} m of infrastructure"
        )
        return results

    # ── annotated image export ───────────────────────────────────────────────

    def annotate_tiles(
        self,
        detections: List[Dict],
        output_dir: Path,
    ) -> List[Path]:
        """
        Draw detection bounding boxes on tile PNGs and save annotated copies.

        Each annotated file is saved as ``<original_stem>_annotated.png``
        in *output_dir*.  Boxes are colour-coded by class (see
        :data:`XVIEW_CLASS_COLORS`).

        Args:
            detections: Raw detection dicts from :meth:`detect_tiles`.
            output_dir: Directory for annotated PNGs.

        Returns:
            List of Paths to annotated PNG files (one per unique tile).
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow not installed — skipping tile annotation")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group detections by tile_path
        by_tile: Dict[str, List[Dict]] = {}
        for det in detections:
            by_tile.setdefault(det["tile_path"], []).append(det)

        annotated_paths = []

        for tile_path_str, tile_dets in by_tile.items():
            tile_path = Path(tile_path_str)
            if not tile_path.exists():
                continue

            img = Image.open(tile_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            for det in tile_dets:
                cls_id = det["class_id"]
                conf   = det["confidence"]
                xyxy   = det["box_pixels"]
                color  = XVIEW_CLASS_COLORS.get(cls_id, (255, 255, 0))
                name   = (det.get("class_name") or str(cls_id)).replace("_", " ")
                label  = f"{name} {conf:.2f}"

                lw = self.config.annotation_box_width
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                # Scale pixel coords from effective_imgsz back to tile_size
                eff = det.get("effective_imgsz", self.config.effective_imgsz)
                sx = w / eff
                sy = h / eff
                x1b, y1b = int(x1 * sx), int(y1 * sy)
                x2b, y2b = int(x2 * sx), int(y2 * sy)

                draw.rectangle([x1b, y1b, x2b, y2b], outline=color, width=lw)

                # Label background
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                tw = int(len(label) * 6)
                th = 12
                draw.rectangle([x1b, max(0, y1b - th - 2), x1b + tw + 4, y1b], fill=color)
                draw.text((x1b + 2, max(0, y1b - th)), label, fill=(0, 0, 0), font=font)

            out_path = output_dir / f"{tile_path.stem}_annotated.png"
            img.save(out_path)
            annotated_paths.append(out_path)

        logger.info(f"Annotated {len(annotated_paths)} tiles → {output_dir}")
        return annotated_paths

    # ── CSV export ───────────────────────────────────────────────────────────

    def save_detection_csv(
        self,
        results: List[Dict],
        output_path: Path,
    ) -> Path:
        """
        Write detection + change results to a CSV table.

        Columns include geo-coordinates, class, confidence, change status,
        OSM proximity, and date coverage.

        Args:
            results:     Output of :meth:`compare_tile_sets` (optionally
                         enriched with :meth:`osm_proximity`).
            output_path: Destination ``.csv`` path.

        Returns:
            The written :class:`~pathlib.Path`.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "class_id", "class_name", "confidence",
            "change_status", "change_confidence",
            "dates_detected", "representative_date",
            "box_geo_west", "box_geo_south", "box_geo_east", "box_geo_north",
            "near_osm_feature", "nearest_osm_type", "nearest_osm_dist_m",
            "tile_path",
        ]

        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                west, south, east, north = r.get("box_geo", [None, None, None, None])
                row = {
                    "class_id":           r.get("class_id"),
                    "class_name":         r.get("class_name"),
                    "confidence":         r.get("confidence"),
                    "change_status":      r.get("change_status"),
                    "change_confidence":  r.get("change_confidence"),
                    "dates_detected":     ";".join(r.get("dates_detected", [])),
                    "representative_date": r.get("representative_date"),
                    "box_geo_west":       west,
                    "box_geo_south":      south,
                    "box_geo_east":       east,
                    "box_geo_north":      north,
                    "near_osm_feature":   r.get("near_osm_feature"),
                    "nearest_osm_type":   r.get("nearest_osm_type"),
                    "nearest_osm_dist_m": r.get("nearest_osm_dist_m"),
                    "tile_path":          r.get("tile_path"),
                }
                writer.writerow(row)

        logger.info(f"Detection table saved → {output_path}  ({len(results)} rows)")
        return output_path

    # ── GeoJSON export ────────────────────────────────────────────────────────

    def to_geojson(self, results: List[Dict]) -> Dict:
        """Convert change-detection results to a GeoJSON FeatureCollection."""
        features = []
        for r in results:
            west, south, east, north = r["box_geo"]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south], [east, south],
                        [east, north], [west, north],
                        [west, south],
                    ]],
                },
                "properties": {
                    "class_id":            r["class_id"],
                    "class_name":          r["class_name"],
                    "confidence":          r["confidence"],
                    "change_status":       r["change_status"],
                    "change_confidence":   r["change_confidence"],
                    "dates_detected":      r["dates_detected"],
                    "representative_date": r["representative_date"],
                    "near_osm_feature":    r.get("near_osm_feature"),
                    "nearest_osm_type":    r.get("nearest_osm_type"),
                    "nearest_osm_dist_m":  r.get("nearest_osm_dist_m"),
                    "detection_type":      "xview_yolov8",
                    "training_gsd_m":      XVIEW_TRAINING_GSD_M,
                    "inference_gsd_m":     PLANET_INFERENCE_GSD_M,
                },
            })
        return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------

def detect_and_annotate(
    tile_metadata_path: Path,
    output_dir: Path,
    osm_geojson: Optional[Dict] = None,
    config: Optional[YOLOv8Config] = None,
) -> Dict:
    """
    Run xView inference, change detection, annotate tiles, and write CSV + GeoJSON.

    Loads tile metadata from ``tiles_metadata.json``, groups tiles by
    acquisition label (before / after), runs GSD-corrected inference, compares
    detections across the two date sets, cross-references with OSM features,
    annotates each tile PNG, and writes per-date CSV tables plus a combined
    GeoJSON.

    Args:
        tile_metadata_path: Path to ``tiles_metadata.json``.
        output_dir:         Root output directory for all yolov8 artefacts.
        osm_geojson:        Optional OSM infrastructure GeoJSON for proximity.
        config:             Optional :class:`YOLOv8Config`.

    Returns:
        Dict with keys:
            ``"detections_by_date"`` — raw per-date detection lists
            ``"comparison"``          — change-comparison result dicts
            ``"geojson"``             — GeoJSON FeatureCollection
            ``"geojson_path"``        — Path to saved GeoJSON
            ``"csv_path"``            — Path to saved CSV table
            ``"annotated_paths"``     — list of annotated tile PNG paths
    """
    tile_metadata_path = Path(tile_metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(tile_metadata_path) as fh:
        all_tiles = json.load(fh)

    if not all_tiles:
        logger.warning("No tile metadata found — nothing to detect")
        return {
            "detections_by_date": {},
            "comparison": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "geojson_path": None,
            "csv_path": None,
            "annotated_paths": [],
        }

    cfg      = config or YOLOv8Config()
    detector = YOLOv8Detector(cfg)

    logger.info(
        f"xView config: model={cfg.model_path}  "
        f"scale={cfg.scale_factor:.1f}×  "
        f"effective_imgsz={cfg.effective_imgsz}  "
        f"classes={cfg.classes_of_interest}"
    )

    # Group tiles by date (use image_date + label as key)
    tiles_by_date: Dict[str, List[Dict]] = {}
    for tile in all_tiles:
        key = f"{tile.get('image_date', 'unknown')}_{tile.get('label', 'scene')}"
        tiles_by_date.setdefault(key, []).append(tile)

    logger.info(
        f"Running xView on {len(all_tiles)} tiles "
        f"across {len(tiles_by_date)} date-label groups"
    )

    detections_by_date: Dict[str, List[Dict]] = {}
    all_raw_detections: List[Dict] = []

    for key in sorted(tiles_by_date):
        tiles = tiles_by_date[key]
        logger.info(f"  Detecting: {len(tiles)} tiles for [{key}]…")
        dets = detector.detect_tiles(tiles)
        detections_by_date[key] = dets
        all_raw_detections.extend(dets)

    comparison = detector.compare_tile_sets(detections_by_date)

    # OSM proximity cross-reference
    if osm_geojson and osm_geojson.get("features"):
        comparison = detector.osm_proximity(comparison, osm_geojson)

    # Annotated tiles
    annotated = []
    if cfg.save_annotated_tiles and all_raw_detections:
        ann_dir = output_dir / "annotated_tiles"
        annotated = detector.annotate_tiles(all_raw_detections, ann_dir)

    # GeoJSON
    geojson = detector.to_geojson(comparison)
    geojson_path = output_dir / "yolo_detections.geojson"
    with open(geojson_path, "w") as fh:
        json.dump(geojson, fh, indent=2)
    logger.info(f"xView GeoJSON saved → {geojson_path}")

    # CSV table
    csv_path = detector.save_detection_csv(comparison, output_dir / "yolo_detections.csv")

    return {
        "detections_by_date": detections_by_date,
        "comparison":          comparison,
        "geojson":             geojson,
        "geojson_path":        geojson_path,
        "csv_path":            csv_path,
        "annotated_paths":     annotated,
    }
