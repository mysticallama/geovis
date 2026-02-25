"""
Planet Labs imagery downloader and tile generator.

Downloads the best PSScene for each of two user-specified target dates
(``before_date`` and ``after_date``) within a configurable search window
(±``search_window_days`` around each target date).  The two images form a
before/after pair for xView change detection.

Tiles are 640 × 640 pixels — the input size expected by xView models trained
on DigitalGlobe WorldView imagery (GSD correction applied in yolov8.py).

Workflow:
    1. ``search_near_date()``        — find the best PSScene within a window
    2. ``activate_and_download()``   — activate assets and stream GeoTIFFs
    3. ``tile_image()``              — slice each GeoTIFF into 640×640 PNGs
    4. ``download_date_pair()``      — end-to-end convenience wrapper

Authentication:
    Set ``PL_API_KEY`` in ``.env``.  Planet v1 API uses HTTP Basic Auth
    (API key as username, empty password).

Planet API v1 docs: https://developers.planet.com/docs/apis/data/
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

PLANET_API_BASE = "https://api.planet.com/data/v1"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlanetConfig:
    """
    Configuration for Planet Labs imagery acquisition.

    Attributes:
        planet_api_key:         PL_API_KEY environment variable.
        item_type:              Planet item type (PSScene = ~3 m PlanetScope).
        asset_type:             Asset to download (surface reflectance 4-band).
        min_coverage:           Minimum AOI coverage fraction (0–1).
        max_cloud_cover:        Maximum cloud cover percentage (0–100).
        tile_size:              Tile width and height in pixels (640 for xView).
        search_window_days:     Half-window (days) around each target date.
        activation_poll_interval: Seconds between asset status polls.
        activation_timeout:     Max seconds to wait for asset activation.
        rate_limit_delay:       Seconds between API requests.
        cache_downloads:        If True, skip download if GeoTIFF already exists.
    """
    planet_api_key: Optional[str] = None
    item_type: str = "PSScene"
    asset_type: str = "ortho_analytic_4b_sr"
    min_coverage: float = 0.90
    max_cloud_cover: float = 10.0
    tile_size: int = 640
    search_window_days: int = 7
    activation_poll_interval: float = 10.0
    activation_timeout: float = 900.0
    rate_limit_delay: float = 1.0
    cache_downloads: bool = True

    def __post_init__(self):
        if self.planet_api_key is None:
            self.planet_api_key = os.environ.get("PL_API_KEY")
        if not self.planet_api_key:
            raise ValueError(
                "Planet API key required. "
                "Set PL_API_KEY in .env — get a key at https://www.planet.com/"
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PlanetImageryClient:
    """
    Planet Labs Data API v1 client for before/after imagery pairs.

    Downloads the single best PSScene within ±``search_window_days`` of each
    target date and tiles it into 640 × 640 PNG patches.

    Example::

        cfg = PlanetConfig(max_cloud_cover=5.0)
        client = PlanetImageryClient(cfg)
        before_item = client.search_near_date(aoi, "2023-10-01")
        after_item  = client.search_near_date(aoi, "2024-02-01")
        paths  = client.activate_and_download([before_item, after_item], out_dir)
        tiles  = client.tile_all(paths, tile_dir)
    """

    def __init__(self, config: Optional[PlanetConfig] = None):
        self.config = config or PlanetConfig()
        self.auth = HTTPBasicAuth(self.config.planet_api_key, "")
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers["Content-Type"] = "application/json"
        self._last_request = 0.0

    # ── internal helpers ────────────────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, **kwargs) -> requests.Response:
        self._rate_limit()
        resp = self.session.get(url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp

    def _post(self, url: str, payload: Dict) -> requests.Response:
        self._rate_limit()
        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp

    def _aoi_geometry(self, aoi: Dict) -> Dict:
        """Simplify complex AOI geometry to stay within Planet API limits."""
        geom = shape(aoi)
        if len(str(mapping(geom))) > 8000:
            geom = geom.simplify(0.001, preserve_topology=True)
        return mapping(geom)

    # ── search ──────────────────────────────────────────────────────────────

    def search_near_date(
        self,
        aoi: Dict,
        target_date: str,
        label: str = "scene",
    ) -> Optional[Dict]:
        """
        Find the best PSScene within ±``search_window_days`` of *target_date*.

        Selection criteria (in order):
            1. Cloud cover ≤ ``max_cloud_cover``
            2. AOI coverage ≥ ``min_coverage``
            3. Among qualifying scenes, lowest cloud cover wins

        Args:
            aoi:         GeoJSON geometry of the analysis AOI.
            target_date: Centre of the search window (``YYYY-MM-DD``).
            label:       Human-readable label for log messages.

        Returns:
            Planet item dict (with ``_target_date`` and ``_label`` keys
            injected), or ``None`` if no qualifying scene was found.
        """
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        window = self.config.search_window_days
        start = (dt - timedelta(days=window)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=window)).strftime("%Y-%m-%d")

        logger.info(f"Planet search [{label}]: {start} → {end}  (target={target_date})")

        geom = self._aoi_geometry(aoi)
        payload = {
            "item_types": [self.config.item_type],
            "filter": {
                "type": "AndFilter",
                "config": [
                    {"type": "GeometryFilter",   "field_name": "geometry",     "config": geom},
                    {"type": "DateRangeFilter",   "field_name": "acquired",
                     "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
                    {"type": "RangeFilter",       "field_name": "cloud_cover",
                     "config": {"lte": self.config.max_cloud_cover / 100.0}},
                    {"type": "RangeFilter",       "field_name": "view_angle",
                     "config": {"lte": 25.0}},
                    {"type": "StringInFilter",    "field_name": "item_type",
                     "config": [self.config.item_type]},
                ],
            },
        }

        try:
            resp = self._post(f"{PLANET_API_BASE}/quick-search?_page_size=50", payload)
        except requests.HTTPError as exc:
            logger.warning(f"Planet search failed for {label}: {exc}")
            return None

        items = resp.json().get("features", [])
        if not items:
            logger.warning(f"  No Planet scenes found for [{label}] window {start}–{end}")
            return None

        qualifying = []
        for item in items:
            props = item.get("properties", {})
            coverage = props.get("item_resource_coverage") or props.get("coverage") or 1.0
            if float(coverage) >= self.config.min_coverage:
                qualifying.append(item)

        if not qualifying:
            logger.warning(
                f"  No scenes with ≥{self.config.min_coverage*100:.0f}% coverage "
                f"for [{label}] {start}–{end}"
            )
            return None

        best = min(qualifying, key=lambda x: x.get("properties", {}).get("cloud_cover", 1.0))
        cloud = best.get("properties", {}).get("cloud_cover", "?")
        cloud_str = f"{cloud:.3f}" if isinstance(cloud, float) else str(cloud)
        logger.info(f"  Selected [{label}] scene {best['id']} (cloud={cloud_str})")

        best["_target_date"] = target_date
        best["_label"] = label
        return best

    # ── download ────────────────────────────────────────────────────────────

    def _get_asset(self, item_id: str) -> Optional[Dict]:
        url = (
            f"{PLANET_API_BASE}/item-types/{self.config.item_type}"
            f"/items/{item_id}/assets"
        )
        assets = self._get(url).json()
        asset = assets.get(self.config.asset_type)
        if asset is None or asset.get("status") == "unavailable":
            for fallback in ("ortho_analytic_4b", "ortho_analytic_4b_sr", "ortho_visual"):
                if fallback == self.config.asset_type:
                    continue
                candidate = assets.get(fallback)
                if candidate is not None and candidate.get("status") != "unavailable":
                    asset = candidate
                    logger.warning(
                        f"Asset '{self.config.asset_type}' unavailable for {item_id}; "
                        f"using '{fallback}'"
                    )
                    break
        return asset

    def _activate(self, asset: Dict) -> bool:
        """Activate an asset and poll until it is ready."""
        activate_url = asset.get("_links", {}).get("activate")
        if not activate_url:
            logger.warning("No activation link in asset")
            return False

        self._rate_limit()
        try:
            self.session.post(activate_url, timeout=30)
        except requests.HTTPError:
            pass

        status_url = asset.get("_links", {}).get("_self", activate_url)
        deadline = time.time() + self.config.activation_timeout

        while time.time() < deadline:
            time.sleep(self.config.activation_poll_interval)
            self._rate_limit()
            resp = self.session.get(status_url, timeout=30)
            if resp.status_code != 200:
                continue
            if resp.json().get("status") == "active":
                return True
            logger.debug(f"  Asset status: {resp.json().get('status')} — waiting…")

        logger.warning(f"Asset activation timed out ({self.config.activation_timeout}s)")
        return False

    def activate_and_download(
        self,
        items: List[Dict],
        output_dir: Path,
    ) -> List[Path]:
        """
        Activate and download GeoTIFF assets for a list of Planet items.

        Args:
            items:      Planet item dicts from :meth:`search_near_date`.
            output_dir: Directory to write downloaded GeoTIFFs.

        Returns:
            List of :class:`pathlib.Path` objects for the downloaded files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        for item in items:
            item_id   = item["id"]
            item_date = item.get("properties", {}).get("acquired", "unknown")[:10]
            label     = item.get("_label", "scene")
            out_path  = output_dir / f"{label}_{item_date}_{item_id}.tif"

            if self.config.cache_downloads and out_path.exists():
                logger.info(f"  [cache] {out_path.name}")
                downloaded.append(out_path)
                continue

            logger.info(f"Downloading {label} scene {item_id} ({item_date})…")

            asset = self._get_asset(item_id)
            if asset is None:
                logger.warning(f"  No usable asset for {item_id} — skipping")
                continue

            if asset.get("status") != "active":
                logger.info(f"  Activating…")
                if not self._activate(asset):
                    logger.warning(f"  Activation failed for {item_id} — skipping")
                    continue
                asset = self._get_asset(item_id)
                if asset is None:
                    continue

            dl_url = asset.get("location")
            if not dl_url:
                logger.warning(f"  No download URL for {item_id}")
                continue

            self._rate_limit()
            with self.session.get(dl_url, stream=True, timeout=300) as dl_resp:
                dl_resp.raise_for_status()
                with open(out_path, "wb") as fh:
                    for chunk in dl_resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)

            logger.info(f"  Saved → {out_path.name}")
            downloaded.append(out_path)

        return downloaded

    # ── tiling ──────────────────────────────────────────────────────────────

    def tile_image(
        self,
        image_path: Path,
        output_dir: Path,
        image_date: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[Dict]:
        """
        Slice a GeoTIFF into 640 × 640 pixel PNG tiles.

        Only non-blank tiles (at least one non-zero pixel) are saved.
        A companion JSON file mapping tile paths to WGS-84 bounds is written
        to *output_dir* as ``tiles_metadata.json``.

        Args:
            image_path: Path to the downloaded GeoTIFF.
            output_dir: Directory for tile PNGs.
            image_date: ISO date string embedded in tile metadata.
            label:      "before" or "after" — embedded in metadata.

        Returns:
            List of tile metadata dicts:
            ``{path, col, row, pixel_bounds, geo_bounds, image_date, label, image_path}``
        """
        try:
            import rasterio
            from rasterio.crs import CRS
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "rasterio and Pillow required: pip install rasterio Pillow"
            ) from exc

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = self.config.tile_size
        metas = []

        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            transform, crs = src.transform, src.crs

            n_bands = min(src.count, 3)
            data = src.read(list(range(1, n_bands + 1)))

            # Normalise each band to uint8 (2nd–98th percentile)
            img_u8 = np.zeros((n_bands, height, width), dtype=np.uint8)
            for b in range(n_bands):
                band = data[b].astype(float)
                valid = band[band > 0]
                if valid.size:
                    lo, hi = np.percentile(valid, (2, 98))
                else:
                    lo, hi = 0.0, 1.0
                hi = max(hi, lo + 1)
                img_u8[b] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

            wgs84 = CRS.from_epsg(4326)
            stem  = image_path.stem

            for row_s in range(0, height, ts):
                for col_s in range(0, width, ts):
                    row_e = min(row_s + ts, height)
                    col_e = min(col_s + ts, width)

                    tile_data = img_u8[:, row_s:row_e, col_s:col_e]
                    if tile_data.max() == 0:
                        continue   # skip blank tiles

                    # Pad to exact tile_size at image edges
                    pad_r = ts - (row_e - row_s)
                    pad_c = ts - (col_e - col_s)
                    if pad_r or pad_c:
                        tile_data = np.pad(
                            tile_data, ((0, 0), (0, pad_r), (0, pad_c)), mode="constant"
                        )

                    arr = np.transpose(tile_data, (1, 2, 0))
                    if arr.shape[2] == 1:
                        img = Image.fromarray(arr[:, :, 0], mode="L")
                    else:
                        img = Image.fromarray(arr[:, :, :3], mode="RGB")

                    fname = f"{stem}_r{row_s:05d}_c{col_s:05d}.png"
                    tile_path = output_dir / fname
                    img.save(tile_path)

                    # Geo-reference the tile corners
                    xs = [col_s, col_e, col_s, col_e]
                    ys = [row_s, row_s, row_e, row_e]
                    gx, gy = rasterio.transform.xy(transform, ys, xs)

                    if crs != wgs84:
                        from rasterio.warp import transform as rp_transform
                        gx, gy = rp_transform(crs, wgs84, gx, gy)

                    metas.append({
                        "path":        str(tile_path),
                        "col":         col_s,
                        "row":         row_s,
                        "pixel_bounds": [col_s, row_s, col_e, row_e],
                        "geo_bounds":  {
                            "west":  float(min(gx)),
                            "south": float(min(gy)),
                            "east":  float(max(gx)),
                            "north": float(max(gy)),
                        },
                        "image_date": image_date or image_path.stem[:10],
                        "label":      label or "scene",
                        "image_path": str(image_path),
                    })

        logger.info(
            f"Tiled {image_path.name}: {len(metas)} non-blank {ts}×{ts} tiles"
        )
        return metas

    def tile_all(self, image_paths: List[Path], output_dir: Path) -> List[Dict]:
        """Tile all downloaded GeoTIFFs and return combined metadata."""
        all_tiles: List[Dict] = []
        for img_path in image_paths:
            img_path = Path(img_path)
            label = "before" if "_before_" in img_path.stem else "after"
            tile_dir = output_dir / img_path.stem
            tiles = self.tile_image(
                img_path, tile_dir,
                image_date=img_path.stem.split("_")[1] if "_" in img_path.stem else img_path.stem[:10],
                label=label,
            )
            all_tiles.extend(tiles)
        return all_tiles


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def download_date_pair(
    aoi: Dict,
    before_date: str,
    after_date: str,
    output_dir: Path,
    config: Optional[PlanetConfig] = None,
) -> Dict:
    """
    Download and tile imagery for a before/after date pair.

    Searches for the best PSScene within ±``search_window_days`` of each
    target date, downloads the GeoTIFFs, tiles them into 640 × 640 PNG
    patches, and writes a ``tiles_metadata.json`` index file.

    Args:
        aoi:         GeoJSON geometry dict (the AOI to image).
        before_date: Baseline date ``YYYY-MM-DD``.
        after_date:  Comparison date ``YYYY-MM-DD``.
        output_dir:  Root output directory.
        config:      Optional :class:`PlanetConfig`.

    Returns:
        Dict with keys:
            ``"items"``            — Planet item dicts found (0–2)
            ``"images"``           — downloaded GeoTIFF :class:`Path` objects
            ``"tiles"``            — flat list of tile metadata dicts
            ``"tile_metadata_path"`` — path to saved ``tiles_metadata.json``
    """
    output_dir = Path(output_dir)
    client = PlanetImageryClient(config)

    # Search for one scene per target date
    items = []
    for target_date, lbl in [(before_date, "before"), (after_date, "after")]:
        item = client.search_near_date(aoi, target_date, label=lbl)
        if item is not None:
            items.append(item)
        else:
            logger.warning(f"No qualifying Planet scene found for [{lbl}] {target_date}")

    if not items:
        logger.warning("No Planet scenes found — skipping imagery step")
        return {"items": [], "images": [], "tiles": [], "tile_metadata_path": None}

    # Download
    geotiff_dir = output_dir / "geotiffs"
    image_paths = client.activate_and_download(items, geotiff_dir)

    # Tile
    tile_dir = output_dir / "tiles"
    tiles = client.tile_all(image_paths, tile_dir)

    # Save metadata
    meta_path = output_dir / "tiles_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(tiles, fh, indent=2)
    logger.info(f"Tile metadata saved → {meta_path}")

    return {
        "items":               items,
        "images":              image_paths,
        "tiles":               tiles,
        "tile_metadata_path":  meta_path,
    }
