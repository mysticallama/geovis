# Water & Food Infrastructure Monitoring Pipeline

A modular Python pipeline for monitoring **water and food infrastructure** in
conflict-affected areas — without relying on NASA FIRMS thermal data.

The pipeline combines:

| Source | Purpose |
|--------|---------|
| **OpenStreetMap** (Overpass API) | Water supply, waterways, irrigation, food storage, markets, agriculture |
| **ACLED** | Conflict events over a user-defined time window |
| **Planet Labs** (PSScene) | Before/after satellite imagery pairs (640 × 640 px tiles) |
| **xView / YOLOv8** | Overhead object detection with GSD correction and change analysis |
| **WFP Logistics Cluster** (HDX) | Roads, airports, border crossings, storage facilities |
| **Mapbox GL JS** | Interactive web application fusing ACLED heatmap + WFP overlays |

---

## Installation

```bash
cd working_paper_version

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

Key packages: `numpy>=2.0`, `requests`, `pandas`, `shapely`, `geopandas`,
`pyproj`, `rasterio`, `Pillow`, `ultralytics`, `python-dotenv`.

---

## API Keys

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Key | Required for | Where to get it |
|-----|-------------|-----------------|
| `ACLED_EMAIL` + `ACLED_PASSWORD` | Step 2 — conflict events | [acleddata.com/register](https://acleddata.com/register/) |
| `PL_API_KEY` | Step 4 — satellite imagery | [planet.com](https://www.planet.com/) |
| `MAPBOX_TOKEN` | Step 7 — web app | [account.mapbox.com](https://account.mapbox.com/access-tokens/) |
| `HDX_API_KEY` *(optional)* | Step 6 — WFP data | [data.humdata.org](https://data.humdata.org/user/register) |

ACLED uses OAuth — access tokens are automatically fetched and refreshed from
your email + password.

---

## Quick Start

### Full pipeline

```bash
python run_pipeline.py
```

### Example script (recommended starting point)

```bash
python example.py
```

### CLI flags

| Flag | Effect |
|------|--------|
| `--skip-imagery` | Skip Planet download (Steps 4–5) |
| `--skip-yolo` | Skip xView inference (Step 5) |
| `--skip-wfp` | Skip WFP data download (Step 6) |
| `--skip-fusion` | Skip Mapbox generation (Step 7) |
| `--osm-extended` | Use the extended OSM tag set (broader infrastructure coverage) |
| `--verbose` | Enable DEBUG-level logging at every step |

```bash
# Test ACLED connection only
python run_pipeline.py --debug-acled

# Check Planet imagery availability (no download)
python run_pipeline.py --debug-imagery

# Verify xView model loads correctly
python run_pipeline.py --debug-yolo

# Run all diagnostics
python run_pipeline.py --debug-all
```

---

## Configuration

Edit the `CONFIGURATION` block at the top of `run_pipeline.py`:

```python
# ── AOI ─────────────────────────────────────────────────────────────────
BBOX = [33.0, 11.0, 37.0, 15.0]   # [west, south, east, north]

# ── ACLED date range ─────────────────────────────────────────────────────
ACLED_START_DATE = "2023-10-01"
ACLED_END_DATE   = "2024-04-01"

# ── Planet imagery before/after pair ─────────────────────────────────────
BEFORE_DATE = "2023-10-01"   # baseline date (best scene ±WINDOW)
AFTER_DATE  = "2024-04-01"   # comparison date
PLANET_SEARCH_WINDOW_DAYS = 7

# ── WFP country ──────────────────────────────────────────────────────────
WFP_COUNTRY_ISO3 = "SDN"
WFP_COUNTRY_NAME = "Sudan"
```

---

## Pipeline Steps

```
Step 1  Load AOI
Step 2  Query ACLED conflict events      (full AOI, explicit date range)
Step 3  Query OSM infrastructure          (water + food tags, full AOI)
Step 4  Download Planet imagery           (before + after date pair)
Step 5  xView / YOLOv8 inference          (annotated tiles, CSV, OSM proximity)
Step 6  Download WFP logistics data       (HDX / LogIE → fallback OSM)
Step 7  ACLED + WFP Mapbox web app        (heatmap + vector overlay)
```

### Step 3 — OSM infrastructure tags

The OSM query covers **both water supply and food infrastructure**:

| Category | Examples |
|----------|---------|
| `water_supply` | Reservoirs, water towers, pumping stations, pipelines, wells, desalination plants |
| `waterway` | Dams, canals, rivers, drains, ditches |
| `irrigation` | Irrigation canals, irrigation dams |
| `food_storage` | Grain silos, warehouses, barns, storage tanks |
| `food_market` | Marketplaces, supermarkets, food banks |
| `agriculture` | Farmland, orchards, greenhouses, allotments |
| `power` | Generators, power plants, substations |

Use `--osm-extended` to include a broader supplementary tag set.

### Step 4 — Planet imagery

Requires `PL_API_KEY`.  Searches for the single best PSScene (lowest cloud
cover, ≥ 90% AOI coverage) within ±`PLANET_SEARCH_WINDOW_DAYS` of each
target date.  Downloads GeoTIFFs and tiles them into **640 × 640 px** PNG
patches.

If a GeoTIFF already exists in the output directory it is reused
(set `cache_downloads = False` in `PlanetConfig` to override).

### Step 5 — xView / YOLOv8

| Setting | Default | Notes |
|---------|---------|-------|
| Training GSD | 0.3 m | xView / DigitalGlobe WorldView |
| Inference GSD | 3.0 m | Planet PSScene |
| Scale factor | 10× | Tiles logically upscaled before inference |
| Effective imgsz | 6400 px | After GSD correction (multiple of 32) |

**Classes of interest** (xView):

| ID | Label |
|----|-------|
| 15 | Fixed-wing Aircraft |
| 40 | Damaged Building |
| 48 | Storage Tank |
| 50 | Shipping Container |
| 55 | Engineering Vehicle (Heavy) |
| 58 | Mobile Crane |

**Outputs per run:**
- `yolo/annotated_tiles/` — PNG tiles with bounding boxes drawn
- `yolo/yolo_detections.csv` — detection table (class, confidence, geo coords, OSM proximity, change status)
- `yolo/yolo_detections.geojson` — GeoJSON FeatureCollection

**OSM proximity cross-reference:** each detection records whether it falls
within `OSM_PROXIMITY_RADIUS_METERS` of a known infrastructure feature, plus
the distance and category of the nearest feature.

### Step 6 — WFP logistics data

Searches HDX for datasets tagged with the configured ISO3 country code and
logistics keywords (roads, airports, border crossings, storage).  Downloads
and converts to GeoJSON.  If no HDX datasets are found, falls back to an OSM
Overpass query for equivalent features within the AOI.

### Step 7 — Mapbox web application

Generates a **self-contained HTML file** (no server required) containing:

- **ACLED heatmap** with intensity ∝ fatalities — colour-ramps from blue → red
- **ACLED time slider** — filters events by date (cumulative up to each date)
- **Point view toggle** — switch between heatmap and individual event points
- **WFP roads** — blue line layer
- **WFP airports** — red circle markers
- **WFP border crossings** — orange circle markers
- **WFP ports / storage** — purple circle markers
- **Interactive legend** with per-layer visibility toggles
- **Pop-ups**: ACLED events on hover, WFP features on click

Open `output/fusion/map.html` in any modern browser.

---

## Output Files

```
output/
├── conflict_events.geojson        # Step 2 — ACLED events
├── infrastructure.geojson         # Step 3 — OSM water + food features
├── imagery/
│   ├── geotiffs/
│   │   ├── before_<date>_<id>.tif
│   │   └── after_<date>_<id>.tif
│   └── tiles_metadata.json        # tile → geo_bounds mapping
├── yolo/
│   ├── annotated_tiles/           # PNG tiles with detection boxes
│   ├── yolo_detections.csv        # detection + change table
│   └── yolo_detections.geojson
├── wfp/
│   ├── downloads/                 # raw HDX downloads
│   ├── wfp_roads.geojson
│   ├── wfp_airports.geojson
│   ├── wfp_crossings.geojson
│   └── wfp_storage.geojson
└── fusion/
    └── map.html                   # self-contained Mapbox web app
```

---

## Project Structure

```
working_paper_version/
├── water_osm_pipeline/
│   ├── __init__.py       Public API
│   ├── aoi.py            AOI input (bbox, GeoJSON, shapefile, WKT)
│   ├── osm.py            OSM Overpass client + extended water/food tags
│   ├── acled.py          ACLED OAuth client (explicit date range)
│   ├── imagery.py        Planet imagery client (before/after pair, 640×640)
│   ├── yolov8.py         xView detector + annotated tiles + CSV + OSM proximity
│   ├── wfp.py            WFP logistics data from HDX / OSM fallback
│   └── fusion.py         Mapbox GL JS web app generator
├── run_pipeline.py        Main entry point
├── example.py             Example usage script
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Debugging

### ACLED

```bash
python run_pipeline.py --debug-acled
```

Tests OAuth authentication and runs a minimal (5-row) query to confirm the
API connection is working.

### Planet imagery

```bash
python run_pipeline.py --debug-imagery
```

Performs a Quick Search for both before/after dates and lists matching scenes
by date and cloud cover.  **No imagery is downloaded.**

### xView / YOLOv8

```bash
python run_pipeline.py --debug-yolo
```

Loads the configured model and prints GSD correction parameters and class
mappings.  **No inference is run.**

---

## Obtaining xView Model Weights

The pipeline defaults to `xview.pt`.  To obtain xView-trained YOLOv8 weights:

1. Download xView annotations from [xviewdataset.org](http://xviewdataset.org/)
2. Train with ultralytics:
   ```bash
   yolo detect train data=xview.yaml model=yolov8s.pt epochs=100
   ```
3. Set `YOLO_MODEL_PATH` to the resulting `best.pt`

Community-converted xView YOLOv8 weights are also available on
Ultralytics HUB and Hugging Face Model Hub.

---

## License

MIT License
