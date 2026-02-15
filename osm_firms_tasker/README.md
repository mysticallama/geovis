# Water Infrastructure Monitoring Pipeline

A modular Python pipeline for monitoring water infrastructure using satellite data, OpenStreetMap, NASA FIRMS thermal detection, and ACLED conflict event data.

## Features

- **AOI Input**: Load areas of interest from coordinates, GeoJSON files, or shapefiles
- **NASA FIRMS Integration**: Query thermal anomalies from VIIRS and MODIS satellites
- **ACLED Integration**: Query conflict event data for contextual analysis
- **OSM Integration**: Query water infrastructure features from OpenStreetMap
- **Spatial Analysis**: Proximity filtering and spatial joins (50m radius default)
- **Smart Classification**: Classify detections as confirmed, potential (novel), or known events
- **Export Formats**: Planet Labs, Maxar Discovery (STAC), YOLOv8, DBSCAN/HDBSCAN, Siamese networks

## Installation

```bash
# Clone or download the repository
cd osm_firms_tasker

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## API Keys Required

1. **NASA FIRMS** (for thermal/fire data) - Two options:

   **Option A: GIBS WMTS (No API key, ~30-90 day archive)**
   - No API key required
   - Data sources: VIIRS NOAA-20, VIIRS SNPP, MODIS Terra/Aqua
   - Limited to NRT data (~30-90 day rolling window)
   - Documentation: https://nasa-gibs.github.io/gibs-api-docs/

   **Option B: FIRMS REST API (Requires MAP_KEY, up to 1 year archive)**
   - Get your MAP_KEY at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
   - Set: `export FIRMS_MAP_KEY='your-32-char-key'`
   - Data sources: VIIRS NOAA-20/21, VIIRS SNPP, MODIS (NRT and Standard)
   - Supports up to 365 days of historical data
   - Includes confidence levels, FRP (Fire Radiative Power), and more metadata
   - Documentation: https://firms.modaps.eosdis.nasa.gov/api/area/

2. **ACLED** (for conflict data)
   - Register at: https://acleddata.com/register/
   - API docs: https://acleddata.com/api-documentation/getting-started
   - Set: `export ACLED_EMAIL='your@email.com'`
   - Set: `export ACLED_PASSWORD='your-password'`
   - Note: Uses OAuth authentication. Access tokens valid for 24 hours (auto-refreshed)

3. **Planet Labs** (optional, for Planet satellite exports)
   - Register at: https://www.planet.com/
   - Set: `export PL_API_KEY='your-key'`

4. **Maxar** (optional, for Maxar Discovery exports)
   - Register at: https://developers.maxar.com/
   - Set: `export MAXAR_API_KEY='your-key'`

## Quick Start

```python
from water_pipeline import run_water_infrastructure_analysis

# Run analysis on a bounding box [west, south, east, north]
results = run_water_infrastructure_analysis(
    aoi=[-122.5, 37.7, -122.3, 37.9],
    output_dir="./results"
)

# Or from a GeoJSON file
results = run_water_infrastructure_analysis(
    aoi="my_region.geojson",
    output_dir="./results"
)

# Access results
print(f"Infrastructure: {len(results['infrastructure']['features'])}")
print(f"Thermal detections: {len(results['thermal_detections']['features'])}")
print(f"Novel detections: {results['summary']['novel_count']}")
```

## Step-by-Step Usage

```python
from water_pipeline import (
    IntegratedPipeline,
    PipelineConfig,
    AOIHandler,
)

# Create custom config
config = PipelineConfig(
    thermal_lookback_days=14,
    acled_lookback_days=14,
    proximity_radius_meters=100,
    output_dir="./output"
)

# Create pipeline
pipeline = IntegratedPipeline(config)

# Load AOI
pipeline.load_aoi("my_region.geojson", name="study_area")

# Query infrastructure
infrastructure = pipeline.query_infrastructure("study_area")

# Query thermal detections
thermal = pipeline.query_thermal_detections("study_area")

# Filter near infrastructure
filtered = pipeline.filter_near_infrastructure(thermal, infrastructure)

# Query ACLED events
acled = pipeline.query_conflict_events("study_area")

# Enrich and classify
enriched = pipeline.enrich_detections(filtered, acled, infrastructure)

# Export results
exports = pipeline.export_results(enriched, "study_area")
```

## FIRMS REST API Usage

For longer historical queries (up to 1 year), use the FIRMS REST API:

```python
from water_pipeline import (
    FIRMSAPIClient,
    FIRMSAPIConfig,
    AOIHandler,
    query_firms_thermal_anomalies,
)

# Option 1: Using environment variable (FIRMS_MAP_KEY)
aoi = AOIHandler.from_bbox(42.5, 12.5, 45.5, 17.5)  # Yemen
thermal = query_firms_thermal_anomalies(
    aoi,
    days=30,
    source="VIIRS_NOAA20_NRT",
    confidence_filter="nominal"  # "high", "nominal", or None (all)
)

# Option 2: Explicit configuration
config = FIRMSAPIConfig(map_key="your-32-char-key")
client = FIRMSAPIClient(config)
features = client.query(
    aoi,
    days=30,
    source="VIIRS_NOAA20_NRT",
    date="2025-01-15"  # Optional end date
)
thermal = client.to_geojson(features)

print(f"Found {len(thermal['features'])} fire detections")
```

### Available FIRMS Sources

| Source | Resolution | Description |
|--------|------------|-------------|
| `VIIRS_NOAA20_NRT` | 375m | VIIRS NOAA-20 Near Real-Time |
| `VIIRS_NOAA21_NRT` | 375m | VIIRS NOAA-21 Near Real-Time |
| `VIIRS_SNPP_NRT` | 375m | VIIRS Suomi NPP Near Real-Time |
| `MODIS_NRT` | 1km | MODIS Combined Near Real-Time |
| `VIIRS_NOAA20_SP` | 375m | VIIRS NOAA-20 Standard Processing |
| `VIIRS_SNPP_SP` | 375m | VIIRS SNPP Standard Processing |

## OSM-Only Mode (No API Keys)

```python
from water_pipeline import AOIHandler, query_water_infrastructure

# Query water infrastructure without API keys
aoi = AOIHandler.from_bbox(-122.5, 37.7, -122.3, 37.9)
infrastructure = query_water_infrastructure(aoi)

print(f"Found {len(infrastructure['features'])} water features")
```

## Classification Categories

Detections are classified into four categories:

| Category | Description |
|----------|-------------|
| `confirmed_incident` | ACLED event match AND near water infrastructure |
| `potential_incident` | Near water infrastructure, no ACLED match (novel) |
| `known_event` | ACLED event match, not near water infrastructure |
| `unclassified` | No matches found |

## Export Formats

### Satellite APIs
- **Planet Labs**: GeoJSON + order parameters for tasking API
- **Maxar Discovery**: STAC-compatible search request with `intersects` geometry
  - Supports WorldView (wv01, wv02, wv03) collections
  - Cloud cover filtering
  - Datetime range queries
- **Generic**: Configurable format for other providers

### ML Models
- **YOLOv8**: Bounding box annotations in YOLO format
- **DBSCAN/HDBSCAN**: Point coordinates for clustering
- **Siamese Networks**: Feature pairs for similarity learning
- **Segmentation**: Binary/multi-class masks

## Configuration

Create a `config.yaml` file (see `config.yaml.template`):

```yaml
temporal:
  thermal_lookback_days: 7
  acled_lookback_days: 7

spatial:
  proximity_radius_meters: 50.0
  acled_match_radius_meters: 500.0

water_infrastructure:
  - key: water
    value: reservoir
  - key: waterway
    value: dam
  - key: waterway
    value: canal
```

## Project Structure

```
osm_firms_tasker/
├── water_pipeline/
│   ├── __init__.py      # Public API exports
│   ├── aoi.py           # AOI input handling
│   ├── firms.py         # NASA GIBS WMTS client (NRT, no API key)
│   ├── firms_api.py     # NASA FIRMS REST API (requires MAP_KEY)
│   ├── acled.py         # ACLED API client
│   ├── osm.py           # OpenStreetMap Overpass API
│   ├── spatial.py       # Proximity analysis
│   ├── enrichment.py    # Metadata enrichment
│   ├── export.py        # Export for APIs/ML
│   └── pipeline.py      # Main orchestration
├── requirements.txt
├── setup.py
├── config.yaml.template
├── example.py
├── test_pipeline_pre_export.py  # Test pipeline (supports GIBS or FIRMS API)
└── README.md
```

## Data Flow

```
AOI Input → AOIHandler → GeoJSON Geometry
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   OSM Overpass          NASA FIRMS              ACLED
 (infrastructure)    (thermal/fire)           (conflict)
        │                     │                     │
        │           ┌─────────┴─────────┐           │
        │           ▼                   ▼           │
        │      GIBS WMTS          FIRMS REST        │
        │    (NRT, no key)      (MAP_KEY, 1yr)      │
        │           └─────────┬─────────┘           │
        └──────────┬──────────┘                     │
                   ▼                                │
          Proximity Filter (50m)                    │
                   │                                │
                   └──────────────┬─────────────────┘
                                  ▼
                    Metadata Enrichment & Classification
                                  │
            ┌─────────────────────┼─────────────────┐
            ▼                     ▼                 ▼
       Planet API            Maxar API         ML Export
        Export                Export        (YOLO/DBSCAN)
```

## License

MIT License
