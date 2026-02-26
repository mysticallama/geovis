#!/usr/bin/env python3
"""
Enhanced Sudan Conflict-Logistics Map
======================================

Generates an enhanced Mapbox GL JS visualization on top of the Sudan pipeline
output, adding three analytical layers:

  1. NDVI mockup overlay        — synthetic vegetation index grid modelling
                                  the arid-to-Sahel gradient and Jebel Marra
                                  mountain massif.  NOT real satellite data.

  2. Road disruption risk        — ACLED conflict events weighted by proximity
     heatmap                      to OSM major roads; hotspots flag supply
                                  corridors at elevated risk.

  3. Confirmed infrastructure    — ACLED events whose notes reference roads,
     incidents                    bridges, convoys, or transport networks,
                                  labelled "Infrastructure Attack / Incident".

Reads from standard Sudan pipeline output (run run_pipeline.py first):
    output/conflict_events.geojson
    output/wfp/wfp_osm_roads.geojson

Outputs:
    output/fusion/map_enhanced_sudan.html
"""

import json
import logging
import math
import os
import random
from pathlib import Path

# Root of the working_paper_version project (one level up from country_runs/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enhanced_map")


# =============================================================================
# CONFIGURATION
# =============================================================================

AOI_BBOX       = [22.0, 10.5, 26.5, 16.5]   # [W, S, E, N]
ACLED_PATH     = _PROJECT_ROOT / "output" / "conflict_events.geojson"
OSM_ROADS_PATH = _PROJECT_ROOT / "output" / "wfp" / "wfp_osm_roads.geojson"
OUTPUT_PATH    = _PROJECT_ROOT / "output" / "fusion" / "map_enhanced_sudan.html"

MAPBOX_TOKEN      = os.environ.get("MAPBOX_TOKEN", "")
NDVI_GRID_STEP    = 0.20     # degrees per cell (~22 km × 22 km)
ROAD_PROXIMITY_KM = 15.0     # ACLED events within this km of a road are scored

MAP_TITLE  = "Sudan — Conflict Risk, Road Corridors & Vegetation (2024–2025)"
MAP_ZOOM   = 6.5
MAP_CENTER = [24.25, 13.5]   # AOI centroid

# ACLED note keywords that indicate infrastructure targeting
INFRA_KEYWORDS = [
    "road", "bridge", "convoy", "supply route", "ied", "landmine",
    "mine", "transport", "truck", "lorry", "humanitarian convoy",
    "aid convoy", "food convoy", "water tank", "water truck",
]


# =============================================================================
# HELPERS
# =============================================================================

def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _safe_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# =============================================================================
# 1.  NDVI MOCK GRID
# =============================================================================

def generate_ndvi_grid(bbox=AOI_BBOX, step=NDVI_GRID_STEP, seed=42) -> dict:
    """
    Produce a synthetic NDVI FeatureCollection over the bbox.

    Values reflect known geography of Western Sudan / Darfur:
      - Saharan north (low NDVI) → Sahelian south (higher NDVI)
      - Jebel Marra volcanic massif (~24.0 E, 13.0 N) — elevated vegetation
      - Wadi corridors — sinusoidal east-west variation

    NOTE: This is a MOCKUP for demonstration purposes only.
    It does not represent real satellite-derived NDVI.
    """
    random.seed(seed)
    west, south, east, north = bbox
    features = []

    lon = west
    while lon < east - 1e-9:
        lat = south
        while lat < north - 1e-9:
            cx = lon + step / 2
            cy = lat + step / 2

            # 1. Latitude gradient: Sahel south → Sahara north
            lat_frac = (cy - south) / (north - south)
            base = 0.07 + 0.38 * (lat_frac ** 0.7)

            # 2. Jebel Marra massif (~24.0 E, 13.0 N)
            dx, dy = cx - 24.0, cy - 13.0
            jebel = 0.28 * math.exp(-(dx ** 2 + dy ** 2) / 2.0)

            # 3. Wadi (seasonal river) corridor sinusoidal effect
            wadi = 0.07 * math.sin(cy * 5.8 + 1.2) * (0.5 + 0.5 * math.cos(cx * 2.9))

            # 4. Reproducible spatial noise
            noise = random.uniform(-0.04, 0.04)

            ndvi = min(0.82, max(0.02, base + jebel + wadi + noise))

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[lon, lat], [lon + step, lat],
                                     [lon + step, lat + step], [lon, lat + step],
                                     [lon, lat]]]
                },
                "properties": {"ndvi": round(ndvi, 3)},
            })
            lat = round(lat + step, 8)
        lon = round(lon + step, 8)

    logger.info(f"NDVI grid: {len(features)} cells ({step}° × {step}°, mock data)")
    return {"type": "FeatureCollection", "features": features}


# =============================================================================
# 2.  ROAD PROXIMITY RISK EVENTS
# =============================================================================

def compute_road_risk_events(
    acled_fc: dict,
    roads_fc: dict,
    proximity_km: float = ROAD_PROXIMITY_KM,
) -> dict:
    """
    Score ACLED events by proximity to OSM road network.

    road_risk = max(0,  1 − dist_km / proximity_km)

    Only events within proximity_km of any road are returned; events beyond
    that threshold are excluded so the heatmap focuses on corridor risk.
    """
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("shapely not available — road risk layer will be empty")
        return {"type": "FeatureCollection", "features": []}

    road_geoms = [
        shape(f["geometry"])
        for f in roads_fc.get("features", [])
        if f.get("geometry")
    ]
    if not road_geoms:
        logger.warning("No road geometries — road risk layer empty")
        return {"type": "FeatureCollection", "features": []}

    roads_union = unary_union(road_geoms)
    prox_deg = proximity_km / 111.0   # rough degree conversion

    scored = []
    for feat in acled_fc.get("features", []):
        if not feat.get("geometry"):
            continue
        pt = shape(feat["geometry"])
        dist_deg = roads_union.distance(pt)
        if dist_deg <= prox_deg:
            score = round(max(0.0, 1.0 - dist_deg / prox_deg), 3)
            scored.append({
                **feat,
                "properties": {**feat["properties"], "road_risk": score},
            })

    logger.info(
        f"Road risk events: {len(scored)} within {proximity_km} km of roads "
        f"(out of {len(acled_fc.get('features', []))} total ACLED events)"
    )
    return {"type": "FeatureCollection", "features": scored}


# =============================================================================
# 3.  INFRASTRUCTURE INCIDENT FILTER
# =============================================================================

def filter_infrastructure_incidents(
    acled_fc: dict,
    keywords: list = INFRA_KEYWORDS,
) -> dict:
    """
    Return ACLED events whose notes reference road/bridge/convoy/supply keywords.
    Flags each feature with _infra_incident=true for popup rendering.
    """
    kw = [k.lower() for k in keywords]
    features = []
    for f in acled_fc.get("features", []):
        notes = str(f.get("properties", {}).get("notes", "")).lower()
        if any(k in notes for k in kw):
            features.append({
                **f,
                "properties": {**f["properties"], "_infra_incident": True},
            })
    logger.info(f"Infrastructure incidents: {len(features)} events match keywords")
    return {"type": "FeatureCollection", "features": features}


# =============================================================================
# 4.  HTML BUILDER
# =============================================================================

def build_html(
    acled_js: str,
    ndvi_js: str,
    road_risk_js: str,
    infra_js: str,
    osm_roads_js: str,
    dates_js: str,
    n_acled: int,
    n_road_risk: int,
    n_infra: int,
    n_roads: int,
    n_ndvi: int,
) -> str:
    token = MAPBOX_TOKEN
    cx, cy = MAP_CENTER
    title  = MAP_TITLE
    zoom   = MAP_ZOOM

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://unpkg.com/@mapbox/mapbox-gl-geocoder@5.0.3/dist/mapbox-gl-geocoder.css" rel="stylesheet"/>
<script src="https://unpkg.com/@mapbox/mapbox-gl-geocoder@5.0.3/dist/mapbox-gl-geocoder.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#111; color:#e5e7eb; }}
  #map {{ position:absolute; top:0; bottom:0; width:100%; }}

  /* ── Header ── */
  #header {{
    position:absolute; top:0; left:0; right:0; z-index:10;
    background:rgba(17,17,17,0.93); padding:9px 14px;
    display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    border-bottom:1px solid #374151;
  }}
  #header h1 {{ font-size:0.9rem; font-weight:600; color:#f9fafb; white-space:nowrap; }}
  .badge {{
    display:inline-block; padding:2px 8px; border-radius:9999px;
    font-size:0.68rem; font-weight:500; white-space:nowrap;
  }}
  .b-red    {{ background:#991b1b; color:#fca5a5; }}
  .b-orange {{ background:#7c2d12; color:#fb923c; }}
  .b-yellow {{ background:#713f12; color:#fde047; }}
  .b-amber  {{ background:#78350f; color:#fde68a; }}
  .b-green  {{ background:#14532d; color:#86efac; }}

  /* ── Legend ── */
  #legend {{
    position:absolute; bottom:36px; right:10px; z-index:10;
    background:rgba(17,17,17,0.93); border:1px solid #374151;
    border-radius:8px; padding:12px 14px; min-width:215px;
    max-height:85vh; overflow-y:auto;
  }}
  #legend h3 {{
    font-size:0.74rem; font-weight:600; margin-bottom:8px;
    color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em;
  }}
  .leg-section {{ margin-bottom:10px; }}
  .leg-section-title {{
    font-size:0.66rem; color:#6b7280; text-transform:uppercase;
    letter-spacing:0.07em; margin-bottom:4px; padding-bottom:2px;
    border-bottom:1px solid #1f2937;
  }}
  .leg-item {{
    display:flex; align-items:center; gap:8px;
    margin-bottom:5px; cursor:pointer; user-select:none;
  }}
  .leg-item:hover .leg-label {{ color:#f9fafb; }}
  .leg-swatch {{ width:14px; height:14px; border-radius:3px; flex-shrink:0; }}
  .leg-swatch.circle {{ border-radius:50%; }}
  .leg-swatch.line   {{ height:4px; border-radius:2px; }}
  .leg-label {{ font-size:0.75rem; color:#d1d5db; }}
  .leg-item.hidden .leg-label {{ opacity:0.4; }}
  .leg-ramp {{ margin-left:22px; margin-bottom:6px; }}
  .leg-ramp-bar {{ height:5px; border-radius:2px; width:110px; }}
  .leg-ramp-ends {{
    display:flex; justify-content:space-between;
    width:110px; font-size:0.61rem; color:#6b7280; margin-top:2px;
  }}
  .leg-ramp-note {{ font-size:0.63rem; color:#6b7280; margin-top:1px; }}

  /* ── Slider panel ── */
  #slider-panel {{
    position:absolute; bottom:36px; left:10px; z-index:10;
    background:rgba(17,17,17,0.93); border:1px solid #374151;
    border-radius:8px; padding:12px; min-width:280px;
  }}
  #slider-panel h3 {{ font-size:0.78rem; color:#9ca3af; text-transform:uppercase; margin-bottom:8px; font-weight:600; }}
  #date-range-label {{ font-size:0.8rem; color:#e5e7eb; margin-bottom:6px; }}
  #date-slider {{ width:100%; accent-color:#ef4444; }}
  #slider-info {{ font-size:0.7rem; color:#9ca3af; margin-top:4px; }}
  #heatmap-toggle {{
    margin-top:10px; font-size:0.76rem;
    background:#1f2937; border:1px solid #374151; color:#d1d5db;
    padding:4px 10px; border-radius:4px; cursor:pointer; width:100%;
  }}
  #heatmap-toggle:hover {{ background:#374151; }}

  /* ── Geocoder ── */
  .mapboxgl-ctrl-top-left {{ top:52px !important; }}
  .mapboxgl-ctrl-geocoder {{ min-width:260px; font-size:0.82rem; }}

  /* ── Popups ── */
  .mapboxgl-popup-content {{
    background:#1f2937 !important; color:#e5e7eb !important;
    border:1px solid #374151; border-radius:6px; max-width:300px;
    font-size:0.78rem; line-height:1.5;
  }}
  .mapboxgl-popup-tip {{ border-top-color:#374151 !important; border-bottom-color:#374151 !important; }}
  .pu-title {{ font-weight:600; margin-bottom:4px; color:#f9fafb; font-size:0.82rem; }}
  .pu-row   {{ display:flex; gap:4px; margin-bottom:1px; }}
  .pu-key   {{ color:#9ca3af; min-width:90px; flex-shrink:0; }}
  .pu-val   {{ color:#e5e7eb; word-break:break-word; }}
  .pu-tag   {{
    display:inline-block; margin-top:6px; padding:2px 8px;
    background:#7c2d12; color:#fb923c; border-radius:9999px;
    font-size:0.7rem; font-weight:600;
  }}
  .pu-risk-tag {{
    display:inline-block; margin-top:6px; padding:2px 8px;
    background:#78350f; color:#fde047; border-radius:9999px;
    font-size:0.7rem; font-weight:600;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>{title}</h1>
  <span class="badge b-red">ACLED {n_acled:,} events</span>
  <span class="badge b-orange">Road-risk {n_road_risk:,}</span>
  <span class="badge b-yellow">&#9888; Infra incidents {n_infra:,}</span>
  <span class="badge b-amber">OSM roads {n_roads:,}</span>
  <span class="badge b-green">NDVI {n_ndvi:,} cells (mock)</span>
</div>

<div id="map"></div>

<!-- Slider -->
<div id="slider-panel">
  <h3>ACLED Time Filter</h3>
  <div id="date-range-label"></div>
  <input type="range" id="date-slider" min="0" step="1" value="9999"/>
  <div id="slider-info">Showing all {n_acled:,} events</div>
  <button id="heatmap-toggle">Switch to point view</button>
</div>

<!-- Legend -->
<div id="legend">
  <h3>Legend</h3>

  <div class="leg-section">
    <div class="leg-section-title">Conflict</div>

    <div class="leg-item" data-layer="acled-heat" id="li-heat">
      <div class="leg-swatch" style="background:linear-gradient(to right,rgba(68,1,84,0.8),rgba(59,82,139,0.9),rgba(33,145,140,1),rgba(94,201,98,1),rgba(253,231,37,1));width:28px;height:10px;border-radius:2px;"></div>
      <span class="leg-label">ACLED heatmap</span>
    </div>
    <div class="leg-ramp">
      <div class="leg-ramp-bar" style="background:linear-gradient(to right,rgba(68,1,84,0.8),rgba(59,82,139,0.9),rgba(33,145,140,1),rgba(94,201,98,1),rgba(253,231,37,1));"></div>
      <div class="leg-ramp-ends"><span>low</span><span>high</span></div>
      <div class="leg-ramp-note">density of incidents (viridis)</div>
    </div>

    <div class="leg-item" data-layer="acled-points" id="li-pts" style="display:none">
      <div class="leg-swatch circle" style="background:#ef4444;"></div>
      <span class="leg-label">ACLED events (points)</span>
    </div>

    <div class="leg-item" data-layer="road-risk-heat" id="li-risk">
      <div class="leg-swatch" style="background:linear-gradient(to right,rgba(255,200,0,0.7),rgba(255,140,0,0.85),rgba(255,69,0,0.95),rgba(180,0,0,1));width:28px;height:10px;border-radius:2px;"></div>
      <span class="leg-label">Road disruption risk</span>
    </div>
    <div class="leg-ramp">
      <div class="leg-ramp-bar" style="background:linear-gradient(to right,rgba(255,200,0,0.7),rgba(255,140,0,0.85),rgba(255,69,0,0.95),rgba(180,0,0,1));"></div>
      <div class="leg-ramp-ends"><span>lower</span><span>higher</span></div>
      <div class="leg-ramp-note">conflict density near major roads</div>
    </div>

    <div class="leg-item" id="li-infra">
      <div class="leg-swatch circle" style="background:#fbbf24; border:2px solid #7c2d12;"></div>
      <span class="leg-label">&#9888; Infrastructure incident</span>
    </div>
  </div>

  <div class="leg-section">
    <div class="leg-section-title">Infrastructure</div>
    <div class="leg-item" data-layer="osm-roads-layer" id="li-roads">
      <div class="leg-swatch line" style="background:#f59e0b;width:28px;"></div>
      <span class="leg-label">OSM major roads</span>
    </div>
  </div>

  <div class="leg-section">
    <div class="leg-section-title">Environment (mock)</div>
    <div class="leg-item" data-layer="ndvi-fill" id="li-ndvi">
      <div class="leg-swatch" style="background:linear-gradient(to right,#8B4513,#C8A464,#D4E157,#66BB6A,#2E7D32);width:28px;height:10px;border-radius:2px;"></div>
      <span class="leg-label">NDVI (simulated)</span>
    </div>
    <div class="leg-ramp">
      <div class="leg-ramp-bar" style="background:linear-gradient(to right,#8B4513,#C8A464,#D4E157,#66BB6A,#2E7D32);"></div>
      <div class="leg-ramp-ends"><span>bare soil</span><span>dense veg.</span></div>
      <div class="leg-ramp-note">synthetic — not real satellite data</div>
    </div>
  </div>
</div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const ACLED_DATA  = {acled_js};
const NDVI_DATA   = {ndvi_js};
const ROAD_RISK   = {road_risk_js};
const INFRA_DATA  = {infra_js};
const OSM_ROADS   = {osm_roads_js};
const ALL_DATES   = {dates_js};

// ── Map initialisation ────────────────────────────────────────────────────
mapboxgl.accessToken = "{token}";
const map = new mapboxgl.Map({{
  container: "map",
  style: "mapbox://styles/mapbox/dark-v11",
  center: [{cx}, {cy}],
  zoom: {zoom},
}});

map.addControl(new mapboxgl.NavigationControl(), "top-right");
map.addControl(new mapboxgl.ScaleControl({{unit:"metric"}}), "bottom-right");
map.addControl(new mapboxgl.FullscreenControl(), "top-right");
try {{
  map.addControl(new MapboxGeocoder({{
    accessToken: mapboxgl.accessToken,
    mapboxgl: mapboxgl,
    placeholder: "Search location\u2026",
    collapsed: false,
  }}), "top-left");
}} catch(e) {{ console.warn("Geocoder unavailable:", e); }}

// ── State ─────────────────────────────────────────────────────────────────
let isHeatmap = true;

function sliceByDate(fc, cutoff) {{
  if (!cutoff) return fc;
  return {{type:"FeatureCollection", features: fc.features.filter(f =>
    (f.properties.event_date || "").slice(0,10) <= cutoff
  )}};
}}

function makePopup(props) {{
  const DISPLAY_KEYS = ["event_type","sub_event_type","event_date","actor1","fatalities","location"];
  const rows = DISPLAY_KEYS
    .filter(k => props[k] != null && props[k] !== "")
    .map(k => `<div class="pu-row"><span class="pu-key">${{k.replace(/_/g," ")}}</span><span class="pu-val">${{props[k]}}</span></div>`)
    .join("");
  const notes = props.notes
    ? `<div class="pu-row" style="margin-top:4px"><span class="pu-key">notes</span><span class="pu-val" style="font-size:0.72rem;color:#9ca3af;">${{String(props.notes).slice(0,220)}}${{props.notes.length>220?"…":""}}</span></div>`
    : "";
  const riskTag = props.road_risk !== undefined
    ? `<div class="pu-risk-tag">Road risk score: ${{props.road_risk}}</div>` : "";
  const infraTag = props._infra_incident
    ? `<div class="pu-tag">&#9888; Infrastructure Attack / Incident</div>` : "";
  return `<div class="pu-title">${{props.event_type || props.name || "Feature"}}</div>${{rows}}${{notes}}${{riskTag}}${{infraTag}}`;
}}

// ── Layer setup on load ───────────────────────────────────────────────────
map.on("load", () => {{

  // ── NDVI fill (bottom layer) ───────────────────────────────────────────
  map.addSource("ndvi-src", {{type:"geojson", data: NDVI_DATA}});
  map.addLayer({{
    id: "ndvi-fill",
    type: "fill",
    source: "ndvi-src",
    paint: {{
      "fill-color": [
        "interpolate", ["linear"], ["get", "ndvi"],
        0.02, "#8B4513",
        0.10, "#C8A464",
        0.20, "#D4E157",
        0.35, "#66BB6A",
        0.55, "#2E7D32",
        0.82, "#1B5E20"
      ],
      "fill-opacity": 0.40,
      "fill-outline-color": "rgba(0,0,0,0)",
    }},
  }});

  // ── ACLED heatmap (viridis) ────────────────────────────────────────────
  map.addSource("acled", {{type:"geojson", data: ACLED_DATA}});
  map.addLayer({{
    id: "acled-heat",
    type: "heatmap",
    source: "acled",
    paint: {{
      "heatmap-weight":    ["interpolate",["linear"],["coalesce",["to-number",["get","fatalities"]],1], 0,0.5, 50,1],
      "heatmap-intensity": ["interpolate",["linear"],["zoom"], 3,1.5, 6,2.5, 9,3.5],
      "heatmap-radius":    ["interpolate",["linear"],["zoom"], 3,10, 6,20, 9,35],
      "heatmap-opacity": 0.75,
      "heatmap-color": [
        "interpolate",["linear"],["heatmap-density"],
        0,    "rgba(0,0,0,0)",
        0.01, "rgba(68,1,84,0.8)",
        0.25, "rgba(59,82,139,0.9)",
        0.5,  "rgba(33,145,140,1.0)",
        0.75, "rgba(94,201,98,1.0)",
        1.0,  "rgba(253,231,37,1.0)"
      ],
    }},
  }});

  // ACLED point layer (hidden by default)
  map.addLayer({{
    id: "acled-points",
    type: "circle",
    source: "acled",
    layout: {{"visibility":"none"}},
    paint: {{
      "circle-color": "#ef4444", "circle-radius": 5,
      "circle-opacity": 0.8, "circle-stroke-width": 1,
      "circle-stroke-color": "#fff",
    }},
  }});

  // ── Road disruption risk heatmap (orange / red) ────────────────────────
  map.addSource("road-risk-src", {{type:"geojson", data: ROAD_RISK}});
  map.addLayer({{
    id: "road-risk-heat",
    type: "heatmap",
    source: "road-risk-src",
    paint: {{
      "heatmap-weight":    ["get", "road_risk"],
      "heatmap-intensity": ["interpolate",["linear"],["zoom"], 3,1.0, 6,2.0, 9,3.5],
      "heatmap-radius":    ["interpolate",["linear"],["zoom"], 3,15, 6,28, 9,45],
      "heatmap-opacity": 0.68,
      "heatmap-color": [
        "interpolate",["linear"],["heatmap-density"],
        0,    "rgba(0,0,0,0)",
        0.01, "rgba(255,200,0,0.65)",
        0.30, "rgba(255,140,0,0.85)",
        0.60, "rgba(255,69,0,0.95)",
        1.0,  "rgba(180,0,0,1.0)"
      ],
    }},
  }});

  // ── OSM major roads ────────────────────────────────────────────────────
  map.addSource("osm-roads-src", {{type:"geojson", data: OSM_ROADS}});
  map.addLayer({{
    id: "osm-roads-layer",
    type: "line",
    source: "osm-roads-src",
    paint: {{
      "line-color": "#f59e0b",
      "line-width": 4,
      "line-opacity": 0.85,
      "line-dasharray": [4, 2],
    }},
  }});

  // ── Infrastructure incident markers ────────────────────────────────────
  map.addSource("infra-src", {{type:"geojson", data: INFRA_DATA}});

  // Outer glow
  map.addLayer({{
    id: "infra-glow",
    type: "circle",
    source: "infra-src",
    paint: {{
      "circle-color": "#fbbf24",
      "circle-radius": 13,
      "circle-opacity": 0.18,
      "circle-blur": 1,
    }},
  }});

  // Core marker
  map.addLayer({{
    id: "infra-circles",
    type: "circle",
    source: "infra-src",
    paint: {{
      "circle-color": "#fbbf24",
      "circle-radius": 6,
      "circle-opacity": 0.95,
      "circle-stroke-color": "#7c2d12",
      "circle-stroke-width": 2,
    }},
  }});

  // Text label (visible at zoom >= 7)
  map.addLayer({{
    id: "infra-labels",
    type: "symbol",
    source: "infra-src",
    minzoom: 7,
    layout: {{
      "text-field": "\u26a0 Infrastructure Attack\u202f/\u202fIncident",
      "text-size": 9,
      "text-anchor": "left",
      "text-offset": [1.1, 0],
      "text-allow-overlap": false,
      "text-ignore-placement": false,
    }},
    paint: {{
      "text-color": "#fbbf24",
      "text-halo-color": "#000",
      "text-halo-width": 1.2,
    }},
  }});

  // ── ACLED hover popup ──────────────────────────────────────────────────
  const acledPopup = new mapboxgl.Popup({{closeButton:false, closeOnClick:false}});
  map.on("mouseenter","acled-points", e => {{
    map.getCanvas().style.cursor = "pointer";
    acledPopup.setLngLat(e.lngLat).setHTML(makePopup(e.features[0].properties)).addTo(map);
  }});
  map.on("mouseleave","acled-points", () => {{
    map.getCanvas().style.cursor = "";
    acledPopup.remove();
  }});

  // ── Click popups: infra incidents + roads ─────────────────────────────
  ["infra-circles","osm-roads-layer"].forEach(lid => {{
    map.on("click", lid, e => {{
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(makePopup(e.features[0].properties))
        .addTo(map);
    }});
    map.on("mouseenter", lid, () => {{ map.getCanvas().style.cursor = "pointer"; }});
    map.on("mouseleave", lid, () => {{ map.getCanvas().style.cursor = ""; }});
  }});

  // ── Legend toggles ────────────────────────────────────────────────────
  document.querySelectorAll(".leg-item[data-layer]").forEach(el => {{
    el.addEventListener("click", () => {{
      const lid = el.dataset.layer;
      if (!map.getLayer(lid)) return;
      const vis = map.getLayoutProperty(lid, "visibility");
      const next = (!vis || vis === "visible") ? "none" : "visible";
      map.setLayoutProperty(lid, "visibility", next);
      el.classList.toggle("hidden", next === "none");
    }});
  }});

  // Infrastructure: toggle circles + glow + labels together
  document.getElementById("li-infra").addEventListener("click", () => {{
    const el = document.getElementById("li-infra");
    ["infra-circles","infra-glow","infra-labels"].forEach(lid => {{
      if (!map.getLayer(lid)) return;
      const vis = map.getLayoutProperty(lid, "visibility");
      map.setLayoutProperty(lid, "visibility", (!vis || vis === "visible") ? "none" : "visible");
    }});
    el.classList.toggle("hidden");
  }});

  // ── Heatmap / point toggle ────────────────────────────────────────────
  document.getElementById("heatmap-toggle").addEventListener("click", () => {{
    isHeatmap = !isHeatmap;
    map.setLayoutProperty("acled-heat",   "visibility", isHeatmap ? "visible" : "none");
    map.setLayoutProperty("acled-points", "visibility", isHeatmap ? "none" : "visible");
    document.getElementById("heatmap-toggle").textContent =
      isHeatmap ? "Switch to point view" : "Switch to heatmap view";
    document.getElementById("li-heat").style.display = isHeatmap ? "" : "none";
    document.getElementById("li-pts").style.display  = isHeatmap ? "none" : "";
  }});

  // ── Time slider ───────────────────────────────────────────────────────
  const slider = document.getElementById("date-slider");
  const info   = document.getElementById("slider-info");
  const label  = document.getElementById("date-range-label");

  const FIRST = ALL_DATES.length > 0 ? ALL_DATES[0]                    : "";
  const LAST  = ALL_DATES.length > 0 ? ALL_DATES[ALL_DATES.length - 1] : "";
  label.textContent = `${{FIRST}} \u2192 ${{LAST}}`;

  if (ALL_DATES.length > 0) {{
    slider.max   = ALL_DATES.length;
    slider.value = ALL_DATES.length;
  }}

  slider.addEventListener("input", () => {{
    const idx = parseInt(slider.value, 10);
    if (idx >= ALL_DATES.length) {{
      map.getSource("acled").setData(ACLED_DATA);
      map.getSource("road-risk-src").setData(ROAD_RISK);
      map.getSource("infra-src").setData(INFRA_DATA);
      info.textContent  = `Showing all {n_acled:,} events`;
      label.textContent = `${{FIRST}} \u2192 ${{LAST}}`;
    }} else {{
      const cutoff = ALL_DATES[idx];
      const fa = sliceByDate(ACLED_DATA, cutoff);
      const fr = sliceByDate(ROAD_RISK,  cutoff);
      const fi = sliceByDate(INFRA_DATA, cutoff);
      map.getSource("acled").setData(fa);
      map.getSource("road-risk-src").setData(fr);
      map.getSource("infra-src").setData(fi);
      info.textContent  = `Up to ${{cutoff}}: ${{fa.features.length}} events`;
      label.textContent = `${{FIRST}} \u2192 ${{cutoff}}`;
    }}
  }});

}});  // end map.on("load")
</script>
</body>
</html>"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Verify input files
    for p in (ACLED_PATH, OSM_ROADS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input missing: {p}\n"
                "Run `python3 run_pipeline.py --skip-imagery --skip-yolo` first."
            )

    logger.info("Loading pipeline output data…")
    acled_fc = _load(ACLED_PATH)
    roads_fc = _load(OSM_ROADS_PATH)

    n_acled = len(acled_fc.get("features", []))
    n_roads = len(roads_fc.get("features", []))
    logger.info(f"  ACLED events:  {n_acled}")
    logger.info(f"  OSM roads:     {n_roads}")

    # Derived layers
    ndvi_fc      = generate_ndvi_grid()
    road_risk_fc = compute_road_risk_events(acled_fc, roads_fc)
    infra_fc     = filter_infrastructure_incidents(acled_fc)

    # Extract sorted unique event dates for the time slider
    dates_set = set()
    for f in acled_fc.get("features", []):
        d = f.get("properties", {}).get("event_date")
        if d:
            dates_set.add(str(d)[:10])
    sorted_dates = sorted(dates_set)

    # Build and write HTML
    html = build_html(
        acled_js     = _safe_json(acled_fc),
        ndvi_js      = _safe_json(ndvi_fc),
        road_risk_js = _safe_json(road_risk_fc),
        infra_js     = _safe_json(infra_fc),
        osm_roads_js = _safe_json(roads_fc),
        dates_js     = _safe_json(sorted_dates),
        n_acled      = n_acled,
        n_road_risk  = len(road_risk_fc.get("features", [])),
        n_infra      = len(infra_fc.get("features", [])),
        n_roads      = n_roads,
        n_ndvi       = len(ndvi_fc.get("features", [])),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_kb = OUTPUT_PATH.stat().st_size // 1024
    logger.info("─" * 50)
    logger.info(f"Enhanced map saved → {OUTPUT_PATH}  ({size_kb} KB)")
    logger.info(f"  ACLED events:          {n_acled:,}")
    logger.info(f"  Road-risk events:      {len(road_risk_fc.get('features', [])):,}")
    logger.info(f"  Infrastructure events: {len(infra_fc.get('features', [])):,}")
    logger.info(f"  OSM road features:     {n_roads:,}")
    logger.info(f"  NDVI grid cells:       {len(ndvi_fc.get('features', [])):,}")
    logger.info("─" * 50)


if __name__ == "__main__":
    main()
