"""
ACLED + WFP logistics fusion — Mapbox GL JS web application generator.

Produces a self-contained HTML file that visualises:

  1. ACLED conflict events as a **heatmap** (intensity ∝ fatalities / event count)
     with a **time slider** to filter events by date.
  2. WFP logistics infrastructure as interactive **vector overlays**:
       - Roads        → line layer (blue)
       - Airports     → circle points (red)
       - Border crossings → circle points (orange)
       - Storage / ports  → circle points (purple)
  3. An interactive **legend** panel with layer toggles.
  4. **Pop-ups** on ACLED points (hover) and WFP features (click) showing
     key attribute values.
  5. The ACLED heatmap can be toggled to a **point view** for individual
     event inspection.

All GeoJSON data is embedded directly in the HTML as JavaScript variables,
so the output is a fully standalone file that works offline or can be served
from any static host.

Usage:
    from water_osm_pipeline.fusion import generate_mapbox_app, MapboxFusionConfig

    cfg = MapboxFusionConfig(mapbox_token="pk.xxx…")
    html_path = generate_mapbox_app(
        acled_geojson   = acled_fc,
        wfp_layers      = {"roads": roads_fc, "airports": airports_fc, ...},
        output_path     = Path("output/fusion/map.html"),
        config          = cfg,
        title           = "Sudan: Conflict & Humanitarian Logistics",
        start_date      = "2023-10-01",
        end_date        = "2024-02-01",
    )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MapboxFusionConfig:
    """
    Configuration for the Mapbox GL JS fusion web app.

    Attributes:
        mapbox_token:       Mapbox public token (pk.xxx…).  Read from
                            MAPBOX_TOKEN env var if not provided.
        initial_zoom:       Starting map zoom level.
        initial_center:     Starting map centre [longitude, latitude].
        acled_heatmap_radius: Pixel radius for ACLED heatmap kernels.
        acled_heatmap_opacity: Opacity of the ACLED heatmap layer (0–1).
        wfp_road_color:     Hex colour for WFP road lines.
        wfp_airport_color:  Hex colour for airport markers.
        wfp_crossing_color: Hex colour for border crossing markers.
        wfp_storage_color:  Hex colour for storage/warehouse markers.
        map_style:          Mapbox style URL.
    """
    mapbox_token: Optional[str] = None
    initial_zoom: float = 7.0
    initial_center: Tuple[float, float] = (35.0, 15.0)  # [lon, lat]
    acled_heatmap_radius: int = 25
    acled_heatmap_opacity: float = 0.75
    wfp_road_color: str = "#3b82f6"       # blue-500
    wfp_airport_color: str = "#ef4444"    # red-500
    wfp_crossing_color: str = "#f97316"   # orange-500
    wfp_storage_color: str = "#a855f7"    # purple-500
    wfp_osm_road_color: str = "#f59e0b"  # amber-500
    map_style: str = "mapbox://styles/mapbox/dark-v11"

    def __post_init__(self):
        if self.mapbox_token is None:
            self.mapbox_token = os.environ.get("MAPBOX_TOKEN", "")
        if not self.mapbox_token:
            logger.warning(
                "MAPBOX_TOKEN not set — the map will not render until a "
                "token is provided.  Set MAPBOX_TOKEN in .env or pass it "
                "via MapboxFusionConfig(mapbox_token='pk.xxx…')"
            )


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

def _safe_json(obj) -> str:
    """Serialise *obj* to compact JSON, safe for embedding in a <script> tag."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _build_html(
    acled_geojson: Dict,
    wfp_layers: Dict[str, Dict],
    config: MapboxFusionConfig,
    title: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Render the full HTML string for the Mapbox GL JS web application.

    Embeds ACLED and WFP GeoJSON as JavaScript variables so the file is
    fully standalone.
    """
    token  = config.mapbox_token or ""
    cx, cy = config.initial_center

    # Embed GeoJSON data as JS vars
    _empty_fc = {"type": "FeatureCollection", "features": []}
    acled_js    = _safe_json(acled_geojson)
    roads_js    = _safe_json(wfp_layers.get("roads",     _empty_fc))
    airports_js = _safe_json(wfp_layers.get("airports",  _empty_fc))
    cross_js    = _safe_json(wfp_layers.get("crossings", _empty_fc))
    storage_js  = _safe_json(wfp_layers.get("storage",   _empty_fc))
    osm_roads_js = _safe_json(wfp_layers.get("osm_roads", _empty_fc))

    # Extract unique dates from ACLED for time slider
    dates_set: set = set()
    for feat in acled_geojson.get("features", []):
        d = feat.get("properties", {}).get("event_date")
        if d:
            dates_set.add(str(d)[:10])
    sorted_dates = sorted(dates_set)
    dates_js = _safe_json(sorted_dates)

    # Summary stats for header
    n_acled    = len(acled_geojson.get("features", []))
    n_roads    = len(wfp_layers.get("roads",     {}).get("features", []))
    n_air      = len(wfp_layers.get("airports",  {}).get("features", []))
    n_cross    = len(wfp_layers.get("crossings", {}).get("features", []))
    n_storage  = len(wfp_layers.get("storage",   {}).get("features", []))
    n_osm_rds  = len(wfp_layers.get("osm_roads", {}).get("features", []))

    html = f"""<!DOCTYPE html>
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
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #111; color: #e5e7eb; }}
  #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}

  /* ── Header ── */
  #header {{
    position: absolute; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(17,17,17,0.92); padding: 10px 16px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #374151;
  }}
  #header h1 {{ font-size: 1rem; font-weight: 600; color: #f9fafb; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 0.72rem; font-weight: 500;
  }}
  .badge-red    {{ background: #991b1b; color: #fca5a5; }}
  .badge-blue   {{ background: #1e3a8a; color: #93c5fd; }}
  .badge-orange {{ background: #92400e; color: #fcd34d; }}
  .badge-purple {{ background: #581c87; color: #d8b4fe; }}
  .badge-amber  {{ background: #78350f; color: #fde68a; }}

  /* ── Legend ── */
  #legend {{
    position: absolute; bottom: 36px; right: 10px; z-index: 10;
    background: rgba(17,17,17,0.92); border: 1px solid #374151;
    border-radius: 8px; padding: 12px; min-width: 200px;
  }}
  #legend h3 {{ font-size: 0.8rem; font-weight: 600; margin-bottom: 8px; color: #9ca3af; text-transform: uppercase; }}
  .legend-item {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px; cursor: pointer; user-select: none;
  }}
  .legend-item:hover .legend-label {{ color: #f9fafb; }}
  .legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0;
  }}
  .legend-swatch.circle {{ border-radius: 50%; }}
  .legend-swatch.line   {{ height: 4px; border-radius: 2px; }}
  .legend-label {{ font-size: 0.78rem; color: #d1d5db; }}
  .legend-item.hidden .legend-label {{ opacity: 0.4; }}

  /* ── Time slider ── */
  #slider-panel {{
    position: absolute; bottom: 36px; left: 10px; z-index: 10;
    background: rgba(17,17,17,0.92); border: 1px solid #374151;
    border-radius: 8px; padding: 12px; min-width: 280px;
  }}
  #slider-panel h3 {{ font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }}
  #date-range-label {{ font-size: 0.8rem; color: #e5e7eb; margin-bottom: 6px; }}
  #date-slider {{ width: 100%; accent-color: #ef4444; }}
  #slider-info {{ font-size: 0.72rem; color: #9ca3af; margin-top: 4px; }}
  #heatmap-toggle {{
    margin-top: 10px; font-size: 0.78rem;
    background: #1f2937; border: 1px solid #374151; color: #d1d5db;
    padding: 4px 10px; border-radius: 4px; cursor: pointer; width: 100%;
  }}
  #heatmap-toggle:hover {{ background: #374151; }}

  /* ── Geocoder — push below header ── */
  .mapboxgl-ctrl-top-left {{ top: 54px !important; }}
  .mapboxgl-ctrl-geocoder {{ min-width: 260px; font-size: 0.82rem; }}

  /* ── Pop-up custom style ── */
  .mapboxgl-popup-content {{
    background: #1f2937 !important; color: #e5e7eb !important;
    border: 1px solid #374151; border-radius: 6px; max-width: 280px;
    font-size: 0.78rem; line-height: 1.5;
  }}
  .mapboxgl-popup-tip {{ border-top-color: #374151 !important; border-bottom-color: #374151 !important; }}
  .popup-title {{ font-weight: 600; margin-bottom: 4px; color: #f9fafb; }}
  .popup-row   {{ display: flex; gap: 4px; }}
  .popup-key   {{ color: #9ca3af; min-width: 90px; }}
  .popup-val   {{ color: #e5e7eb; word-break: break-word; }}
</style>
</head>
<body>

<div id="header">
  <h1>{title}</h1>
  <span class="badge badge-red">ACLED {n_acled:,} events</span>
  <span class="badge badge-blue">WFP Roads {n_roads:,}</span>
  <span class="badge badge-amber">OSM Roads {n_osm_rds:,}</span>
  <span class="badge badge-red">Airports {n_air}</span>
  <span class="badge badge-orange">Crossings {n_cross}</span>
  <span class="badge badge-purple">Storage {n_storage}</span>
  <span style="font-size:0.72rem;color:#6b7280;margin-left:auto;">{start_date} → {end_date}</span>
</div>

<div id="map"></div>

<!-- Time slider -->
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
  <div class="legend-item" data-layer="acled-heat" id="li-heat">
    <div class="legend-swatch" style="background:linear-gradient(to right,rgba(68,1,84,0.8),rgba(59,82,139,0.9),rgba(33,145,140,1),rgba(94,201,98,1),rgba(253,231,37,1));width:28px;height:10px;border-radius:2px;"></div>
    <span class="legend-label">ACLED heatmap</span>
  </div>
  <div style="margin-left:22px; margin-bottom:6px;">
    <div style="background:linear-gradient(to right,rgba(68,1,84,0.8),rgba(59,82,139,0.9),rgba(33,145,140,1),rgba(94,201,98,1),rgba(253,231,37,1));height:5px;border-radius:2px;width:110px;"></div>
    <div style="display:flex;justify-content:space-between;width:110px;font-size:0.63rem;color:#6b7280;margin-top:1px;">
      <span>low</span><span>high</span>
    </div>
    <div style="font-size:0.65rem;color:#6b7280;margin-top:1px;">density of incidents</div>
  </div>
  <div class="legend-item" data-layer="acled-points" id="li-pts" style="display:none">
    <div class="legend-swatch circle" style="background:#ef4444;"></div>
    <span class="legend-label">ACLED events</span>
  </div>
  <div class="legend-item" data-layer="wfp-roads" id="li-roads">
    <div class="legend-swatch line" style="background:{config.wfp_road_color};width:28px;"></div>
    <span class="legend-label">WFP Roads</span>
  </div>
  <div class="legend-item" data-layer="wfp-osm-roads" id="li-osm-roads">
    <div class="legend-swatch line" style="background:{config.wfp_osm_road_color};width:28px;"></div>
    <span class="legend-label">OSM major roads</span>
  </div>
  <div class="legend-item" data-layer="wfp-airports" id="li-air">
    <div class="legend-swatch circle" style="background:{config.wfp_airport_color};"></div>
    <span class="legend-label">Airports</span>
  </div>
  <div class="legend-item" data-layer="wfp-crossings" id="li-cross">
    <div class="legend-swatch circle" style="background:{config.wfp_crossing_color};"></div>
    <span class="legend-label">Border crossings</span>
  </div>
  <div class="legend-item" data-layer="wfp-storage" id="li-stor">
    <div class="legend-swatch circle" style="background:{config.wfp_storage_color};"></div>
    <span class="legend-label">Ports / Storage</span>
  </div>
</div>

<script>
// ── Embedded GeoJSON data ──────────────────────────────────────────────────
const ACLED_DATA    = {acled_js};
const WFP_ROADS     = {roads_js};
const WFP_AIRPORTS  = {airports_js};
const WFP_CROSSINGS = {cross_js};
const WFP_STORAGE   = {storage_js};
const WFP_OSM_ROADS = {osm_roads_js};
const ALL_DATES     = {dates_js};

// ── Map initialisation ────────────────────────────────────────────────────
mapboxgl.accessToken = "{token}";
const map = new mapboxgl.Map({{
  container: "map",
  style: "{config.map_style}",
  center: [{cx}, {cy}],
  zoom: {config.initial_zoom},
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
let activeFilter = null;  // date string or null

// ── Helper: filter ACLED by date ─────────────────────────────────────────
function filteredAcled(cutoff) {{
  if (!cutoff) return ACLED_DATA;
  const feats = ACLED_DATA.features.filter(f => {{
    const d = (f.properties.event_date || "").slice(0, 10);
    return d && d <= cutoff;
  }});
  return {{type:"FeatureCollection", features: feats}};
}}

// ── Helper: build popup HTML ──────────────────────────────────────────────
function makePopup(props, keys) {{
  const rows = keys
    .filter(k => props[k] != null && props[k] !== "")
    .map(k => `<div class="popup-row">
                 <span class="popup-key">${{k.replace(/_/g," ")}}</span>
                 <span class="popup-val">${{props[k]}}</span>
               </div>`)
    .join("");
  return `<div class="popup-title">${{props.event_type || props.name || props.osm_type || "Feature"}}</div>${{rows}}`;
}}

// ── On map load ───────────────────────────────────────────────────────────
map.on("load", () => {{

  // ── ACLED source ──────────────────────────────────────────────────────
  map.addSource("acled", {{type:"geojson", data: ACLED_DATA}});

  // Heatmap layer
  map.addLayer({{
    id: "acled-heat",
    type: "heatmap",
    source: "acled",
    paint: {{
      "heatmap-weight":    ["interpolate",["linear"],["coalesce",["to-number",["get","fatalities"]],1], 0,0.5, 50,1],
      "heatmap-intensity": ["interpolate",["linear"],["zoom"], 3,1.5, 6,2.5, 9,3.5],
      "heatmap-radius":    ["interpolate",["linear"],["zoom"], 3,10, 6,20, 9,35],
      "heatmap-opacity":   {config.acled_heatmap_opacity},
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

  // Point layer (hidden by default)
  map.addLayer({{
    id: "acled-points",
    type: "circle",
    source: "acled",
    layout: {{"visibility":"none"}},
    paint: {{
      "circle-color":   "#ef4444",
      "circle-radius":  5,
      "circle-opacity": 0.8,
      "circle-stroke-width": 1,
      "circle-stroke-color": "#fff",
    }},
  }});

  // ── WFP layers ────────────────────────────────────────────────────────
  map.addSource("wfp-roads",     {{type:"geojson", data: WFP_ROADS}});
  map.addSource("wfp-airports",  {{type:"geojson", data: WFP_AIRPORTS}});
  map.addSource("wfp-crossings", {{type:"geojson", data: WFP_CROSSINGS}});
  map.addSource("wfp-storage",   {{type:"geojson", data: WFP_STORAGE}});
  map.addSource("wfp-osm-roads", {{type:"geojson", data: WFP_OSM_ROADS}});

  map.addLayer({{
    id: "wfp-roads", type: "line", source: "wfp-roads",
    paint: {{"line-color": "{config.wfp_road_color}", "line-width": 3, "line-opacity": 0.9}},
  }});
  map.addLayer({{
    id: "wfp-osm-roads", type: "line", source: "wfp-osm-roads",
    paint: {{
      "line-color": "{config.wfp_osm_road_color}",
      "line-width": 4,
      "line-opacity": 0.85,
      "line-dasharray": [4, 2],
    }},
  }});

  function addWfpCircle(id, source, color) {{
    map.addLayer({{
      id, type:"circle", source,
      paint: {{
        "circle-color":  color,
        "circle-radius": 10,
        "circle-opacity": 0.9,
        "circle-stroke-width": 2,
        "circle-stroke-color": "#fff",
      }},
    }});
  }}

  map.addLayer({{
    id: "wfp-airports", type: "circle", source: "wfp-airports",
    paint: {{
      "circle-color":        "{config.wfp_airport_color}",
      "circle-radius":       4,
      "circle-opacity":      0.5,
      "circle-stroke-width": 1,
      "circle-stroke-color": "#fff",
      "circle-stroke-opacity": 0.4,
    }},
  }});
  addWfpCircle("wfp-crossings", "wfp-crossings", "{config.wfp_crossing_color}");
  addWfpCircle("wfp-storage",   "wfp-storage",   "{config.wfp_storage_color}");

  // ── ACLED hover popup ─────────────────────────────────────────────────
  const acledPopup = new mapboxgl.Popup({{closeButton:false, closeOnClick:false}});

  map.on("mouseenter","acled-points", e => {{
    map.getCanvas().style.cursor = "pointer";
    const p = e.features[0].properties;
    acledPopup
      .setLngLat(e.lngLat)
      .setHTML(makePopup(p, ["event_type","sub_event_type","event_date",
                             "actor1","actor2","fatalities","location","notes"]))
      .addTo(map);
  }});
  map.on("mouseleave","acled-points", () => {{
    map.getCanvas().style.cursor = "";
    acledPopup.remove();
  }});

  // ── WFP click popups ──────────────────────────────────────────────────
  ["wfp-airports","wfp-crossings","wfp-storage","wfp-roads","wfp-osm-roads"].forEach(lid => {{
    map.on("click", lid, e => {{
      const p = e.features[0].properties;
      const keys = Object.keys(p).filter(k => !["osm_id","osm_type"].includes(k));
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(makePopup(p, keys.slice(0, 10)))
        .addTo(map);
    }});
    map.on("mouseenter", lid, () => {{ map.getCanvas().style.cursor = "pointer"; }});
    map.on("mouseleave", lid, () => {{ map.getCanvas().style.cursor = ""; }});
  }});

  // ── Legend layer toggles ──────────────────────────────────────────────
  document.querySelectorAll(".legend-item[data-layer]").forEach(el => {{
    el.addEventListener("click", () => {{
      const layerId = el.dataset.layer;
      if (!map.getLayer(layerId)) return;
      const vis = map.getLayoutProperty(layerId, "visibility");
      const next = (!vis || vis === "visible") ? "none" : "visible";
      map.setLayoutProperty(layerId, "visibility", next);
      el.classList.toggle("hidden", next === "none");
    }});
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

  // Derive actual date bounds from embedded event data
  const FIRST_DATE = ALL_DATES.length > 0 ? ALL_DATES[0]                    : "{start_date}";
  const LAST_DATE  = ALL_DATES.length > 0 ? ALL_DATES[ALL_DATES.length - 1] : "{end_date}";

  // Initialise display
  label.textContent = `${{FIRST_DATE}} → ${{LAST_DATE}}`;

  if (ALL_DATES.length > 0) {{
    slider.max   = ALL_DATES.length;  // max index = show all
    slider.value = ALL_DATES.length;
  }}

  slider.addEventListener("input", () => {{
    const idx = parseInt(slider.value, 10);
    if (idx >= ALL_DATES.length) {{
      // Show all
      map.getSource("acled").setData(ACLED_DATA);
      info.textContent  = `Showing all {n_acled:,} events`;
      label.textContent = `${{FIRST_DATE}} → ${{LAST_DATE}}`;
    }} else {{
      const cutoff = ALL_DATES[idx];
      const fc = filteredAcled(cutoff);
      map.getSource("acled").setData(fc);
      info.textContent  = `Up to ${{cutoff}}: ${{fc.features.length}} events`;
      label.textContent = `${{FIRST_DATE}} → ${{cutoff}}`;
    }}
  }});

}});  // end map.on("load")
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_mapbox_app(
    acled_geojson: Dict,
    wfp_layers: Dict[str, Dict],
    output_path: Path,
    config: Optional[MapboxFusionConfig] = None,
    title: str = "Conflict & Humanitarian Logistics",
    start_date: str = "",
    end_date: str = "",
) -> Path:
    """
    Generate a self-contained Mapbox GL JS web application.

    Combines ACLED conflict heatmap and WFP logistics vector overlays into
    an interactive HTML file.

    Args:
        acled_geojson: GeoJSON FeatureCollection of ACLED events.
        wfp_layers:    Dict mapping layer name → GeoJSON FeatureCollection.
                       Expected keys: ``"roads"``, ``"airports"``,
                       ``"crossings"``, ``"storage"``.
        output_path:   Destination path for the HTML file.
        config:        Optional :class:`MapboxFusionConfig`.
        title:         Page title and header label.
        start_date:    Analysis start date (``YYYY-MM-DD``), for display.
        end_date:      Analysis end date (``YYYY-MM-DD``), for display.

    Returns:
        :class:`~pathlib.Path` to the written HTML file.
    """
    cfg = config or MapboxFusionConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_acled = len(acled_geojson.get("features", []))
    n_wfp   = sum(
        len(fc.get("features", [])) for fc in wfp_layers.values()
    )
    logger.info(
        f"Generating Mapbox app: {n_acled} ACLED events, "
        f"{n_wfp} WFP features → {output_path}"
    )

    html = _build_html(
        acled_geojson=acled_geojson,
        wfp_layers=wfp_layers,
        config=cfg,
        title=title,
        start_date=start_date,
        end_date=end_date,
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info(
        f"Mapbox web app saved → {output_path}  "
        f"({output_path.stat().st_size // 1024} KB)"
    )
    return output_path
