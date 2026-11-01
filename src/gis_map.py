# =============================================================
# gis_map.py  —  GIS-based fish density map
# =============================================================
# What this file does:
#   1. Takes fish detection results + user-provided GPS location
#   2. Maps detections to GPS coordinates within pond boundary
#   3. Generates interactive HTML map with:
#      - Pond boundary
#      - Fish detection points (colored by species)
#      - Density heat zones
#      - Clickable markers with species info
#
# Usage:
#   python src/gis_map.py --lat 23.4120 --lon 85.3094
#                         --location "Kanke Dam, Ranchi"
# =============================================================

import folium
import folium.plugins
import random
import json
import argparse
import os
import sys

# ── Configuration ─────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\ravib\ML_Fish_Classifier\models\gis"

# ── Species colors for map markers ────────────────────────────
SPECIES_COLORS = {
    "Catfish"        : "red",
    "Goby"           : "blue",
    "Grass Carp"     : "green",
    "Tilapia"        : "purple",
    "Indian Carp"    : "orange",
    "Glass Perchlet" : "darkred",
    "Gold Fish"      : "cadetblue",
    "Gourami"        : "darkblue",
    "Unknown"        : "gray",
}
DEFAULT_COLOR = "lightgray"

def random_point_in_pond(center, radius_deg=0.002):
    """
    Generate a random GPS point within the pond area.

    Args:
        center     : [lat, lon] of pond center
        radius_deg : spread radius in degrees

    Returns:
        [lat, lon]
    """
    lat = center[0] + random.uniform(-radius_deg, radius_deg)
    lon = center[1] + random.uniform(-radius_deg, radius_deg)
    return [lat, lon]

def generate_gis_map(species_count, total_fish, density,
                     water_safe=True, ph=7.0,
                     lat=23.4120, lon=85.3094,
                     location_name="Kanke Dam, Ranchi"):
    """
    Generate interactive GIS map with fish distribution.

    Args:
        species_count : dict {species: count}
        total_fish    : total fish detected
        density       : fish/m²
        water_safe    : bool
        ph            : predicted pH
        lat           : latitude of pond center
        lon           : longitude of pond center
        location_name : human-readable location name
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    pond_center = [lat, lon]

    # Calculate pond corners from center
    r = 0.003  # ~300m radius
    pond_corners = [
        [lat + r, lon - r],
        [lat + r, lon + r],
        [lat - r, lon + r],
        [lat - r, lon - r],
    ]

    # ── Create base map ───────────────────────────────────────
    m = folium.Map(
        location   = pond_center,
        zoom_start = 16,
        tiles      = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr       = "OpenStreetMap"
    )

    # ── Draw pond boundary ────────────────────────────────────
    folium.Polygon(
        locations    = pond_corners,
        color        = "blue",
        weight       = 3,
        fill         = True,
        fill_color   = "lightblue",
        fill_opacity = 0.2,
        popup        = f"Pond Boundary — {location_name}"
    ).add_to(m)

    # ── Add pond center marker ────────────────────────────────
    folium.Marker(
        location = pond_center,
        popup    = f"📍 {location_name}",
        tooltip  = location_name,
        icon     = folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    # ── Add fish detection markers ────────────────────────────
    heat_data = []

    for species, count in species_count.items():
        color = SPECIES_COLORS.get(species, DEFAULT_COLOR)

        for i in range(count):
            point = random_point_in_pond(pond_center)
            heat_data.append(point)

            folium.CircleMarker(
                location     = point,
                radius       = 6,
                color        = color,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.7,
                popup        = folium.Popup(
                    f"<b>Species:</b> {species}<br>"
                    f"<b>Location:</b> {point[0]:.4f}°N, {point[1]:.4f}°E",
                    max_width=200
                ),
                tooltip = species
            ).add_to(m)

    # ── Add heat map layer ────────────────────────────────────
    folium.plugins.HeatMap(
        heat_data,
        min_opacity = 0.4,
        radius      = 20,
        blur        = 15,
        name        = "Fish Density Heat Map"
    ).add_to(m)

    # ── Add info box ──────────────────────────────────────────
    safe_emoji = "✅" if water_safe else "❌"
    info_html = f"""
    <div style="position:fixed; bottom:30px; left:30px;
                background:white; padding:15px;
                border:2px solid grey; border-radius:8px;
                font-size:13px; z-index:1000;">
        <b>🐟 Fish Detection Report</b><br>
        Total fish : {total_fish}<br>
        Density    : {density:.4f} fish/m²<br>
        <br>
        <b>💧 Water Quality</b><br>
        pH         : {ph}<br>
        Safe       : {safe_emoji}<br>
        <br>
        <b>📍 Location</b><br>
        {location_name}<br>
        {lat}°N, {lon}°E
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # ── Add legend ────────────────────────────────────────────
    legend_html = """
    <div style='position:fixed; top:30px; right:30px;
                background:white; padding:10px;
                border:1px solid grey; border-radius:8px;
                font-size:12px; z-index:1000;'>
    <b>Species Legend</b><br>
    """
    for species, color in SPECIES_COLORS.items():
        if species in species_count:
            cnt = species_count.get(species, 0)
            legend_html += f"<span style='color:{color}'>●</span> {species}: {cnt}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Add layer control ─────────────────────────────────────
    folium.LayerControl().add_to(m)

    # ── Save map ──────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "fish_gis_map.html")
    m.save(output_path)
    print(f"\nGIS map saved → {output_path}")
    print(f"Serve with: python -m http.server 8080 in {OUTPUT_DIR}")
    print(f"Then open: http://localhost:8080/fish_gis_map.html")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--species_json", type=str,
        help="JSON string e.g. '{\"Catfish\":9,\"Goby\":8}'")
    parser.add_argument("--total",        type=int,   default=27)
    parser.add_argument("--density",      type=float, default=2.25)
    parser.add_argument("--water_safe",   type=int,   default=1)
    parser.add_argument("--ph",           type=float, default=7.59)
    parser.add_argument("--lat",          type=float, default=23.4120,
                        help="Latitude of pond/river location")
    parser.add_argument("--lon",          type=float, default=85.3094,
                        help="Longitude of pond/river location")
    parser.add_argument("--location",     type=str,
                        default="Kanke Dam, Ranchi",
                        help="Human-readable location name")
    args = parser.parse_args()

    if args.species_json:
        species_count = json.loads(args.species_json)
    else:
        species_count = {
            "Glass Perchlet": 9,
            "Gold Fish"     : 4,
            "Goby"          : 3,
            "Catfish"       : 3,
            "Unknown"       : 8
        }

    generate_gis_map(
        species_count = species_count,
        total_fish    = args.total,
        density       = args.density,
        water_safe    = bool(args.water_safe),
        ph            = args.ph,
        lat           = args.lat,
        lon           = args.lon,
        location_name = args.location
    )