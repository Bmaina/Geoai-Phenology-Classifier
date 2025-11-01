import pandas as pd
import numpy as np
import random
import geojson

# --- Global Configuration ---

FEATURE_NAMES = [
    'NDVI_Max', 'NDVI_Min', 'SOS_Date', 'EOS_Date', 'Peak_Length',
    'Amplitude', 'Temp_Avg_Annual', 'Precip_Total_Annual'
]

CLASS_LABELS = {
    0: "Grassland/Savanna", 
    1: "Bushland/Shrubland", 
    2: "Forest/Riverine"
}

# Define color mapping for the map legend 
LANDCOVER_COLORS = {
    "Grassland/Savanna": "#90ee90",  # Light Green
    "Bushland/Shrubland": "#daa520", # Goldenrod
    "Forest/Riverine": "#228b22"     # Forest Green
}

# --- GEE Asset Placeholders (REPLACE THESE) ---

# REQUIRED ACTION 1: Replace with your actual GEE Tile URL
CLASSIFIED_MAP_TILE_URL = 'https://tiles.stadiamaps.com/tiles/stamen_terrain_labels/{z}/{x}/{y}{r}.png'

# REQUIRED ACTION 2: Load your GeoJSON for AOI selection (e.g., your "NewSites.geojson" data)
def get_site_geojson():
    """Loads a mock GeoJSON for interactive site selection (simulating ground truth sites)."""
    mock_sites = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Riverine Transect Site A", "class_id": 2},
                "geometry": {"type": "Polygon", "coordinates": [[ [35.2, -1.5], [35.3, -1.5], [35.3, -1.6], [35.2, -1.6], [35.2, -1.5] ]]}
            },
            {
                "type": "Feature",
                "properties": {"name": "North Rangeland Monitoring Plot", "class_id": 0},
                "geometry": {"type": "Polygon", "coordinates": [[ [34.9, -1.2], [35.0, -1.2], [35.0, -1.3], [34.9, -1.3], [34.9, -1.2] ]]}
            },
             {
                "type": "Feature",
                "properties": {"name": "Shrubland Transition Zone", "class_id": 1},
                "geometry": {"type": "Polygon", "coordinates": [[ [35.5, -1.9], [35.6, -1.9], [35.6, -2.0], [35.5, -2.0], [35.5, -1.9] ]]}
            }
        ]
    }
    return geojson.dumps(mock_sites)

# --- Time Series Data (kept here as it's static) ---
def get_time_series_data():
    """Generates a mock time series for visualization."""
    dates = pd.date_range(start='2015-01-01', periods=10 * 23, freq='2W')
    time_series_data = pd.DataFrame({
        'Date': dates,
        # Simulate annual NDVI cycles with some noise and trend
        'NDVI': np.clip(np.sin(np.linspace(0, 10*np.pi, len(dates))) * 0.3 + 
                        np.linspace(0.5, 0.6, len(dates)) + # Small positive trend
                        np.random.normal(0, 0.05, len(dates)), # Noise
                        0.1, 0.9)
    })
    return time_series_data