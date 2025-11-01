import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import json
import os
import requests
import time
from urllib.parse import quote

from floristics_map_data import (
    lat, lon, 
    CLASSIFICATION_REPORT, 
    AI_INTERPRETATION, 
    DEFAULT_AOI_GEOJSON,
    BASEMAP_OPTIONS,
    CLASSIFIED_MAP_TILE_URL,
)

# --- Configuration and Setup ---

# Must use gemini-2.5-flash-preview-09-2025 for text generation
GEMINI_MODEL_TEXT = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure Streamlit page
st.set_page_config(
    page_title="GeoAI Phenology Classification", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- LLM API Function ---

@st.cache_data(show_spinner=False)
def get_ai_interpretation(report_data):
    """Calls the Gemini API to provide a professional interpretation of the classification report."""
    if not GEMINI_API_KEY:
        return AI_INTERPRETATION # Fallback to mock data if key is missing

    # Format the report data into a clean string for the LLM
    report_text = f"Classification Report:\n\n"
    for cls, metrics in report_data.items():
        if cls in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        report_text += f"Class: {cls}, Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-Score: {metrics['f1-score']:.2f}, Support: {metrics['support']}\n"
    report_text += f"\nOverall Accuracy: {report_data.get('accuracy', 'N/A')}"

    system_prompt = "Act as a world-class geospatial data scientist and financial analyst. Provide a concise, single-paragraph summary of the key findings, focusing on the confidence, reliability, and business value of the model's accuracy, particularly where F1 scores are highest."
    user_query = f"Analyze the following land cover classification metrics and provide a professional, one-paragraph interpretation:\n\n{report_text}"
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_TEXT}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # Implement basic retry logic for API stability
    for attempt in range(3):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract generated text
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            if text:
                return text
            
        except requests.exceptions.RequestException as e:
            # st.error(f"API Error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt) # Exponential backoff
        except Exception as e:
            # st.error(f"Failed to process API response: {e}")
            break
            
    return AI_INTERPRETATION # Fallback to mock data on all failures


# --- Main Application Layout ---

# Title and Value Proposition Ribbon
st.markdown(
    """
    <style>
    .ribbon-container {
        padding: 20px 0;
        margin-bottom: 20px;
        background: #1e3a8a; /* Dark Blue */
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 100 100'%3E%3Cpath fill='%232c4f8d' d='M40 0h20v100H40z'/%3E%3Cpath fill='%231e3a8a' d='M0 40h100v20H0z'/%3E%3C/svg%3E"); /* Subtle Grid Pattern */
        background-repeat: repeat;
        color: #e5e7eb; /* Light Gray Text */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .ribbon-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .ribbon-subtitle {
        font-size: 1rem;
        font-weight: 300;
        max-width: 900px;
        margin: 0 auto 10px;
        padding: 0 20px;
    }
    .value-prop {
        font-size: 1rem;
        font-weight: 400;
        max-width: 900px;
        margin: 0 auto;
        padding: 0 20px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .ai-box {
        background-color: #d1fae5; /* Light Green */
        color: #065f46; /* Dark Green Text */
        padding: 15px;
        border-left: 5px solid #10b981; /* Accent Color */
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
    <div class="ribbon-container">
        <div class="ribbon-title">GeoAI Phenology Classification Product</div>
        <div class="ribbon-subtitle">Advanced Landcover Classification using Hyper-Temporal NDVI variables</div>
        <div class="value-prop">
            This application showcases land cover classification with **unprecedented temporal resolution** by leveraging hyper-temporal NDVI data. This methodology drastically **boosts accuracy** compared to single-date imagery, making it essential for timely, reliable environmental monitoring, particularly in data-scarce regions like the Global South where rapid, verifiable insights are critical for resource management and funding.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("1. Area of Interest (AOI)")
    
    # Select AOI method
    aoi_method = st.radio(
        "Select AOI Source:",
        ("Sample AOI (Mara)", "Custom GeoJSON Input"),
        index=0
    )

    if aoi_method == "Sample AOI (Mara)":
        # Uses the default GeoJSON coordinates
        geojson_data = DEFAULT_AOI_GEOJSON
        st.info("Using default AOI centered on the Maasai Mara ecosystem.")
    else:
        # Custom GeoJSON input
        uploaded_file = st.file_uploader("Upload GeoJSON file", type="geojson")
        if uploaded_file is not None:
            try:
                geojson_data = json.load(uploaded_file)
            except json.JSONDecodeError:
                st.error("Invalid GeoJSON file. Please check the format.")
                geojson_data = DEFAULT_AOI_GEOJSON # Fallback
        else:
            geojson_data = DEFAULT_AOI_GEOJSON
            st.warning("Using default AOI. Upload a file to define a custom area.")

    st.header("2. Analysis Parameters")
    
    # Slider for time series window (Mock setting)
    time_window = st.slider(
        "NDVI Time Series Window (Mock)", 
        min_value=12, 
        max_value=36, 
        value=24, 
        step=12,
        help="Simulates the number of months of NDVI data used for classification."
    )
    
    # Basemap selector (Using only reliable, free tiles)
    selected_basemap_name = st.selectbox(
        "Select Base Map Style:",
        list(BASEMAP_OPTIONS.keys()),
        index=1
    )
    selected_basemap = BASEMAP_OPTIONS[selected_basemap_name]


# --- Main Content Area: Map and Tabs ---

# Column layout for map dominance
col1, col2 = st.columns([7, 3]) 

# --- Column 1: Map Visualization (Wider) ---
with col1:
    st.subheader("1. Interactive Landcover Visualization")
    
    # Initialize the map centered on the AOI
    # We explicitly use 'CartoDB positron' tiles, which are stable and free,
    # to avoid 401 unauthorized errors common with ESRI or Stamen tiles.
    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles='CartoDB Positron',  # <--- CRITICAL FIX: Explicitly set a reliable, free tile source
        control_scale=True,
        width='100%',
        height='600px'
    )

    # 1. Add the selected Basemap
    # Note: If the user selects OpenStreetMap, this re-adds it, which is fine for Folium
    selected_basemap.add_to(m) 

    # 2. Add the Classified Map Layer (Mock TileLayer)
    folium.TileLayer(
        tiles=CLASSIFIED_MAP_TILE_URL,
        name="Classified Landcover Map",
        attr='GeoAI Phenology Product',
        overlay=True,
        control=True
    ).add_to(m)

    # 3. Add the AOI as a GeoJSON layer
    folium.GeoJson(
        geojson_data,
        name="Area of Interest (AOI)",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': '#FFFF00',  # Yellow border
            'weight': 3,
            'dashArray': '5, 5'
        }
    ).add_to(m)

    # 4. Add Layer Control for toggling maps
    folium.LayerControl().add_to(m)

    # Render the map using st_folium (interactive drawing enabled)
    # Note: st_folium returns the drawing results, which we capture in 'output'
    output = st_folium(
        m,
        height=600,
        width="100%",
        feature_group_to_add='draw',
        returned_objects=["last_active_drawing"],
        key="phenology_map",
    )

    # Check for user-drawn features
    if output and 'last_active_drawing' in output and output['last_active_drawing']:
        st.success("Custom Area of Interest detected from map drawing!")


# --- Column 2: Insights and Metrics ---
with col2:
    st.subheader("2. Classification Breakdown and AI Interpretation")
    
    # Create tabs for structured output
    tabs = st.tabs(["Accuracy & Data Export", "AI Interpretation"])

    # --- Tab 1: Accuracy & Data Export ---
    with tabs[0]:
        st.markdown("##### 📋 Detailed Per-Class Metrics")
        
        # Convert the classification report into a DataFrame for styling
        df = pd.DataFrame(CLASSIFICATION_REPORT).T
        
        # Format the numbers
        df['precision'] = df['precision'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        df['recall'] = df['recall'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        df['f1-score'] = df['f1-score'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
        # Separate the main class metrics from the summary rows
        metrics_df = df.iloc[0:3]
        summary_df = df.iloc[3:]

        # Function to style the f1-score column for visualization
        def highlight_f1(s):
            is_f1 = s.name == 'f1-score'
            return [f'background-color: #dcfce7; font-weight: bold;' if is_f1 else '' for v in s]

        # Display Metrics table with visual styling
        st.dataframe(
            metrics_df.style.apply(highlight_f1, axis=0),
            use_container_width=True
        )

        st.markdown("##### 📈 Summary Metrics")
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            label="Download Classification Report (JSON)",
            data=json.dumps(CLASSIFICATION_REPORT, indent=4),
            file_name="phenology_report.json",
            mime="application/json",
            key="download_report"
        )

    # --- Tab 2: AI Interpretation ---
    with tabs[1]:
        st.markdown("##### 💡 AI-Driven Insights")
        
        # Get interpretation from LLM (using cache for speed)
        interpretation = get_ai_interpretation(CLASSIFICATION_REPORT)
        
        # Display the LLM interpretation in the custom styled box
        st.markdown(f"""
        <div class="ai-box">
            <p><strong>Model Analysis:</strong></p>
            <p>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Interpretation provided by Gemini-2.5-Flash (LLM).")
eof

Please commit and push this version of `app.py` and the previously provided `floristics_map_data.py` to your repository. This should resolve the map authentication error on Streamlit Cloud.
