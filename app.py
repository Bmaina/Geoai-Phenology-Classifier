import streamlit as st
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import folium
from streamlit_folium import folium_static, st_folium 
import geemap.foliumap as geemap 
import json
import matplotlib.pyplot as plt
from folium.plugins import Draw

# Import custom data and configuration
from floristics_map_data import (
    FEATURE_NAMES, CLASS_LABELS, LANDCOVER_COLORS, 
    CLASSIFIED_MAP_TILE_URL, get_site_geojson, get_time_series_data
)

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="GeoAI Phenology Product",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the value proposition banner (Ribbon with Thematic Background)
# Using a subtle SVG background for a technical/map aesthetic
st.markdown("""
<style>
    .ribbon-box {
        /* Dark Teal/Green background for high-tech look */
        background-color: #004D40; 
        color: white; 
        border-radius: 8px;
        padding: 20px 25px;
        margin-bottom: 25px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        
        /* Thematic Background: Subtle, abstract vector topography/grid */
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><defs><pattern id="p" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M-1 1h2v-2h-2zM0 0h10v10h-10zM1 9h8v-8h-8z" stroke-width="0.5" stroke="%233CB371" fill="none"/></pattern></defs><rect width="100" height="100" fill="url(%23p)" opacity="0.1"/></svg>');
        background-repeat: repeat;
        background-size: 100px 100px;
    }
    .ribbon-text {
        font-size: 1.1em;
        line-height: 1.6;
        color: #E0F2F1; /* Light text for contrast */
    }
    .ribbon-heading {
        font-size: 1.5em;
        font-weight: bold;
        color: #80CBC4; /* Accent color */
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. FILE UPLOADER AND DATA GENERATION FALLBACK
# ==============================================================================

# Custom function to generate simulated data (FALLBACK ONLY)
def generate_simulated_data_fallback():
    """Generates synthetic data (FALLBACK) if no file is uploaded."""
    np.random.seed(42)
    random.seed(42)
    N = 500 
    X = np.random.rand(N, len(FEATURE_NAMES)) * 100
    
    # Classification logic influenced by phenology features
    y = np.where(
        (X[:, 0] > 70) & (X[:, 4] < 40), 2, 
        np.where((X[:, 0] > 50) & (X[:, 2] > 60), 1, 0)
    ) 
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['Landcover_Class_ID'] = y
    return df

# --- Sidebar for File Upload ---
st.sidebar.header("Data Source Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload GEE-Exported Feature CSV (Required Columns: Phenology Features + Landcover_Class_ID)", 
    type="csv"
)
if uploaded_file is None:
    st.sidebar.warning("⚠️ Using **simulated fallback data**. Upload a CSV for real results.")

# ==============================================================================
# 2. DATA LOADING AND MODELING
# ==============================================================================
@st.cache_data(show_spinner="Training classification models on feature data...")
def load_and_train_model(uploaded_file):
    """Loads feature data, trains the Random Forest model, and provides assets."""
    
    df_features = None
    
    if uploaded_file is not None:
        try:
            # Load real data from CSV
            df_features = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ GEE feature data loaded successfully.")
            
            # Validation check
            required_cols = FEATURE_NAMES + ['Landcover_Class_ID']
            missing_cols = [col for col in required_cols if col not in df_features.columns]
            if missing_cols:
                st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}. Using fallback data.")
                df_features = generate_simulated_data_fallback() 
            
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded file: {e}. Using fallback data.")
            df_features = generate_simulated_data_fallback() 
    
    if df_features is None:
        df_features = generate_simulated_data_fallback()

    # Get data for time series visualization (static mock)
    time_series_data = get_time_series_data()

    X = df_features[FEATURE_NAMES]
    y = df_features['Landcover_Class_ID']
    
    # Check for enough data points to split
    if len(X) < 2:
        st.error("Not enough data points to train the model. Need at least 2 samples.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Train the main classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Train class-specific importance models
    class_importance_models = {}
    for class_id, class_name in CLASS_LABELS.items():
        y_binary = (y == class_id).astype(int)
        binary_model = RandomForestClassifier(n_estimators=50, random_state=42)
        binary_model.fit(X, y_binary)
        class_importance_models[class_name] = pd.Series(
            binary_model.feature_importances_, index=FEATURE_NAMES
        ).sort_values(ascending=False)
        
    # Mock a single-point prediction (first test point)
    mock_input = pd.DataFrame([X_test.iloc[0]], columns=FEATURE_NAMES)
    prediction = model.predict(mock_input)[0]
    overall_feature_importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    y_pred_test = model.predict(X_test)
    prediction_label = CLASS_LABELS[prediction]

    return (
        prediction_label, overall_feature_importances, 
        mock_input.iloc[0].to_dict(), y_test, y_pred_test, 
        time_series_data, class_importance_models
    )

# Attempt to load data and train model
results = load_and_train_model(uploaded_file)

# Check if model loading was successful before proceeding
if results is None:
    st.stop()

(
    prediction, overall_feature_importances, mock_data, 
    y_test, y_pred_test, time_series_data, class_importance_models
) = results

# Prepare data for accuracy tab
y_test_labels = np.array([CLASS_LABELS[i] for i in y_test])
y_pred_labels = np.array([CLASS_LABELS[i] for i in y_pred_test])

# --- Main Title ---
st.title("🛰️ GeoAI Phenology Classification Product")
st.caption("Advanced Landcover Classification using Hyper-Temporal NDVI variables.")

# --- ADDED: Product Value Proposition Ribbon ---
st.markdown("""
<div class='ribbon-box'>
    <div class='ribbon-heading'>Unlocking High-Accuracy, Scalable Land Monitoring</div>
    <p class='ribbon-text'>
        This application leverages **Hyper-Temporal Phenology**—the precise timing and magnitude of vegetation changes derived from multi-year MODIS NDVI data—to classify land cover. 
        Unlike standard spectral classification, which struggles with dynamic environments, our GeoAI approach provides **exceptional accuracy** by differentiating landcover based on their unique seasonal growth cycles. 
        This methodology is particularly valuable for large-scale **Global South** projects, such as those in East Africa's rangelands, as it uses **Google Earth Engine (GEE)-compatible** satellite data for **global scalability** and cost-effective monitoring where field data is scarce.
    </p>
</div>
""", unsafe_allow_html=True)
# --- END ADDED SECTION ---

# ==============================================================================
# 3. LLM EXPLANATION FUNCTION
# ==============================================================================

def get_llm_explanation(prediction: str, feature_importances_dict: dict):
    """Generates an LLM-based explanation using the Gemini model in bullet form."""
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        return ["LLM explanation failed: API Key is missing. Cannot provide interpretation."]

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             (
                 "You are a leading GeoAI expert specializing in rangeland ecology and MODIS phenology. "
                 "The prediction is: **{prediction}**. Feature importances: **{feature_importances_dict}**. "
                 "Analyze the key features influencing the classification. "
                 "Respond with **three distinct bullet points** covering: 1. The most critical phenological metric and its influence. 2. A secondary metric and its supporting role. 3. The general phenological signature of the predicted class."
                 "Use relevant emojis."
             )
            ),
            ("human", "Provide the interpretation in a professional and concise bulleted list.")
        ])

        chain = prompt | llm | StrOutputParser()
        interpretation_text = chain.invoke({
            "prediction": prediction, 
            "feature_importances_dict": str(feature_importances_dict)
        })
        
        # Split the interpretation into bullet points for separate formatting
        return [line.strip('-* ') for line in interpretation_text.split('\n') if line.strip().startswith(('-', '*'))]

    except Exception as e:
        return [f"An error occurred during LLM generation: {e}"]

# --- TAB LAYOUT ---
tab1, tab2, tab3 = st.tabs(["Interactive Map & AOI Analysis", "Classification & Feature Importance", "Accuracy & Data Export"])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE GEOSPATIAL MAP & AOI ANALYSIS
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Interactive Classified Map & Analysis")

    # Layout for Map and Sidebar Controls
    # CHANGED: Increased map column size (3.5 to 4.5) to make the map wider
    col_map, col_analysis = st.columns([4.5, 1.5]) 
    
    with col_analysis:
        st.subheader("AOI Controls")
        
        # AOI Selection
        site_geojson = json.loads(get_site_geojson())
        site_names = [f["properties"]["name"] for f in site_geojson["features"]]
        selected_site = st.selectbox(
            "Select Pre-defined Site:", 
            ["Draw Custom AOI"] + site_names,
            key="aoi_select"
        )
        
        # Base Map Selection
        st.subheader("Base Map")
        basemap_options = {
            'ESRI_WorldImagery': 'ESRI World Imagery (Satellite)',
            'ESRI_Vector_Canvas_Light': 'ESRI Light Canvas (Vector)',
            'OpenStreetMap': 'OpenStreetMap Standard'
        }
        selected_basemap_key = st.selectbox(
            "Select Base Map Type:", 
            options=list(basemap_options.keys()), 
            format_func=lambda x: basemap_options[x],
            key="basemap_select"
        )
        
        st.subheader("Prediction Sample")
        # Display the single-point prediction prominently
        st.info(f"**Sample Prediction:** {prediction}", icon="🎯")
        st.markdown(f"**Sample Input Data:**")
        st.json(mock_data)

        st.subheader("Legend")
        for lc_class, color in LANDCOVER_COLORS.items():
            st.markdown(
                f'<span style="background-color: {color}; color: black; padding: 2px 6px; border-radius: 3px; margin-right: 5px; font-weight: bold;">&nbsp;</span> {lc_class}',
                unsafe_allow_html=True
            )
            
    # --- MAP RENDERING ---
    with col_map:
        default_center = [-1.5, 35.5] 
        m = folium.Map(location=default_center, zoom_start=9, tiles=selected_basemap_key, attr=selected_basemap_key)
        
        # Add the Classified Landcover Layer (The real GeoAI product)
        folium.raster_layers.TileLayer(
            tiles=CLASSIFIED_MAP_TILE_URL,
            attr='Classified Landcover Product (MODIS Phenology)',
            name='Classified Landcover Map (MODIS)',
            overlay=True,
            control=True
        ).add_to(m)
        
        # Add Pre-defined Sites (Polygons from GeoJSON)
        folium.GeoJson(
            site_geojson,
            name='Pre-defined Sites',
            style_function=lambda feature: {
                'fillColor': LANDCOVER_COLORS.get(CLASS_LABELS.get(feature['properties'].get('class_id'), 'default'), 'grey'),
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.5,
            },
            tooltip=folium.features.GeoJsonTooltip(fields=['name']),
        ).add_to(m)
        
        # Add Folium Draw Control for Custom AOI
        draw = Draw(
            export=False,
            position="topleft",
            filename="my_aoi.geojson",
            draw_options={
                "polyline": False,
                "marker": False,
                "circlemarker": False,
                "circle": False,
                "rectangle": True,
                "polygon": True
            },
            edit_options={"edit": True, "remove": True}
        )
        draw.add_to(m)

        folium.LayerControl().add_to(m)
        
        # Capture draw results and display map
        # CHANGED: Increased map width to 1200 for better visualization
        output = st_folium(m, width=1200, height=700) 
        
        # Check for drawn geometry (Custom AOI)
        if output and 'last_active_drawing' in output:
            drawing = output['last_active_drawing']
            if drawing and drawing.get('geometry'):
                st.success("Custom AOI Drawn! 🗺️ (In a production app, this geometry would trigger a new GEE analysis job.)")
                st.json(drawing['geometry'])

        # Time Series Plot (Simulated for the Selected Site)
        if selected_site != "Draw Custom AOI":
            st.subheader(f"10-Year NDVI Time Series for: {selected_site}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time_series_data['Date'], time_series_data['NDVI'], color='#1f77b4', linewidth=2)
            ax.set_title(f"MODIS NDVI Time Series (2015-2025 Mock Data)", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("NDVI", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)


# ------------------------------------------------------------------------------
# TAB 2: CLASSIFICATION AND FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. Classification Breakdown and AI Interpretation")

    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.subheader("🌿 Class-Specific Feature Importance")
        
        # Dropdown to select which class's importance to view
        selected_class = st.selectbox(
            "View Feature Importance for Landcover Class:",
            options=list(CLASS_LABELS.values())
        )
        
        selected_importance = class_importance_models[selected_class]
        
        st.bar_chart(selected_importance, height=500)
        st.markdown(
            f"This chart shows the **discriminatory power** of each phenology variable when identifying a sample as **{selected_class}** (e.g., higher **Amplitude** might signify Grassland)."
        )

    with col2:
        st.subheader("🤖 AI Interpretation of Phenological Signature (Gemini)")
        
        with st.spinner("Generating expert interpretation..."):
            interpretation_points = get_llm_explanation(
                prediction=prediction,
                feature_importances_dict=overall_feature_importances.to_dict()
            )
        
        # Improved visual appeal using custom styling and a box
        st.markdown(
            """
            <div style='border: 2px solid #3CB371; padding: 15px; border-radius: 10px; background-color: #f0fff0;'>
                <p style='font-size: 1.1em; font-weight: bold; margin-bottom: 10px;'>
                    Key Insights for <span style="color: #2E8B57;">{prediction}</span> Classification:
                </p>
            """.format(prediction=prediction), unsafe_allow_html=True)
            
        for point in interpretation_points:
            st.markdown(f"**🟢 {point}**")
            
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Interpretation based on the overall feature importance of the trained Random Forest model.")
        
# ------------------------------------------------------------------------------
# TAB 3: ACCURACY AND EXPORT
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. Accuracy Assessment and Data Export")

    # --- ACCURACY ASSESSMENT ---
    st.subheader("Model Validation Metrics")
    
    col_metrics = st.columns(4)
    accuracy = accuracy_score(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average='macro')
    
    # Custom colored badges for key metrics
    col_metrics[0].markdown(f"**Overall Accuracy**")
    col_metrics[0].markdown(f"<p style='background-color:#007BFF; color:white; padding:10px; border-radius:5px; font-size:1.5em; text-align:center;'>{accuracy:.2%}</p>", unsafe_allow_html=True)
    
    col_metrics[1].markdown(f"**Macro F1-Score**")
    col_metrics[1].markdown(f"<p style='background-color:#28A745; color:white; padding:10px; border-radius:5px; font-size:1.5em; text-align:center;'>{f1_macro:.2f}</p>", unsafe_allow_html=True)
    
    col_metrics[2].markdown(f"**Classes Used**")
    col_metrics[2].markdown(f"<p style='background-color:#FFC107; color:black; padding:10px; border-radius:5px; font-size:1.5em; text-align:center;'>{len(CLASS_LABELS)}</p>", unsafe_allow_html=True)
    
    col_metrics[3].markdown(f"**Test Samples**")
    col_metrics[3].markdown(f"<p style='background-color:#DC3545; color:white; padding:10px; border-radius:5px; font-size:1.5em; text-align:center;'>{len(y_test)}</p>", unsafe_allow_html=True)

    st.markdown("---")
    
    col_cm, col_report = st.columns(2)
    
    with col_cm:
        st.markdown("#### 📉 Error Matrix / Confusion Matrix")
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=list(CLASS_LABELS.values()))
        cm_df = pd.DataFrame(cm, index=CLASS_LABELS.values(), columns=CLASS_LABELS.values())
        cm_df.index.name = 'True Class'
        cm_df.columns.name = 'Predicted Class'
        st.dataframe(cm_df, use_container_width=True)

    with col_report:
        st.markdown("#### 📋 Detailed Per-Class Metrics")
        
        # Convert classification report to DataFrame and apply styling
        report_data = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report_df = pd.DataFrame(report_data).transpose()
        
        # Apply visual styling: background gradient on f1-score for visual emphasis
        st.dataframe(
            report_df.style.background_gradient(cmap='YlGn', subset=['f1-score', 'recall', 'precision']).format(precision=2), 
            use_container_width=True
        )

    st.divider()

    # --- DATA EXPORT ---
    st.subheader("Export Analysis Assets")
    export_cols = st.columns(3)

    # 1. Export Feature Importances
    csv_data = overall_feature_importances.reset_index(name='Importance_Score')
    export_cols[0].download_button(
        label="Download Feature Importances (CSV) 💾",
        data=csv_data.to_csv(index=False).encode('utf-8'),
        file_name='feature_importances_report.csv',
        mime='text/csv',
        use_container_width=True
    )

    # 2. Export Accuracy Results
    # Reuse the styled DataFrame data for export
    export_cols[1].download_button(
        label="Download Accuracy Report (CSV) 📊",
        data=report_df.to_csv().encode('utf-8'),
        file_name='accuracy_assessment_report.csv',
        mime='text/csv',
        use_container_width=True
    )
    
    # 3. Export Landcover Map (Placeholder)
    export_cols[2].download_button(
        label="Download Classified Map (GeoTIFF Link) 🌐",
        data="Placeholder: Exported GeoTIFF would come from GEE export. Provide your cloud storage link here.",
        file_name='Landcover_Map_AOI_Placeholder.txt',
        mime='text/plain',
        help="In a production app, this button would serve the link to the final GEE-exported GeoTIFF.",
        use_container_width=True
    )
