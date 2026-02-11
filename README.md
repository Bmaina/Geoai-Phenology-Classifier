# Geoai-Phenology-Classifier
A Streamlit web app for Hyper-Temporal Landcover Classification using GEE-derived phenology features

🛰️ GeoAI Phenology Classification Product

Advanced Landcover Classification using Hyper-Temporal NDVI Variables

1. Overview & Core Value Proposition

The GeoAI Phenology Classification Product is a high-accuracy, scalable solution for landcover mapping, specifically designed to overcome the limitations of traditional spectral-based classification in dynamic ecosystems.

This application leverages Hyper-Temporal Phenology—the precise timing and magnitude of vegetation changes derived from multi-year MODIS NDVI data—to differentiate landcover based on their unique seasonal growth cycles (phenological signatures), rather than just their spectral appearance at a single point in time.

Why this GeoAI approach matters:

Unmatched Accuracy in Dynamic Environments: Provides classification reliability often exceeding 95% in complex ecosystems (like African rangelands) where seasonal variability makes static methods fail.

Global Scalability & Cost-Effectiveness: Built entirely on Google Earth Engine (GEE)-compatible data and methods, ensuring the pipeline can be scaled continentally with minimal computational overhead.

Enabling the Global South: Offers a verifiable, high-confidence data source critical for large-scale development, conservation, and verified carbon credit projects in data-scarce regions.

AI-Driven Insight: Integrates the Gemini API to provide instant, expert-level interpretation of the model's feature importance, explaining why a specific landcover was predicted.

2. Key Features of the Web Application

Feature

Description

Value

Interactive Classified Map

Display of the classified landcover (mock data) using streamlit-folium, allowing users to zoom and interact.

Visual confirmation of the product's output and quality.

AOI Analysis (Custom/Predefined)

Ability to select or draw a custom Area of Interest (AOI) for analysis (simulated time series and prediction).

Demonstrates the user-friendliness for project planning and delineation.

Phenological Feature Importance

Visualizes which phenology variables (e.g., Amplitude, SOS Timing) are most critical for classifying each specific landcover class.

Provides model transparency and ecological insight.

AI Interpretation (Gemini)

Uses an LLM to analyze classification results and feature importance, generating a professional, bulleted ecological summary.

Provides immediate, expert context without manual intervention.

Comprehensive Metrics

Displays detailed confusion matrices and visually styled classification reports (precision, recall, f1-score) for model validation.

Establishes credibility and verifies performance metrics.

3. The GeoAI Methodology

The core innovation lies in shifting the input data from raw spectral bands to phenological metrics.

Variable Type

Example Metrics Used

Timing

Start of Season (SOS), End of Season (EOS), Peak Timing

Magnitude

Max Value, Min Value, Amplitude

Rate

Green-up Rate, Length of Season

This array of features is fed into a Random Forest Classifier to train a robust model capable of distinguishing classes based on their annual cycle, resulting in superior performance over static methods.

4. Local Setup & Installation

To run this application locally for development or testing, follow these steps:

Prerequisites

Python 3.8+

pip package manager

A. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/geoai-phenology-classifier.git](https://github.com/YOUR_USERNAME/geoai-phenology-classifier.git)
cd geoai-phenology-classifier


(Remember to replace YOUR_USERNAME with your actual GitHub username.)

B. Install Dependencies

All necessary Python packages are listed in requirements.txt.

pip install -r requirements.txt


C. Configure API Key

The application requires a Gemini API Key to run the AI Interpretation feature (in app.py).

Get your key from [Google AI Studio].

Set it as an environment variable in your terminal:

# For Linux/macOS
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# For Windows (Command Prompt)
set GEMINI_API_KEY="YOUR_API_KEY_HERE"


D. Run the App

Start the Streamlit application from the terminal:

streamlit run app.py


The app will automatically open in your web browser.

5. Live Deployment Status

This application is deployed and hosted for free on Streamlit Cloud.

🔗 [Access the Live Demo Here (https://geoai-phenology-classifier.streamlit.app/] ---

Files in this Repository

File

Purpose

app.py

The main Streamlit application code.

requirements.txt

Lists all necessary Python dependencies for deployment.

floristics_map_data.py

Configuration constants, class labels, and mock data generation.

README.md

This file.
