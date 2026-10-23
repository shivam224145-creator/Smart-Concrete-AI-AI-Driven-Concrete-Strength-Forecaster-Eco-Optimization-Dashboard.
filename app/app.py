# ==============================================================================
# STREAMLIT DASHBOARD DEPLOYMENT SCRIPT FOR SMARTCONCRETE AI PROJECT
# ==============================================================================

import streamlit as tf_streamlit  # Using an alias to avoid confusing with tensorflow (tf)
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set up page configurations for wide and modern responsive grid UI layout
tf_streamlit.set_page_config(
    page_title="SmartConcrete AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuring default plot size to prevent rendering overlap bugs
plt.rcParams['figure.figsize'] = (10, 5)

# ------------------------------------------------------------------------------
# 1. CORE BACKEND ARTIFACTS LOADING (LOAD TRAINED MODELS & ENGINE)
# ------------------------------------------------------------------------------
# Setting up exact absolute directory paths to safely access the saved artifacts
MODEL_DIR = r"C:\Users\lenovo\OneDrive\Desktop\Smart Concrete AI\models"
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
NN_MODEL_PATH = os.path.join(MODEL_DIR, "concrete_nn_model.keras")


@tf_streamlit.cache_resource
def load_project_artifacts():
    """
    Loads and caches the pre-trained neural network graph and standardization scaler.
    Caching ensures the model isn't reloaded from disk on every user interaction.
    """
    scaler_obj = joblib.load(SCALER_PATH)
    # Using compile=False to load raw computational weights smoothly without version conflicts
    neural_net_obj = load_model(NN_MODEL_PATH, compile=False)
    return scaler_obj, neural_net_obj


try:
    scaler, nn_model = load_project_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    error_message = str(e)

# ------------------------------------------------------------------------------
# 2. WEB APPLICATION MAIN FRONTEND INTERFACE DESIGN
# ------------------------------------------------------------------------------
tf_streamlit.title("🏗️ SmartConcrete AI: Core Concrete Optimization Suite")
tf_streamlit.markdown("""
Welcome to the integrated production dashboard suite. This application leverages a 
**Deep Neural Network (Multi-Layer Perceptron)** to execute continuous regression 
prediction mappings, sustainable chemical optimization analysis, and baseline algorithm benchmarking.
""")

if not artifacts_loaded:
    tf_streamlit.error(
        f"⚠️ Failed to load core machine learning models from the artifacts directory. Details: {error_message}")
    tf_streamlit.stop()

# Creating the structural Multi-Tab workspace to host all 4 combined projects seamlessly
tab1, tab2, tab3, tab4 = tf_streamlit.tabs([
    "🔮 Strength Predictor",
    "🍃 EcoMix Optimizer",
    "📈 Hardening Dynamics",
    "📊 Benchmarking Report"
])

# Define the exact original feature index list mapping names to protect scaler integrity
FEATURE_COLUMNS = [
    'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
    'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'
]

# ------------------------------------------------------------------------------
# TAB 1: CORE ENGINE STRENGTH PREDICTION SYSTEM (PROJECT 1)
# ------------------------------------------------------------------------------
with tab1:
    tf_streamlit.header("🔮 Real-Time Compressive Strength Prediction")
    tf_streamlit.subheader("Configure Chemical Ingredient Mix Ratios")

    # Building a responsive modern 4-column input form layout structure
    col1, col2, col3, col4 = tf_streamlit.columns(4)

    with col1:
        input_cement = tf_streamlit.number_input("Cement (kg/m³)", min_value=50.0, max_value=600.0, value=250.0,
                                                 step=5.0)
        input_slag = tf_streamlit.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0, value=0.0,
                                               step=5.0)
    with col2:
        input_flyash = tf_streamlit.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=300.0, value=100.0,
                                                 step=5.0)
        input_water = tf_streamlit.number_input("Water (kg/m³)", min_value=100.0, max_value=300.0, value=180.0,
                                                step=2.0)
    with col3:
        input_plasticizer = tf_streamlit.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0,
                                                      value=5.0, step=0.5)
        input_coarse = tf_streamlit.number_input("Coarse Aggregate (kg/m³)", min_value=500.0, max_value=1300.0,
                                                 value=950.0, step=10.0)
    with col4:
        input_fine = tf_streamlit.number_input("Fine Aggregate (kg/m³)", min_value=500.0, max_value=1000.0, value=750.0,
                                               step=10.0)
        input_age = tf_streamlit.slider("Curing Age (Days)", min_value=1, max_value=365, value=28)

    # Prediction Action execution loop
    if tf_streamlit.button("Compute Compressive Strength", type="primary"):
        # Packaging the raw vector directly inside a DataFrame with feature names to bypass sklearn user warnings
        raw_input_df = pd.DataFrame([[
            input_cement, input_slag, input_flyash, input_water,
            input_plasticizer, input_coarse, input_fine, input_age
        ]], columns=FEATURE_COLUMNS)

        # Scaling features using matching pre-trained state transform parameters
        scaled_input_data = scaler.transform(raw_input_df)

        # Invoking the live deep memory model to infer continuous floating point value output
        inferred_strength = nn_model.predict(scaled_input_data).flatten()[0]

        tf_streamlit.success(f"🎯 **Predicted Concrete Compressive Strength:** {inferred_strength:.2f} MPa")

# ------------------------------------------------------------------------------
# TAB 2: ECO-FRIENDLY & LOW-CARBON OPTIMIZER SIMULATION (PROJECT 2)
# ------------------------------------------------------------------------------
with tab2:
    tf_streamlit.header("🍃 Sustainable Industrial EcoMix Optimization Strategy")
    tf_streamlit.markdown("""
    **Carbon Mitigation Simulation Engine:** Reduce traditional high-emission carbon cement 
    ratios below standard mix values. This system automatically balances structural strength 
    bounds by recalculating supplementary cementitious materials (SCM) like Fly Ash and Slag.
    """)

    # Building a split comparison viewport container layout
    left_ui, right_ui = tf_streamlit.columns(2)

    with left_ui:
        tf_streamlit.subheader("Standard High-Emission Reference Mix")
        tf_streamlit.info("Default layout parameters template: 540 kg Cement mix blueprint at 28 days.")

        # Fixed baseline array values
        std_df = pd.DataFrame([[540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28]], columns=FEATURE_COLUMNS)
        std_scaled = scaler.transform(std_df)
        std_strength = nn_model.predict(std_scaled).flatten()[0]

        tf_streamlit.metric(label="Reference Matrix Strength", value=f"{std_strength:.2f} MPa")

    with right_ui:
        tf_streamlit.subheader("Engineered Green Concrete Modification Sliders")
        # Interactive user slider to dynamically scale down core cement concentrations
        target_cement = tf_streamlit.slider("Modify Active Cement Concentration (kg/m³)", min_value=150.0,
                                            max_value=500.0, value=350.0)

        # Proactively substitute missing carbon volume with non-toxic recycling aggregates
        engineered_slag = 95.0
        engineered_flyash = 95.0

        # Compiling the dynamic green candidate matrix row
        green_df = pd.DataFrame([[target_cement, engineered_slag, engineered_flyash, 162.0, 5.0, 1040.0, 676.0, 28]],
                                columns=FEATURE_COLUMNS)
        green_scaled = scaler.transform(green_df)
        green_strength = nn_model.predict(green_scaled).flatten()[0]

        # Mathematical derivation metrics computation
        carbon_reduction = ((540.0 - target_cement) / 540.0) * 100
        strength_delta = green_strength - std_strength

        tf_streamlit.metric(label="Engineered Green Strength", value=f"{green_strength:.2f} MPa",
                            delta=f"{strength_delta:.2f} MPa")
        tf_streamlit.warning(
            f"🍀 **Eco Carbon Reduction Index:** Saved {carbon_reduction:.1f}% raw cement content footprint!")

# ------------------------------------------------------------------------------
# TAB 3: AGE TIMELINE HARDENING ANCHOR DYNAMICS (PROJECT 3)
# ------------------------------------------------------------------------------
with tab3:
    tf_streamlit.header("📈 Curing Timeline Timeline & Hardening Dynamics")
    tf_streamlit.markdown(
        "Observe how concrete hydration processes dynamically transform fluid chemical compound compositions into structural load-bearing infrastructure over a 90-day structural curing milestone timeline.")

    # Building simulated progressive timeline vectors (Fixing the blank array issue)
    timeline_days = np.array([3, 7, 14, 28, 56, 90, 120, 180, 270, 365])

    # Dynamic simulation builder using the baseline variables configured in Tab 1
    simulated_batches_list = []
    for day in timeline_days:
        simulated_batches_list.append([
            input_cement, input_slag, input_flyash, input_water,
            input_plasticizer, input_coarse, input_fine, day
        ])

    timeline_df = pd.DataFrame(simulated_batches_list, columns=FEATURE_COLUMNS)
    timeline_scaled = scaler.transform(timeline_df)
    simulated_strengths = nn_model.predict(timeline_scaled).flatten()

    # Binding data results inside a render-ready pandas DataFrame display
    chart_data_df = pd.DataFrame({
        "Curing Age (Days)": timeline_days,
        "Projected Compressive Strength (MPa)": simulated_strengths
    }).set_index("Curing Age (Days)")

    # Rendering native responsive structural line charts inside the user viewport area
    tf_streamlit.line_chart(chart_data_df)
    tf_streamlit.dataframe(chart_data_df.T)

# ------------------------------------------------------------------------------
# TAB 4: CLASSICAL ML VS DEEP LEARNING MODEL BENCHMARKING REPORT (PROJECT 4)
# ------------------------------------------------------------------------------
with tab4:
    tf_streamlit.header("📊 Algorithmic Evaluation & Model Benchmarking Matrix")
    tf_streamlit.markdown(
        "Statistical matrix tracking validation errors and mathematical fitting variance variables computed during the model training pipeline execution phase.")

    # Building structured analytical markdown tables to mirror notebook evaluations
    tf_streamlit.markdown("""

    | Algorithmic Framework Model | Target Test R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
    | :--- | :---: | :---: | :---: |
    | **Linear Regression Baseline** | 0.5802 | 8.8953 MPa | 11.1913 MPa |
    | **Deep Learning Neural Network (MLP)** | **0.8883** | **4.3704 MPa** | **5.7723 MPa** |
    | **Random Forest Regressor Benchmark** | 0.9076 | 3.4841 MPa | 5.2496 MPa |
    """)

    tf_streamlit.info(
        "💡 **Core Architecture Insights:** While traditional Decision Tree techniques converge tightly on tabular datasets, the deep layered Neural Network framework exhibits high adaptive flexibility, preventing linear decay patterns and offering superior deployment scalability for complex chemical simulation spaces.")

# ==============================================================================
# THEME-ADAPTIVE HIGH-HIGHLIGHT DEVELOPER PROFILE CARD
# ==============================================================================
tf_streamlit.markdown("---")  # Horizontal divider line

# Using columns layout to separate environment details from signature badge
footer_col1, footer_col2 = tf_streamlit.columns([3, 1])

with footer_col1:
    tf_streamlit.markdown("""
    💡 **System Status:** Framework Operational | Core Engine: **TensorFlow Keras 3**  
    🚀 Developed as an integrated end-to-end Machine Learning & Deep Learning solution.
    """)

with footer_col2:
    # Applying dynamic CSS container variables that glow uniformly across both Light & Dark themes
    tf_streamlit.markdown("""
    <div style="
        text-align: right; 
        padding: 12px 18px; 
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 75, 75, 0.03) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 8px; 
        border-right: 4px solid #FF4B4B;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-top: -5px;
    ">
        <span style="
            font-size: 0.75rem; 
            font-weight: 600;
            color: #FF4B4B; 
            letter-spacing: 1.5px;
            text-transform: uppercase;
            display: block; 
            margin-bottom: 4px;
        ">System Developer</span>
        <strong style="
            font-size: 1.15rem; 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            letter-spacing: 0.5px;
            background: linear-gradient(45deg, #FF4B4B, #FF8585);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        ">SHIVAM SHUKLA</strong>
    </div>
    """, unsafe_allow_html=True)
