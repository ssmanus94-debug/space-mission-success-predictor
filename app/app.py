"""
Streamlit App for ML Model Deployment
=====================================

This is your Streamlit application that deploys both your regression and
classification models. Users can input feature values and get predictions.

HOW TO RUN LOCALLY:
    streamlit run app/app.py

HOW TO DEPLOY TO STREAMLIT CLOUD:
    1. Push your code to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Set the main file path to: app/app.py
    5. Deploy!

WHAT YOU NEED TO CUSTOMIZE:
    1. Update the page title and description
    2. Update feature input fields to match YOUR features
    3. Update the model paths if you changed them
    4. Customize the styling if desired

Author: Sean McManus # 
Dataset: Global_Space_Exploration_Dataset.csv  
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Attempted Space Mission Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_resource
def load_models():
    """Load all saved models and artifacts."""
    base_path = Path(__file__).parent.parent / "models"
    models = {}
    try:
        # Load regression artifacts
        models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
        models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
        models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

        # Load classification artifacts
        models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
        models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
        models['label_encoder'] = joblib.load(base_path / "ordered_encoding.pkl")
        models['classification_features'] = joblib.load(base_path / "classification_features.pkl")
        
        try:
            models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
        except:
            models['binning_info'] = None
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    return models

def make_regression_prediction(models, input_data):
    input_scaled = models['regression_scaler'].transform(input_data)
    prediction = models['regression_model'].predict(input_scaled)
    return prediction[0]

def make_classification_prediction(models, input_data):
    input_scaled = models['classification_scaler'].transform(input_data)
    prediction_num = models['classification_model'].predict(input_scaled)[0]
    reverse_mapping = {v: k for k, v in models['label_encoder'].items()}
    return reverse_mapping[prediction_num]

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🚀 Mission Control")
page = st.sidebar.radio("Navigate:", ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"])

st.sidebar.markdown("---")

# --- DATA DICTIONARY SECTION ---
st.sidebar.markdown("### 📊 Data Dictionary")

with st.sidebar.expander("🎯 Target Variables"):
    st.write("**Success Rate (%):** The predicted numerical probability of mission achievement (Regression).")
    st.write("**Success Category:** Classification of mission outcome into Low, Medium, or High success (Classification).")

with st.sidebar.expander("⚙️ Input Variables"):
    st.write("**Budget:** Total funding in Billions of USD.")
    st.write("**Duration:** Expected mission length in Earth days.")
    st.write("**Mission Type:** Whether the craft is Manned (crewed) or Unmanned (robotic).")
    st.write("**Tech Maturity:** A calculated rank (1-5) based on the chosen technology, where 1 is legacy and 5 is experimental/AI-driven.")

st.sidebar.markdown("---")
st.sidebar.info(f"**Built by:** Sean McManus\n\nFull Stack Academy AI & ML Bootcamp 2026")
st.sidebar.markdown("[Project GitHub](https://github.com/ssmanus94-debug/space-mission-success-predictor)")info("Built by: **Sean McManus**\n\nFull Stack Academy AI & ML Bootcamp")

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.title("🚀 Space Mission Success Predictor")
    st.markdown("### Welcome, Commander!")
    st.write("This app uses Machine Learning to predict the success of space exploration missions based on historical data patterns.")
    st.markdown("""
    **Available Tools:**
    - **Regression**: Predicts the exact **Success Rate (%)**.
    - **Classification**: Predicts the **Success Category** (Low, Medium, High).
    """)

# =============================================================================
# REGRESSION PAGE
# =============================================================================
elif page == "📈 Regression Model":
    st.title("📈 Success Rate Prediction")
    models = load_models()
    if models:
        features = models['regression_features']
        
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("Budget (in Billion $)", 0.1, 100.0, 10.0, key="reg_b")
            duration = st.slider("Duration (in Days)", 1, 3000, 365, key="reg_d")
            mission_type = st.selectbox("Mission Type", ["Manned", "Unmanned"], key="reg_m")
        with col2:
            tech_choice = st.selectbox("Technology Used", ["Traditional Rocket", "Nuclear Propulsion", "Reusable Rocket", "Solar Propulsion", "AI_Navigation"], key="reg_t")
            tech_rank_map = {"Traditional Rocket": 1, "Solar Propulsion": 2, "Nuclear Propulsion": 3, "Reusable Rocket": 4, "AI_Navigation": 5}
            tech_maturity = tech_rank_map[tech_choice]
            st.info(f"**Tech Maturity Rank:** {tech_maturity}")

        # Build DataFrame
        reg_df = pd.DataFrame(0.0, index=[0], columns=features)
        reg_df['Budget (in Billion $)'] = budget
        reg_df['Duration (in Days)'] = duration
        reg_df['Tech_Maturity'] = tech_maturity
        if mission_type == "Unmanned": reg_df['Mission Type_Unmanned'] = 1
        if f"Technology Used_{tech_choice}" in reg_df.columns: reg_df[f"Technology Used_{tech_choice}"] = 1

        if st.button("🔮 Predict Success Percentage", type="primary"):
            res = make_regression_prediction(models, reg_df)
            st.success(f"### Predicted Success Rate: {res:,.2f}%")

# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================
elif page == "🏷️ Classification Model":
    st.title("🏷️ Success Category Prediction")
    models = load_models()
    if models:
        features = models['classification_features']
        
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("Budget (in Billion $)", 0.1, 100.0, 10.0, key="cls_b")
            duration = st.slider("Duration (in Days)", 1, 3000, 365, key="cls_d")
            mission_type = st.selectbox("Mission Type", ["Manned", "Unmanned"], key="cls_m")
        with col2:
            tech_choice = st.selectbox("Technology Used", ["Traditional Rocket", "Nuclear Propulsion", "Reusable Rocket", "Solar Propulsion", "AI_Navigation"], key="cls_t")
            tech_rank_map = {"Traditional Rocket": 1, "Solar Propulsion": 2, "Nuclear Propulsion": 3, "Reusable Rocket": 4, "AI_Navigation": 5}
            tech_maturity = tech_rank_map[tech_choice]
            st.info(f"**Tech Maturity Rank:** {tech_maturity}")

        # Build DataFrame
        cls_df = pd.DataFrame(0.0, index=[0], columns=features)
        cls_df['Budget (in Billion $)'] = budget
        cls_df['Duration (in Days)'] = duration
        cls_df['Tech_Maturity'] = tech_maturity
        if mission_type == "Unmanned": cls_df['Mission Type_Unmanned'] = 1
        if f"Technology Used_{tech_choice}" in cls_df.columns: cls_df[f"Technology Used_{tech_choice}"] = 1

        if st.button("🔮 Predict Success Category", type="primary"):
            label = make_classification_prediction(models, cls_df)
            emoji = {'Low Success': '🔴', 'Medium Success': '🟡', 'High Success': '🟢'}.get(label, '⚪')
            st.success(f"### Result: {emoji} {label}")
            if label == "High Success": st.balloons()

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built by Sean McManus | Full Stack Academy AI & ML Bootcamp 2026</div>", unsafe_allow_html=True)
