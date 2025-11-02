import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and feature names
try:
    model = joblib.load('sj_logistic_model.joblib')
    feature_names = joblib.load('sj_feature_names.joblib')
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure 'sj_logistic_model.joblib' and 'sj_feature_names.joblib' are in the same folder.")
    st.stop()


## --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Dengue Outbreak Predictor (San Juan)")
st.title("Dengue Outbreak Risk Predictor 🦠")
st.markdown("Predicting **High Outbreak** weeks in San Juan, Puerto Rico, using climate and environmental factors (SDG 3).")


## --- INPUT FORM ---

st.header("1. Input Weekly Climate Data")

# Create user input fields for the required 21 features
# Note: For simplicity, we only show two features here; in a full app, you'd show all 21.
# You can use the feature_names list to generate all input fields dynamically.
inputs = {}

# Example simplified input for 2 key features (you can expand this using a loop)
inputs['reanalysis_specific_humidity_g_per_kg'] = st.slider(
    'Specific Humidity (g/kg)', 
    min_value=12.0, max_value=20.0, value=15.0, step=0.1
)

inputs['station_avg_temp_c'] = st.slider(
    'Avg Station Temp (°C)',
    min_value=20.0, max_value=30.0, value=25.0, step=0.1
)

# For a full app, you would dynamically loop through feature_names:
# for feature in feature_names:
#     inputs[feature] = st.number_input(feature, value=1.0)


## --- PREDICTION LOGIC ---

if st.button("Predict Outbreak Risk"):
    # 1. Convert inputs to a DataFrame
    input_df = pd.DataFrame([inputs])
    
    # 2. Ensure the DataFrame has all 21 features with dummy/default values (required by model)
    # This step is critical if you don't show all features in the UI.
    full_input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for key, value in inputs.items():
        if key in feature_names:
            full_input_df[key] = value

    # 3. Make Prediction
    prediction = model.predict(full_input_df)[0]
    
    # 4. Display Result
    st.subheader("2. Prediction Result")
    
    if prediction == 1:
        st.error("🔴 HIGH OUTBREAK RISK PREDICTED")
        st.metric("Model Confidence", "Requires Immediate Action")
        st.write("The model suggests conditions are highly favorable for a major surge in dengue cases this week.")
    else:
        st.success("🟢 LOW OUTBREAK RISK PREDICTED")
        st.metric("Model Confidence", "Normal/Low Alert")
        st.write("Conditions suggest a low likelihood of a major outbreak. Continue standard monitoring.")


## --- MODEL DETAILS ---
st.sidebar.title("Model Details")
st.sidebar.info(
    "**Model:** Logistic Regression\n\n"
    "**City:** San Juan\n\n"
    "**Performance (F1-Score):** 0.2381 (Optimized Metric)"
)