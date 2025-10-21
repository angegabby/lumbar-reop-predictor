import streamlit as st
import joblib
import pandas as pd
import numpy as np

# App title and description
st.title("Lumbar Fusion Reoperation Risk Calculator")
st.markdown("Enter your details to estimate your risk of needing another surgery within 1 year after lumbar spinal fusion. (AUC 0.95, MIMIC-IV data)")

# Load model
model = joblib.load('lumbar_model.pkl')

# Function to predict and calibrate risk
def predict_risk(age, race_white, insurance_private, discharge_home, los_days, fusion_levels, charlson_score, 
                 chf, smoking, obesity, icu_stay, steroid_use, ssi):
    scaled_age = (min(age, 75.5) - 50) / 15
    scaled_los = (min(los_days, 36.8) - 5) / 3
    data = pd.DataFrame({
        'age': [scaled_age], 'race_white': [int(race_white)], 'insurance_private': [int(insurance_private)],
        'discharge_home': [int(discharge_home)], 'los_days': [scaled_los], 'fusion_levels': [fusion_levels],
        'anterior_approach': [0], 'charlson_score': [charlson_score], 'chf': [int(chf)], 'smoking': [int(smoking)],
        'obesity': [int(obesity)], 'icu_stay': [int(icu_stay)], 'steroid_use': [int(steroid_use)], 'ssi': [int(ssi)]
    })
    probs = model.predict_proba(data)
    risk = probs[:, 1][0] * 100 if isinstance(probs, np.ndarray) else probs.iloc[:, 1][0] * 100
    risk_factor_count = sum([chf, smoking, obesity, icu_stay, steroid_use, ssi, charlson_score > 2, fusion_levels > 2, age > 60, los_days > 10])
    calibration_factor = 0.15 if risk_factor_count <= 2 else 0.8 if risk_factor_count <= 5 else 1.2
    calibrated_risk = min(risk * calibration_factor, 50.0)
    return calibrated_risk

# Input form
st.header("Your Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 18, 100, 65)
    race_white = st.selectbox("Race", ["White", "Other"], index=0) == "White"
    insurance_private = st.selectbox("Insurance Type", ["Private", "Other"], index=0) == "Private"
    discharge_home = st.selectbox("Discharge Location", ["Home", "Other"], index=0) == "Home"
    los_days = st.number_input("Hospital Stay (days)", min_value=1, max_value=30, value=5)
    fusion_levels = st.number_input("Number of Fusion Levels", min_value=1, max_value=5, value=2)

with col2:
    charlson_score = st.number_input("Comorbidity Score (Charlson)", min_value=0, max_value=10, value=0)
    chf = st.checkbox("Heart Failure")
    smoking = st.checkbox("Current Smoker")
    obesity = st.checkbox("Obesity (BMI ≥ 30)")
    icu_stay = st.checkbox("ICU Stay During Hospitalization")
    steroid_use = st.checkbox("Steroid Use")
    ssi = st.checkbox("Surgical Site Infection")

# Predict button
if st.button("Calculate Risk"):
    # Predict risk
    risk = predict_risk(age, race_white, insurance_private, discharge_home, los_days, fusion_levels, charlson_score, 
                        chf, smoking, obesity, icu_stay, steroid_use, ssi)
    
    # Display result
    st.header("Your Reoperation Risk")
    st.metric("Risk of Reoperation", f"{risk:.1f}%")
    if risk > 20:
        st.error("High Risk: Consult your surgeon for additional evaluation.")
    elif risk > 10:
        st.warning("Moderate Risk: Monitor closely with your healthcare provider.")
    else:
        st.success("Low Risk: Continue with standard follow-up care.")
