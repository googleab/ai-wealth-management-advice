# dashboards/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from src.data_processing import load_and_preprocess_data
from src.config import DATA_PATH
from src.explainability import load_model, calculate_shap_values, plot_global_feature_importance

st.set_page_config(page_title="Client Risk Intelligence Dashboard", layout="wide")

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data/client_data.csv')
    return df

@st.cache(allow_output_mutation=True)
def load_trained_model():
    try:
        model = load_model()
    except Exception as e:
        st.error("Model not found. Please run the training pipeline first.")
        model = None
    return model

st.title("AI-Powered Client Risk Intelligence")
st.markdown("### Interactive Dashboard for Wealth Management Advisers")

df = load_data()
model = load_trained_model()

client_index = st.number_input("Select Client Index", min_value=0, max_value=len(df)-1, value=0, step=1)
client_data = df.iloc[[client_index]]
st.write("**Client Details:**", client_data)

if model is not None:
    # Preprocess client data
    from src.data_processing import build_preprocessing_pipeline
    pipeline = build_preprocessing_pipeline()
    # Preprocess the full data to get the same shape and transformation;
    # for production, store the transformed pipeline output.
    X_processed, _, _ = load_and_preprocess_data(DATA_PATH)
    single_client_features = X_processed[client_index].reshape(1, -1)
    prediction_prob = model.predict_proba(single_client_features)[0][1]
    risk_label = "High Risk" if prediction_prob >= 0.5 else "Low Risk"
    st.write(f"**Predicted Risk:** {risk_label} (Probability: {prediction_prob:.2f})")
    
    st.subheader("Global Feature Importance")
    explainer, shap_values = calculate_shap_values(model, X_processed)
    fig = plot_global_feature_importance(explainer, X_processed)
    st.pyplot(fig)
    
    st.subheader("Automated Summary Report")
    summary = ""
    row = client_data.squeeze()
    if row['engagement_score'] < 0.4:
        summary += "Low client engagement. "
    if row['income'] < 100000:
        summary += "Client income is below average. "
    if row['AUM'] < 500000:
        summary += "Wealth portfolio size is modest. "
    if row['age'] > 55:
        summary += "Higher age group, indicating higher risk. "
    st.info(f"Summary: {summary if summary else 'Client appears stable.'}")
else:
    st.warning("Model could not be loaded. Please ensure training is complete.")

st.markdown("### End-to-End Process Complete")
st.caption("This dashboard demonstrates production-level data processing, modeling, and explainability.")
