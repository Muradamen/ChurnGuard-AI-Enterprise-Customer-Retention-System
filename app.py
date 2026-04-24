import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Page Config
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

# Initialize SHAP JS for force plots
shap.initjs()

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("deployment_ready_data.csv")
    summary = pd.read_csv("summary_data.csv")
    return df, summary

@st.cache_resource
def load_model():
    model = pickle.load(open("churn_model.pkl", "rb"))
    features = pickle.load(open("feature_names.pkl", "rb"))
    return model, features

df, summary = load_data()
model, feature_names = load_model()

# ----------------------------
# SHAP Setup
# ----------------------------
@st.cache_resource
def load_shap():
    explainer = shap.TreeExplainer(model)
    return explainer

explainer = load_shap()

# ----------------------------
# Title
# ----------------------------
st.title("📊 ChurnGuard AI Dashboard")
st.markdown("### Telecom Customer Retention & Revenue Optimization System")

# ----------------------------
# KPIs
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Customers", len(df))
col2.metric("⚠️ High Risk Customers", (df["Risk_Level"] == "High Risk").sum())
col3.metric("💰 Revenue at Risk (ETB)", f"{df['Revenue_At_Risk'].sum():,.0f}")

st.divider()

# ----------------------------
# Global Insights
# ----------------------------
st.subheader("🧠 Global Churn Drivers (SHAP)")
sample_size = min(300, len(df))
sample = df.sample(sample_size, random_state=42)
X_sample = pd.get_dummies(sample, drop_first=True)
for col in feature_names: 
    if col not in X_sample.columns: X_sample[col] = 0
X_sample = X_sample[feature_names]
shap_values = explainer.shap_values(X_sample)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(fig)

st.divider()

# ----------------------------
# Customer-Level Analysis
# ----------------------------
st.subheader("🔍 Customer-Level Analysis")
idx = st.number_input("Select Customer Index", 0, len(df)-1, 0)
customer = df.iloc[int(idx)]

st.write("### Recommended Action")
st.success(customer["Recommended_Action"])

# Local SHAP Explanation
customer_input = pd.get_dummies(pd.DataFrame([customer]), drop_first=True)
for col in feature_names:
    if col not in customer_input.columns: customer_input[col] = 0
customer_input = customer_input[feature_names]

# ⚡ NEW: Interactive SHAP Force Plot
st.subheader("⚡ Interactive SHAP Force Plot")
shap_val_local = explainer.shap_values(customer_input)

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_val_local[0],
    customer_input.iloc[0],
    matplotlib=False
)

# Convert to HTML and render
shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
components.html(shap_html, height=200)

st.divider()

# 🧪 What-If Simulation
st.subheader("🧪 What-If Simulation")
tenure = st.slider("Tenure (Months)", 0, 72, int(customer['tenure']))
monthly = st.slider("Monthly Charges", 0, 200, float(customer['MonthlyCharges']))
sim_input = customer_input.copy()
sim_input['tenure'] = tenure
sim_input['MonthlyCharges'] = monthly
prob = model.predict_proba(sim_input)[:,1][0]
st.metric("Simulated Churn Probability", f"{prob:.2%}")
