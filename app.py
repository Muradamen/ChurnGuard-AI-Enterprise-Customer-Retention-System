import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

# -----------------------------------
# Safe Load
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/deployment_ready_data.csv")
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open("models/churn_model.pkl", "rb"))
    features = pickle.load(open("models/feature_names.pkl", "rb"))
    return model, features

try:
    df = load_data()
    model, feature_names = load_model()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------------
# SHAP Setup
# -----------------------------------
@st.cache_resource
def load_explainer():
    return shap.Explainer(model)

explainer = load_explainer()

# -----------------------------------
# Preprocess Function (CRITICAL FIX)
# -----------------------------------
def preprocess(input_df, feature_names):
    df_enc = pd.get_dummies(input_df, drop_first=True)
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0
    return df_enc[feature_names]

# -----------------------------------
# Derived Metrics
# -----------------------------------
df["Priority"] = df["Revenue_At_Risk"].rank(ascending=False)

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["Overview", "Customer Analysis", "Simulation"]
)

# -----------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------
if page == "Overview":

    st.title("📊 ChurnGuard AI Dashboard")

    # KPIs
    total_revenue = df["CLTV"].sum() if "CLTV" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", len(df))
    col2.metric("⚠️ High Risk", (df["Risk_Level"] == "High Risk").sum())
    col3.metric("💰 Revenue at Risk", f"{df['Revenue_At_Risk'].sum():,.0f} ETB")
    col4.metric("💵 Total Customer Value", f"{total_revenue:,.0f} ETB")

    st.divider()

    # Segment Analysis
    st.subheader("📊 Risk Segmentation Summary")

    segment = df.groupby("Risk_Level").agg({
        "Churn_Probability": "mean",
        "Revenue_At_Risk": "sum"
    }).reset_index()

    st.dataframe(segment)

    st.divider()

    # Top Customers
    st.subheader("🚨 Top Revenue-Risk Customers")

    top_customers = df.sort_values("Revenue_At_Risk", ascending=False).head(10)

    st.dataframe(top_customers[[
        "tenure",
        "MonthlyCharges",
        "Churn_Probability",
        "Revenue_At_Risk",
        "Recommended_Action"
    ]])

    st.divider()

    # Global SHAP
    st.subheader("🧠 Global Churn Drivers")

    sample = df.sample(min(300, len(df)), random_state=42)
    X_sample = preprocess(sample, feature_names)

    @st.cache_data
    def compute_shap(X):
        return explainer(X)

    shap_values = compute_shap(X_sample)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

    st.divider()

    # Recommendations
    st.subheader("🎯 Strategic Recommendations")

    st.info("""
    - Prioritize high CLTV customers with high churn risk  
    - Offer contract incentives to reduce churn  
    - Target high monthly charge users with promotions  
    - Improve onboarding for low-tenure customers  
    """)

# -----------------------------------
# PAGE 2: CUSTOMER ANALYSIS
# -----------------------------------
elif page == "Customer Analysis":

    st.title("🔍 Customer Analysis")

    idx = st.number_input("Select Customer Index", 0, len(df)-1, 0)
    customer = df.iloc[int(idx)]

    st.write("### Customer Overview")
    st.write(customer)

    st.success(customer["Recommended_Action"])

    # SHAP Local
    customer_df = pd.DataFrame([customer])
    customer_input = preprocess(customer_df, feature_names)

    shap_values_local = explainer(customer_input)

    st.subheader("⚡ SHAP Force Plot")

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values_local.values[0],
        customer_input.iloc[0],
        matplotlib=False
    )

    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(shap_html, height=250)

# -----------------------------------
# PAGE 3: SIMULATION
# -----------------------------------
elif page == "Simulation":

    st.title("🧪 What-If Simulation")

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)

    if st.button("Predict Churn"):

        new_data = pd.DataFrame(columns=feature_names)
        new_data.loc[0] = 0

        if "tenure" in new_data.columns:
            new_data["tenure"] = tenure
        if "MonthlyCharges" in new_data.columns:
            new_data["MonthlyCharges"] = monthly

        prob = model.predict_proba(new_data)[:,1][0]

        st.metric("Predicted Churn Probability", f"{prob:.2%}")

        if prob > 0.8:
            st.error("🔥 High Risk → Call + Discount")
        elif prob > 0.6:
            st.warning("⚠️ Medium Risk → Promotion")
        else:
            st.success("✅ Low Risk → No Action")

# -----------------------------------
# DOWNLOAD BUTTON
# -----------------------------------
st.sidebar.download_button(
    "⬇️ Download High-Risk Customers",
    df[df["Risk_Level"] == "High Risk"].to_csv(index=False),
    file_name="high_risk_customers.csv"
)