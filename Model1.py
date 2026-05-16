import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Risk Analyzer",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------------------
# Custom Modern Styling
# ------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #0A3D62;
}
.stButton>button {
    background: linear-gradient(to right, #0A3D62, #3C6382);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = joblib.load("Bankruptcy_model.pkl")

# ------------------------------------------------
# Risk Mapping
# ------------------------------------------------
risk_mapping = {
    "Low": 0.0,
    "Medium": 0.5,
    "High": 1.0
}

# ------------------------------------------------
# Header
# ------------------------------------------------
st.title("📉 Bankruptcy Risk Prediction System")
st.markdown("### AI-Powered Corporate Financial Risk Intelligence Platform")
st.write("---")

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Company Risk Inputs")

industrial_ui = st.sidebar.selectbox("Industrial Risk", ["Low", "Medium", "High"])
management_ui = st.sidebar.selectbox("Management Risk", ["Low", "Medium", "High"])
financial_ui = st.sidebar.selectbox("Financial Flexibility", ["Low", "Medium", "High"])
credibility_ui = st.sidebar.selectbox("Credibility", ["Low", "Medium", "High"])
competitiveness_ui = st.sidebar.selectbox("Competitiveness", ["Low", "Medium", "High"])
operating_ui = st.sidebar.selectbox("Operating Risk", ["Low", "Medium", "High"])

predict_button = st.sidebar.button("Predict Bankruptcy Risk")

# Convert to numeric
industrial = risk_mapping[industrial_ui]
management = risk_mapping[management_ui]
financial = risk_mapping[financial_ui]
credibility = risk_mapping[credibility_ui]
competitiveness = risk_mapping[competitiveness_ui]
operating = risk_mapping[operating_ui]

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Performance", "ℹ️ About"])

# =====================================================
# TAB 1 — Prediction
# =====================================================
with tab1:

    if predict_button:

        input_data = np.array([[industrial, management, financial,
                                credibility, competitiveness, operating]])

        probability = model.predict_proba(input_data)[0][1]

        col1, col2 = st.columns(2)

        # -----------------------------
        # Risk Result Card
        # -----------------------------
        with col1:

            st.subheader("Prediction Result")

            if probability > 0.7:
                st.error("🔴 High Bankruptcy Risk")
            elif probability > 0.4:
                st.warning("🟡 Medium Bankruptcy Risk")
            else:
                st.success("🟢 Low Bankruptcy Risk")

            # Modern Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0A3D62"},
                    'steps': [
                        {'range': [0, 40], 'color': "#58D68D"},
                        {'range': [40, 70], 'color': "#F4D03F"},
                        {'range': [70, 100], 'color': "#EC7063"}
                    ],
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Input Summary
        # -----------------------------
        with col2:

            st.subheader("Selected Risk Profile")

            input_df = pd.DataFrame({
                "Feature": [
                    "Industrial Risk",
                    "Management Risk",
                    "Financial Flexibility",
                    "Credibility",
                    "Competitiveness",
                    "Operating Risk"
                ],
                "Level": [
                    industrial_ui,
                    management_ui,
                    financial_ui,
                    credibility_ui,
                    competitiveness_ui,
                    operating_ui
                ]
            })

            st.dataframe(input_df, use_container_width=True)

        st.write("---")

        # -----------------------------
        # Enhanced Feature Importance
        # -----------------------------
        st.subheader("Feature Importance Analysis")

        importances = model.feature_importances_
        feature_names = [
            "Industrial Risk",
            "Management Risk",
            "Financial Flexibility",
            "Credibility",
            "Competitiveness",
            "Operating Risk"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        fig2.update_layout(
            height=400,
            showlegend=False,
            title="Relative Feature Contribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# TAB 2 — Model Performance
# =====================================================
with tab2:

    st.subheader("Model Performance Summary")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Accuracy", "100%")
    colB.metric("Recall", "100%")
    colC.metric("Precision", "100%")
    colD.metric("ROC-AUC", "1.0")

    st.write("---")

    st.subheader("Confusion Matrix")

    cm = np.array([[29, 0],
                   [0, 21]])

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-Bankrupt", "Bankrupt"],
        y=["Non-Bankrupt", "Bankrupt"]
    )

    st.plotly_chart(fig_cm, use_container_width=True)

# =====================================================
# TAB 3 — About
# =====================================================
with tab3:

    st.markdown("""
    ### Project Overview

    This AI system predicts corporate bankruptcy risk based on structured financial risk indicators.

    ### Highlights
    - Trained using Random Forest
    - Achieved 99.6% Cross-Validation Accuracy
    - Real-time interactive prediction
    - Professional ML deployment using Streamlit

    This project demonstrates end-to-end machine learning deployment capability.
    """)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.write("---")
st.markdown(
    "<center>Developed by Shaik Rashid | AI Bankruptcy Intelligence System</center>",
    unsafe_allow_html=True
)