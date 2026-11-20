"""
sidebar.py
----------
Renders the sidebar navigation and global filter widgets.
Returns a dict of active filter values.
"""

import streamlit as st
import pandas as pd


PAGES = {
    "🏠 Executive Overview": "overview",
    "🔍 Churn Analysis": "analysis",
    "🤖 Model Performance": "model",
    "🔮 Customer Prediction": "prediction",
    "💡 Business Insights": "insights",
}


def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 16px;
            text-align: center;
        ">
            <div style="color:white; font-size:14px; font-weight:700; letter-spacing:1px;">
                📡 TELECOM CHURN ANALYTICS
            </div>
            <div style="color:rgba(255,255,255,0.7); font-size:11px; margin-top:4px;">
                Powered by Gradient Boosting
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📂 Navigation")
        page = st.radio(
            label="Go to",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎛️ Global Filters")
        st.caption("Filters apply to Overview & Analysis pages")

        # Contract type
        contracts = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
        selected_contracts = st.multiselect(
            "Contract Type", contracts, default=["All"]
        )

        # Internet service
        internet_opts = ["All"] + sorted(df["Internet Service"].dropna().unique().tolist())
        selected_internet = st.multiselect(
            "Internet Service", internet_opts, default=["All"]
        )

        # Payment method
        payment_opts = ["All"] + sorted(df["Payment Method"].dropna().unique().tolist())
        selected_payment = st.multiselect(
            "Payment Method", payment_opts, default=["All"]
        )

        # Senior citizen
        senior_opt = st.radio("Customer Type", ["All", "Senior", "Non-Senior"], horizontal=True)

        # Churn status
        churn_opt = st.radio("Churn Status", ["All", "Yes", "No"], horizontal=True)

        # Tenure range
        min_tenure = int(df["Tenure Months"].min())
        max_tenure = int(df["Tenure Months"].max())
        tenure_range = st.slider(
            "Tenure Range (months)",
            min_value=min_tenure,
            max_value=max_tenure,
            value=(min_tenure, max_tenure),
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:11px; color:#6C757D; text-align:center;'>"
            "IBM Telco Customer Churn Dataset<br>n = 7,043 customers</div>",
            unsafe_allow_html=True,
        )

    return {
        "page": PAGES[page],
        "contract": selected_contracts,
        "internet": selected_internet,
        "payment": selected_payment,
        "senior": senior_opt,
        "churn_status": churn_opt,
        "tenure_range": tenure_range,
    }

