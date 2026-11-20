"""
pages/02_churn_analysis.py
---------------------------
Page 2 — Detailed Customer Churn Analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from chart_utils import bar_churn_rate, box_by_churn, histogram_by_churn, COLORS
from data_loader import apply_filters


def _insight_card(title: str, body: str, icon: str = "📌"):
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {COLORS['secondary']};
            background: #F0F4F8;
            border-radius: 0 8px 8px 0;
            padding: 14px 18px;
            margin-bottom: 12px;
        ">
            <div style="font-weight:700; color:{COLORS['primary']}; margin-bottom:4px;">{icon} {title}</div>
            <div style="font-size:13px; color:#333;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(df: pd.DataFrame, filters: dict):
    st.markdown("## 🔍 Customer Churn Analysis")
    st.markdown(
        "Detailed exploratory analysis of churn patterns across demographics, "
        "services, and account attributes. All statistics are computed directly from the dataset."
    )

    dff = apply_filters(df, filters)

    # ── Section 1: Demographic Factors ───────────────────────────────────────
    st.markdown("### 👥 Demographic Factors")

    col1, col2 = st.columns(2)
    with col1:
        # Senior Citizen — map Yes/No to labels for display
        df_senior = dff.copy()
        df_senior["Senior Citizen Label"] = df_senior["Senior Citizen"].map(
            {"No": "Non-Senior", "Yes": "Senior", 0: "Non-Senior", 1: "Senior"}
        )
        st.plotly_chart(
            bar_churn_rate(df_senior, "Senior Citizen Label", "Churn Rate by Senior Citizen Status"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            bar_churn_rate(dff, "Partner", "Churn Rate by Partner Status"),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            bar_churn_rate(dff, "Dependents", "Churn Rate by Dependents"),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            bar_churn_rate(dff, "Gender", "Churn Rate by Gender"),
            use_container_width=True,
        )

    # ── Section 2: Contract & Billing ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Contract & Billing")

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(
            bar_churn_rate(dff, "Contract", "Churn Rate by Contract Type"),
            use_container_width=True,
        )
    with col6:
        st.plotly_chart(
            bar_churn_rate(dff, "Paperless Billing", "Churn Rate by Paperless Billing"),
            use_container_width=True,
        )

    st.plotly_chart(
        bar_churn_rate(dff, "Payment Method", "Churn Rate by Payment Method"),
        use_container_width=True,
    )

    # ── Section 3: Internet Services ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Internet & Add-on Services")

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(
            bar_churn_rate(dff, "Internet Service", "Churn Rate by Internet Service"),
            use_container_width=True,
        )
    with col8:
        st.plotly_chart(
            bar_churn_rate(dff, "Online Security", "Churn Rate by Online Security"),
            use_container_width=True,
        )

    col9, col10 = st.columns(2)
    with col9:
        st.plotly_chart(
            bar_churn_rate(dff, "Tech Support", "Churn Rate by Tech Support"),
            use_container_width=True,
        )
    with col10:
        st.plotly_chart(
            bar_churn_rate(dff, "Online Backup", "Churn Rate by Online Backup"),
            use_container_width=True,
        )

    # ── Section 4: Numerical Analysis ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Tenure & Monthly Charges")

    col11, col12 = st.columns(2)
    with col11:
        st.plotly_chart(
            box_by_churn(dff, "Tenure Months", "Tenure vs Churn", "Tenure (Months)"),
            use_container_width=True,
        )
    with col12:
        st.plotly_chart(
            box_by_churn(dff, "Monthly Charges", "Monthly Charges vs Churn", "Monthly Charges ($)"),
            use_container_width=True,
        )

    # Tenure distribution histogram
    st.plotly_chart(
        histogram_by_churn(dff, "Tenure Months",
                           "Tenure Distribution — Churned vs Retained", "Tenure (Months)"),
        use_container_width=True,
    )

    # ── Summary Stats Table ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Summary Statistics by Churn Status")

    summary = (
        dff.groupby("Churn Label")[["Tenure Months", "Monthly Charges", "Total Charges"]]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    summary.columns = [" ".join(c).title() for c in summary.columns]
    st.dataframe(summary, use_container_width=True)

    # ── Key Insight Cards ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Key Analytical Findings")

    _insight_card(
        "Contract Type is the Strongest Predictor",
        "Month-to-month customers show dramatically higher observed churn rates compared to "
        "one-year and two-year contract customers. This is consistent with both the EDA and "
        "the model's feature importance rankings.",
        "📌"
    )
    _insight_card(
        "New Customers Are at Highest Risk",
        "Customers in their first 12 months of tenure have the highest observed churn rate. "
        "Early onboarding and engagement programs are critical for reducing early-stage churn.",
        "⏱️"
    )
    _insight_card(
        "Fiber Optic Segment Shows Elevated Churn",
        "Fiber optic internet customers churn at a higher rate than DSL customers. "
        "This may reflect higher pricing, unmet expectations, or service quality concerns. "
        "Note: this is an observed correlation, not a proven causal relationship.",
        "📡"
    )
    _insight_card(
        "Security & Support Add-ons Correlated with Retention",
        "Customers with Online Security and Tech Support services show notably lower observed "
        "churn rates, suggesting that value-added services improve customer stickiness.",
        "🔒"
    )
    _insight_card(
        "Customers Without Dependents Churn More",
        "Customers without dependents have higher observed churn rates, possibly reflecting "
        "lower switching costs and fewer reasons to stay with one provider.",
        "👨‍👩‍👧"
    )
