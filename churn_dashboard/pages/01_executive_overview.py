"""
pages/01_executive_overview.py
--------------------------------
Page 1 — Executive Overview
"""

import streamlit as st
import pandas as pd
from chart_utils import (
    kpi_card, donut_churn, bar_churn_rate, histogram_by_churn
)
from data_loader import apply_filters


def render(df: pd.DataFrame, filters: dict):
    st.markdown("## 🏠 Executive Overview")
    st.markdown(
        "High-level snapshot of customer churn across the telecom customer base. "
        "Use the sidebar filters to drill down by segment."
    )

    # ── Apply filters ────────────────────────────────────────────────────────
    dff = apply_filters(df, filters)

    total      = len(dff)
    churned    = int(dff["Churned"].sum())
    retained   = total - churned
    churn_rate = churned / total * 100 if total > 0 else 0

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")

    kpi_cols = st.columns(4)
    kpi_data = [
        ("Total Customers",    f"{total:,}",         "", "#1E3A5F"),
        ("Churned Customers",  f"{churned:,}",        "High-risk customers", "#E63946"),
        ("Retained Customers", f"{retained:,}",       "Currently active",   "#2DC653"),
        ("Churn Rate",         f"{churn_rate:.2f}%",  "Overall rate",       "#F4A261"),
    ]
    for col, (label, value, delta, color) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(kpi_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Contract bar ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(donut_churn(total, churned), use_container_width=True)
    with col2:
        st.plotly_chart(
            bar_churn_rate(dff, "Contract", "Churn Rate by Contract Type"),
            use_container_width=True,
        )

    # ── Row 2: Internet + Payment ─────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            bar_churn_rate(dff, "Internet Service", "Churn Rate by Internet Service"),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            bar_churn_rate(dff, "Payment Method", "Churn Rate by Payment Method"),
            use_container_width=True,
        )

    # ── Row 3: Tenure + Monthly Charges histograms ────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(
            histogram_by_churn(dff, "Tenure Months",
                               "Customer Tenure Distribution", "Tenure (Months)"),
            use_container_width=True,
        )
    with col6:
        st.plotly_chart(
            histogram_by_churn(dff, "Monthly Charges",
                               "Monthly Charges: Churned vs Retained", "Monthly Charges ($)"),
            use_container_width=True,
        )

    # ── Summary insight box ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Quick Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            f"**Month-to-month contracts** show the highest churn. "
            f"Longer contracts are strongly associated with retention."
        )
    with c2:
        st.warning(
            f"**Fiber optic** internet customers churn at a noticeably higher rate "
            f"compared to DSL and customers with no internet service."
        )
    with c3:
        st.success(
            f"Customers with **tenure > 24 months** show significantly lower churn, "
            f"underscoring the value of early-tenure retention programs."
        )

