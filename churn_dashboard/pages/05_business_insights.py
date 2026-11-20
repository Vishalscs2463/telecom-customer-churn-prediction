"""
pages/05_business_insights.py
-------------------------------
Page 5 — Business Insights & Recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from chart_utils import bar_churn_rate, COLORS
from data_loader import apply_filters


def _section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            border-bottom: 3px solid {COLORS['secondary']};
            padding-bottom: 8px;
            margin: 24px 0 16px 0;
        ">
            <span style="font-size:24px;">{icon}</span>
            <span style="font-size:20px; font-weight:700; color:{COLORS['primary']}; margin-left:8px;">{title}</span>
            {"<div style='font-size:13px; color:#6C757D; margin-top:4px;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _rec_card(priority: str, title: str, body: str, metric: str = ""):
    colors = {
        "Critical":    ("#E63946", "#FFF5F5"),
        "High":        ("#F4A261", "#FFFBF0"),
        "Medium":      ("#2E86AB", "#F0F8FF"),
        "Opportunity": ("#2DC653", "#F0FFF4"),
    }
    border_color, bg_color = colors.get(priority, ("#6C757D", "#F8F9FA"))
    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {border_color};
            background: {bg_color};
            border-radius: 0 10px 10px 0;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-weight:700; font-size:15px; color:{border_color};">{title}</span>
                <span style="font-size:11px; font-weight:600; background:{border_color}; color:white; padding:2px 8px; border-radius:10px;">{priority}</span>
            </div>
            <div style="font-size:13px; color:#333; line-height:1.6;">{body}</div>
            {"<div style='font-size:12px; color:" + border_color + "; margin-top:8px; font-weight:600;'>" + metric + "</div>" if metric else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _correlation_disclaimer():
    st.caption(
        "⚠️ **Important:** Patterns described here are **correlations** observed in the dataset "
        "and model predictions. They do not establish causal relationships. Business decisions "
        "should combine these insights with domain expertise and further investigation."
    )


def render(df: pd.DataFrame, filters: dict):
    st.markdown("## 💡 Business Insights & Recommendations")
    st.markdown(
        "Data-driven insights derived from the IBM Telco dataset EDA and the Tuned Gradient Boosting model. "
        "Each insight is linked to a concrete business recommendation."
    )
    _correlation_disclaimer()

    # ── Overview metrics ──────────────────────────────────────────────────────
    _section_header("📊", "Churn Landscape Summary")

    total = len(df)
    churned = df["Churned"].sum()
    churn_rate = churned / total

    mtm_df = df[df["Contract"] == "Month-to-month"]
    mtm_churn = mtm_df["Churned"].mean()

    new_df = df[df["Tenure Months"] <= 12]
    new_churn = new_df["Churned"].mean()

    fiber_df = df[df["Internet Service"] == "Fiber optic"]
    fiber_churn = fiber_df["Churned"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, note in [
        (c1, "Overall Churn Rate", f"{churn_rate:.1%}", f"{churned:,} of {total:,}"),
        (c2, "Month-to-Month Churn", f"{mtm_churn:.1%}", "vs 26.5% overall"),
        (c3, "New Customer Churn (<1yr)", f"{new_churn:.1%}", "Highest risk segment"),
        (c4, "Fiber Optic Churn", f"{fiber_churn:.1%}", "vs DSL and No Internet"),
    ]:
        with col:
            st.metric(label=label, value=val, delta=note, delta_color="off")

    # ── Key Churn Patterns ────────────────────────────────────────────────────
    _section_header("🔍", "Key Churn Patterns", "Observed in dataset + model feature importance")

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(
            bar_churn_rate(df, "Contract", "Churn Rate by Contract Type"),
            use_container_width=True,
        )
        st.markdown(
            f"**Finding:** Month-to-month customers churn at **{mtm_churn:.1%}**, "
            f"far above the overall rate of {churn_rate:.1%}. "
            "Contract type is the #1 feature in both model importance and EDA."
        )
    with p2:
        st.plotly_chart(
            bar_churn_rate(df, "Internet Service", "Churn Rate by Internet Service"),
            use_container_width=True,
        )
        dsl_churn = df[df["Internet Service"] == "DSL"]["Churned"].mean()
        st.markdown(
            f"**Finding:** Fiber optic churn ({fiber_churn:.1%}) is notably higher "
            f"than DSL ({dsl_churn:.1%}). This warrants investigation into pricing, "
            "service quality, and customer expectations."
        )

    p3, p4 = st.columns(2)
    with p3:
        st.plotly_chart(
            bar_churn_rate(df, "Tech Support", "Churn Rate by Tech Support"),
            use_container_width=True,
        )
        no_tech = df[df["Tech Support"] == "No"]["Churned"].mean()
        yes_tech = df[df["Tech Support"] == "Yes"]["Churned"].mean()
        st.markdown(
            f"**Finding:** Customers without Tech Support churn at **{no_tech:.1%}** vs "
            f"**{yes_tech:.1%}** for those with it — a {(no_tech - yes_tech):.1%} gap."
        )
    with p4:
        st.plotly_chart(
            bar_churn_rate(df, "Online Security", "Churn Rate by Online Security"),
            use_container_width=True,
        )
        no_sec = df[df["Online Security"] == "No"]["Churned"].mean()
        yes_sec = df[df["Online Security"] == "Yes"]["Churned"].mean()
        st.markdown(
            f"**Finding:** Customers without Online Security churn at **{no_sec:.1%}** vs "
            f"**{yes_sec:.1%}** — add-on services are strongly correlated with retention."
        )

    # Tenure insight
    _section_header("⏱️", "Tenure & Early-Stage Churn Risk")
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        import plotly.express as px
        tenure_bins = pd.cut(df["Tenure Months"], bins=[0, 6, 12, 24, 48, 72],
                             labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"])
        tenure_analysis = df.groupby(tenure_bins, observed=True)["Churned"].agg(["mean", "count"]).reset_index()
        tenure_analysis.columns = ["Tenure Band", "Churn Rate", "Count"]
        tenure_analysis["Churn %"] = tenure_analysis["Churn Rate"] * 100

        fig = go.Figure(go.Bar(
            x=tenure_analysis["Tenure Band"].astype(str),
            y=tenure_analysis["Churn %"],
            marker=dict(
                color=tenure_analysis["Churn %"],
                colorscale=[[0, COLORS["retained"]], [1, COLORS["churned"]]],
            ),
            text=[f"{v:.1f}%" for v in tenure_analysis["Churn %"]],
            textposition="outside",
            hovertemplate="Tenure: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title="Churn Rate by Tenure Band",
            xaxis_title="Customer Tenure",
            yaxis_title="Churn Rate (%)",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    with t_col2:
        st.markdown("#### Tenure Insight")
        for _, row in tenure_analysis.iterrows():
            st.metric(str(row["Tenure Band"]), f"{row['Churn %']:.1f}%",
                      delta=f"{int(row['Count']):,} customers", delta_color="off")

    # ── Business Recommendations ──────────────────────────────────────────────
    _section_header("🎯", "Business Recommendations")

    _rec_card(
        "Critical",
        "🚀 Improve New Customer Onboarding (< 12 months)",
        "New customers carry the highest churn risk. Implement a structured 90-day onboarding "
        "journey: welcome calls, service check-ins at 30/60/90 days, and a dedicated "
        "customer success representative for high-value accounts.",
        f"📈 Target: Reduce early churn from {new_churn:.1%} → below 20%"
    )

    _rec_card(
        "Critical",
        "📄 Convert Month-to-Month Customers to Long-Term Contracts",
        "Month-to-month customers churn at significantly higher rates. Launch targeted campaigns "
        "offering 10-15% discounts for switching to 1-year or 2-year contracts. Highlight "
        "stability benefits and add-on service bundles.",
        f"📈 Target: Convert 15% of M2M customers to 1+ year contracts"
    )

    _rec_card(
        "High",
        "🎯 Activate AI-Powered Proactive Outreach",
        "Use this churn prediction model operationally: score all customers weekly, flag those "
        "with probability ≥ 0.33, and route them to your retention team. Prioritise customers "
        "with probability ≥ 0.65 for immediate intervention.",
        "📈 Model detects 73.8% of actual churners before they leave"
    )

    _rec_card(
        "High",
        "📡 Investigate Fiber Optic Service Quality",
        "Fiber optic customers churn at disproportionately higher rates. Conduct NPS surveys "
        "and focus groups specifically for this segment. Investigate pricing competitiveness, "
        "speed/reliability, and service quality vs. customer expectations.",
        f"📈 Fiber churn: {fiber_churn:.1%} — requires dedicated segment strategy"
    )

    _rec_card(
        "Medium",
        "🔒 Bundle Security & Support Add-ons with New Plans",
        "Customers with Online Security and Tech Support churn significantly less. Consider "
        "including these add-ons for free in the first 6 months for new customers, then convert "
        "to paid. This reduces churn and increases ARPU.",
        "📈 Opportunity: Up to 15% improvement in retention for add-on users"
    )

    _rec_card(
        "Medium",
        "💳 Encourage Automatic Payment Methods",
        "Electronic check users show higher churn rates. Incentivise auto-pay enrollment "
        "(bank transfer or credit card) with small monthly discounts. Automatic payments "
        "reduce friction and correlate with longer customer lifetimes.",
    )

    _rec_card(
        "Opportunity",
        "👨‍👩‍👧 Develop Family & Multi-Line Value Plans",
        "Customers with dependents and partners have lower observed churn. Create family "
        "bundle plans that increase switching costs and add value for household decision-makers.",
    )

    # ── ROI Estimate ─────────────────────────────────────────────────────────
    _section_header("💰", "Estimated Business Impact")
    st.markdown(
        "Using conservative assumptions to illustrate potential value of a retention programme:"
    )

    arpu = df["Monthly Charges"].mean()
    avg_tenure_churned = df[df["Churned"] == 1]["Tenure Months"].mean()

    ri_col1, ri_col2, ri_col3 = st.columns(3)
    with ri_col1:
        st.metric("Avg Monthly Revenue / Customer", f"${arpu:.2f}")
    with ri_col2:
        st.metric("Avg Tenure of Churned Customer", f"{avg_tenure_churned:.0f} months")
    with ri_col3:
        # If model catches 73.8% of churners and 20% conversion on outreach
        saved = int(churned * 0.738 * 0.20)
        annual_revenue_saved = saved * arpu * 12
        st.metric("Potential Customers Saved*", f"{saved:,}",
                  delta=f"≈ ${annual_revenue_saved/1e6:.2f}M annual revenue", delta_color="normal")

    st.caption(
        "\\* Assumes model detects 73.8% of churners, 20% respond positively to retention outreach. "
        "Actual impact depends on outreach cost, conversion rate, and offer value. "
        "This is an illustrative estimate, not a financial projection."
    )

