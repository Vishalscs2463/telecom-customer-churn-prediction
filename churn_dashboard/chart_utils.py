"""
chart_utils.py
--------------
Shared Plotly chart helpers used across dashboard pages.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1E3A5F",   # deep navy
    "secondary": "#2E86AB",   # medium blue
    "accent":    "#00B4D8",   # bright cyan-blue
    "churned":   "#E63946",   # red for churn
    "retained":  "#2DC653",   # green for retained
    "neutral":   "#6C757D",   # grey
    "light_bg":  "#F0F4F8",
    "white":     "#FFFFFF",
}

PLOTLY_TEMPLATE = "plotly_white"

FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#1E3A5F")


def _base_layout(**kwargs):
    return dict(
        template=PLOTLY_TEMPLATE,
        font=FONT,
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["light_bg"],
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


# ── KPI helper ────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "", color: str = "#1E3A5F"):
    return f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, #2E86AB 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size:13px; opacity:0.85; letter-spacing:1px; text-transform:uppercase; font-weight:600; margin-bottom:8px;">{label}</div>
        <div style="font-size:32px; font-weight:800; line-height:1.1;">{value}</div>
        {"<div style='font-size:12px; opacity:0.75; margin-top:6px;'>" + delta + "</div>" if delta else ""}
    </div>
    """


# ── Donut chart ───────────────────────────────────────────────────────────────
def donut_churn(total: int, churned: int) -> go.Figure:
    retained = total - churned
    fig = go.Figure(go.Pie(
        labels=["Churned", "Retained"],
        values=[churned, retained],
        hole=0.60,
        marker=dict(colors=[COLORS["churned"], COLORS["retained"]]),
        textinfo="percent+label",
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Churned vs Retained Customers"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        annotations=[dict(
            text=f"<b>{churned / total * 100:.1f}%</b><br>Churn",
            x=0.5, y=0.5, font_size=16, showarrow=False
        )],
    )
    return fig


# ── Bar chart: churn rate by category ────────────────────────────────────────
def bar_churn_rate(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = (
        df.groupby(col)["Churned"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grp.columns = [col, "Churn Rate", "Count"]
    grp["Churn Rate %"] = grp["Churn Rate"] * 100
    grp = grp.sort_values("Churn Rate %", ascending=True)

    fig = go.Figure(go.Bar(
        y=grp[col],
        x=grp["Churn Rate %"],
        orientation="h",
        marker=dict(
            color=grp["Churn Rate %"],
            colorscale=[[0, COLORS["accent"]], [1, COLORS["churned"]]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in grp["Churn Rate %"]],
        textposition="outside",
        hovertemplate="%{y}<br>Churn Rate: %{x:.1f}%<br>Count: %{customdata:,}<extra></extra>",
        customdata=grp["Count"],
    ))
    fig.update_layout(
        **_base_layout(title=title),
        xaxis=dict(title="Churn Rate (%)", range=[0, grp["Churn Rate %"].max() * 1.2]),
        yaxis=dict(title=""),
        height=max(300, 60 * len(grp) + 100),
    )
    return fig


# ── Histogram: tenure or monthly charges ─────────────────────────────────────
def histogram_by_churn(df: pd.DataFrame, col: str, title: str, xlabel: str) -> go.Figure:
    churned = df[df["Churn Label"] == "Yes"][col].dropna()
    retained = df[df["Churn Label"] == "No"][col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=retained, name="Retained",
        marker_color=COLORS["retained"], opacity=0.7,
        hovertemplate=f"Retained<br>{xlabel}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=churned, name="Churned",
        marker_color=COLORS["churned"], opacity=0.7,
        hovertemplate=f"Churned<br>{xlabel}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title=title),
        barmode="overlay",
        xaxis_title=xlabel,
        yaxis_title="Number of Customers",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


# ── Box plot ──────────────────────────────────────────────────────────────────
def box_by_churn(df: pd.DataFrame, col: str, title: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    for label, color in [("No", COLORS["retained"]), ("Yes", COLORS["churned"])]:
        sub = df[df["Churn Label"] == label][col].dropna()
        fig.add_trace(go.Box(
            y=sub, name="Retained" if label == "No" else "Churned",
            marker_color=color, boxmean=True,
        ))
    fig.update_layout(
        **_base_layout(title=title),
        yaxis_title=ylabel,
        showlegend=True,
    )
    return fig


# ── Feature importance chart ──────────────────────────────────────────────────
def feature_importance_chart(df: pd.DataFrame, top_n: int = 15, title: str = "Feature Importance") -> go.Figure:
    top = df.head(top_n).sort_values("Importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker_color=COLORS["secondary"],
        text=[f"{v:.4f}" for v in top["Importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(title=title),
        xaxis_title="Importance Score",
        height=max(400, 30 * top_n + 100),
    )
    return fig


# ── Confusion matrix heatmap ──────────────────────────────────────────────────
def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    labels = ["No Churn", "Churn"]
    z = cm[::-1]           # flip so (0,0) = TN is bottom-left
    text = [[f"<b>{v:,}</b>" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels[::-1],
        colorscale=[[0, "#EAF2FF"], [1, COLORS["primary"]]],
        showscale=False,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=22),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Confusion Matrix"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
    )
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────
def roc_curve_chart(fpr, tpr, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC Curve (AUC = {auc_score:.4f})",
        line=dict(color=COLORS["secondary"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
    ))
    fig.update_layout(
        **_base_layout(title="ROC Curve — Tuned Gradient Boosting"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.5, y=0.1),
        height=420,
    )
    return fig


# ── Precision-Recall curve ────────────────────────────────────────────────────
def pr_curve_chart(precision, recall) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name="Precision-Recall",
        line=dict(color=COLORS["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.12)",
    ))
    fig.update_layout(
        **_base_layout(title="Precision-Recall Curve"),
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
    )
    return fig


# ── Model comparison grouped bar ──────────────────────────────────────────────
def model_comparison_chart(model_data: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              "#F4A261", COLORS["churned"]]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=model_data["Model"],
            y=model_data[metric],
            marker_color=color,
            text=[f"{v:.2%}" for v in model_data[metric]],
            textposition="outside",
            textfont=dict(size=10),
        ))
    fig.update_layout(
        **_base_layout(title="Model Comparison — All Models"),
        barmode="group",
        xaxis_tickangle=-30,
        yaxis=dict(title="Score", range=[0, 1.15], tickformat=".0%"),
        legend=dict(orientation="h", y=1.1),
        height=500,
    )
    return fig


# ── Gauge chart for churn probability ────────────────────────────────────────
def probability_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    color = COLORS["churned"] if probability >= 0.33 else COLORS["retained"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color=color)),
        delta=dict(reference=33, increasing=dict(color=COLORS["churned"]),
                   decreasing=dict(color=COLORS["retained"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            steps=[
                dict(range=[0, 33], color="#D4EDDA"),
                dict(range=[33, 65], color="#FFF3CD"),
                dict(range=[65, 100], color="#F8D7DA"),
            ],
            threshold=dict(line=dict(color="black", width=3), thickness=0.75, value=33),
        ),
        title=dict(text="Churn Probability", font=dict(size=18)),
    ))
    fig.update_layout(
        paper_bgcolor="white",
        height=320,
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig

