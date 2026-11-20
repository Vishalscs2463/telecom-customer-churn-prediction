"""
app.py
------
Main entry point for the Telecom Customer Churn Analytics & Prediction Dashboard.

Run with:
    streamlit run app.py
"""

import sys
import os
import importlib.util

# Allow pages/ to import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F7F9FC;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2B4D7A 100%);
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 14px;
    }
    [data-testid="stSidebar"] .stMultiSelect div {
        color: #1E3A5F !important;
    }
    [data-testid="stSidebar"] h3 {
        color: rgba(255,255,255,0.9) !important;
        font-size: 15px;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.6) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }

    /* Header */
    h1, h2, h3 {
        color: #1E3A5F;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F, #2E86AB);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(30,58,95,0.3);
    }

    /* Form submit button */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1E3A5F, #2E86AB) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px !important;
        transition: all 0.2s !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #E0E8F0;
        border-radius: 8px;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Info/warning/success boxes */
    .stAlert {
        border-radius: 8px;
    }

    /* Plotly charts */
    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Page title divider */
    h2 {
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data & sidebar ───────────────────────────────────────────────────────
from data_loader import load_data
from sidebar import render_sidebar

try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Dataset not found: {e}")
    st.stop()

filters = render_sidebar(df)

# ── Dashboard header ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 60%, #00B4D8 100%);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        box-shadow: 0 6px 24px rgba(30,58,95,0.2);
    ">
        <div style="color:white;">
            <div style="font-size:11px; letter-spacing:3px; opacity:0.8; font-weight:600; margin-bottom:6px;">
                MACHINE LEARNING · BUSINESS INTELLIGENCE
            </div>
            <div style="font-size:28px; font-weight:800; line-height:1.2;">
                📡 Telecom Customer Churn Analytics & Prediction
            </div>
            <div style="font-size:14px; opacity:0.85; margin-top:8px;">
                IBM Telco Dataset · Gradient Boosting · 7,043 Customers · Threshold = 0.33
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Route to selected page ────────────────────────────────────────────────────
page = filters["page"]

import importlib, sys

def _load_page(module_file: str):
    """Dynamically load a page module from the pages/ folder."""
    pages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
    spec = importlib.util.spec_from_file_location(
        module_file, os.path.join(pages_dir, module_file + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if page == "overview":
    _load_page("01_executive_overview").render(df, filters)

elif page == "analysis":
    _load_page("02_churn_analysis").render(df, filters)

elif page == "model":
    _load_page("03_model_performance").render(df, filters)

elif page == "prediction":
    _load_page("04_customer_prediction").render(df, filters)

elif page == "insights":
    _load_page("05_business_insights").render(df, filters)
