"""
data_loader.py
--------------
Central data loading and caching module.
All pages import from here so the Excel is read only once.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os

# ── Path resolution ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Check candidate paths where the dataset might be located
_CANDIDATE_PATHS = [
    os.path.join(_THIS_DIR, "data", "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "..", "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "..", "data", "Telco_customer_churn.xlsx"),
    r"d:\Projects\customer churn prediction\churn_dashboard\data\Telco_customer_churn.xlsx",
    r"d:\Projects\customer churn prediction\churn_dashboard\Telco_customer_churn.xlsx",
    r"d:\Projects\customer churn prediction\Telco_customer_churn.xlsx",
]


def _find_excel() -> str:
    for p in _CANDIDATE_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Cannot find Telco_customer_churn.xlsx. "
        "Place it in the churn_dashboard/ folder or update _CANDIDATE_PATHS in data_loader.py."
    )


@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    """Load, clean, and return the full dataset."""
    path = _find_excel()
    df = pd.read_excel(path)

    # Fix Total Charges
    df["Total Charges"] = df["Total Charges"].replace("", pd.NA)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(0)

    # Binary churn column for easier aggregation
    df["Churned"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to the dataframe."""
    out = df.copy()

    if filters.get("contract") and filters["contract"] != ["All"]:
        out = out[out["Contract"].isin(filters["contract"])]

    if filters.get("internet") and filters["internet"] != ["All"]:
        out = out[out["Internet Service"].isin(filters["internet"])]

    if filters.get("payment") and filters["payment"] != ["All"]:
        out = out[out["Payment Method"].isin(filters["payment"])]

    if filters.get("senior") != "All":
        mapping = {"Senior": "Yes", "Non-Senior": "No"}
        out = out[out["Senior Citizen"] == mapping[filters["senior"]]]

    if filters.get("churn_status") != "All":
        out = out[out["Churn Label"] == filters["churn_status"]]

    tenure_range = filters.get("tenure_range")
    if tenure_range:
        out = out[
            (out["Tenure Months"] >= tenure_range[0]) &
            (out["Tenure Months"] <= tenure_range[1])
        ]

    return out


def get_model_feature_columns() -> list:
    """Return the ordered list of feature columns the model expects."""
    return [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ]
