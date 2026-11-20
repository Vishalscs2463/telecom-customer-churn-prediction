"""
model_loader.py
---------------
Loads the saved Gradient Boosting pipeline and test results.
Falls back gracefully with a warning if the model file is missing.
"""

import os
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, "model", "gradient_boosting_pipeline.joblib")
_RESULTS_PATH = os.path.join(_THIS_DIR, "model", "test_results.joblib")

THRESHOLD = 0.33


@st.cache_resource(show_spinner="Loading model …")
def load_pipeline():
    """Return the trained sklearn Pipeline, or None if not yet exported."""
    try:
        import joblib
        return joblib.load(_MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading test results …")
def load_test_results():
    """Return the saved test-result dict, or None if not yet exported."""
    try:
        import joblib
        return joblib.load(_RESULTS_PATH)
    except FileNotFoundError:
        return None


def model_missing_warning():
    st.warning(
        "⚠️ **Model file not found.**  \n"
        "Please run `export_model.py` from your Jupyter/Colab environment "
        "and copy the resulting `model/` folder into this directory.  \n"
        "See **README.md** for step-by-step instructions."
    )

