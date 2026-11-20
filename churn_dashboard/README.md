# Telecom Customer Churn Analytics & Prediction Dashboard

A professional, interactive Streamlit dashboard that combines **Exploratory Data Analysis**, **Statistical Insights**, and **live ML-powered Churn Predictions** using the IBM Telco Customer Churn dataset.

---

## 📂 Project Structure

```
churn_dashboard/
├── app.py                      ← Main Streamlit entry point
├── data_loader.py              ← Data loading & filter logic
├── model_loader.py             ← Load trained model & test results
├── chart_utils.py              ← Reusable Plotly chart functions
├── sidebar.py                  ← Sidebar navigation & filter widgets
├── export_model.py             ← ⚠️ Run this ONCE to save your trained model
├── requirements.txt
├── .streamlit/
│   └── config.toml             ← Theme configuration
├── pages/
│   ├── 01_executive_overview.py
│   ├── 02_churn_analysis.py
│   ├── 03_model_performance.py
│   ├── 04_customer_prediction.py
│   └── 05_business_insights.py
└── model/                      ← Created by export_model.py
    ├── gradient_boosting_pipeline.joblib
    └── test_results.joblib
```

---

## 🚀 Setup Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> On Python 3.15 managed by uv, use:
> ```bash
> python3.15 -m pip install --break-system-packages -r requirements.txt
> ```

### Step 2 — Copy your dataset

Place `Telco_customer_churn.xlsx` in the `churn_dashboard/` folder.  
*(Or leave it at `d:\Projects\customer churn prediction\` — the app will find it automatically.)*

### Step 3 — Export your trained model *(required for live predictions)*

You have two options:

**Option A — Run from your existing Jupyter/Colab environment:**
1. Open `export_model.py`
2. Confirm `EXCEL_PATH` at the top points to your Excel file
3. Run the script — it will retrain and save:
   - `model/gradient_boosting_pipeline.joblib`
   - `model/test_results.joblib`
4. Copy the `model/` folder into `churn_dashboard/`

**Option B — Run directly (if scikit-learn is available):**
```bash
cd churn_dashboard
python export_model.py
```

> ⚠️ **Without the model file:** The dashboard still works fully — EDA, charts, and analysis pages will display correctly. Only the prediction page and live model charts require the saved model.

### Step 4 — Launch the dashboard

```bash
cd churn_dashboard
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Executive Overview | KPI cards, churn distribution, segment breakdowns, global filters |
| 🔍 Churn Analysis | Detailed EDA across demographics, services, billing, tenure |
| 🤖 Model Performance | Confusion matrix, ROC/PR curves, feature importance, model comparison |
| 🔮 Customer Prediction | Interactive form → live Gradient Boosting prediction with gauge |
| 💡 Business Insights | Key patterns, recommendations, estimated business impact |

---

## 🤖 Model Details

| Setting | Value |
|---------|-------|
| Algorithm | Gradient Boosting Classifier |
| Tuning | GridSearchCV (5-fold stratified CV) |
| Threshold | **0.33** (chosen for high recall) |
| Accuracy | 77.57% |
| Precision | 55.87% |
| Recall | **73.80%** |
| F1-Score | 63.59% |
| ROC-AUC | 85.52% |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web app
- **Plotly** — interactive charts
- **scikit-learn** — ML model & preprocessing
- **Pandas / NumPy** — data processing
- **Joblib** — model serialisation
- **openpyxl** — Excel file reading

