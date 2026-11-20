# 📡 Telecom Customer Churn Prediction

> An end-to-end Machine Learning project with an interactive **5-page Streamlit dashboard** for analysing and predicting customer churn in the telecom industry.

---

## 🔗 Live Demo

> _Deploy on Streamlit Cloud to get a live link — https://telecom-customer-churn-prediction-glpauttdpthspgnktyzgbe.streamlit.app/

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project uses the **IBM Telco Customer Churn dataset** to:

- Explore churn patterns across customer segments
- Train a **Gradient Boosting Classifier** (tuned with GridSearchCV)
- Deploy a live prediction dashboard with business insights

The model uses a custom threshold of **0.33** (instead of default 0.50) to maximise **Recall** — ensuring more at-risk customers are caught early.

---

## 🤖 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 77.57% |
| **Precision** | 55.87% |
| **Recall** | 73.80% |
| **F1-Score** | 63.59% |
| **ROC-AUC** | 85.52% |

> Model: **Tuned Gradient Boosting** · Threshold: **0.33** · CV: **5-Fold StratifiedKFold**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Executive Overview | KPI cards, churn rate, revenue impact |
| 🔍 Churn Analysis | Segment breakdowns by contract, tenure, services |
| 🤖 Model Performance | Confusion matrix, ROC curve, feature importance |
| 🎯 Customer Prediction | Live churn probability for a single customer |
| 💡 Business Insights | Retention strategies & recommendations |

---

## 🗂️ Project Structure

```
telecom-customer-churn-prediction/
│
├── churn_dashboard/
│   ├── app.py                   # Main Streamlit entry point
│   ├── data_loader.py           # Dataset loading with caching
│   ├── model_loader.py          # ML model loading & prediction
│   ├── sidebar.py               # Navigation & filter widgets
│   ├── chart_utils.py           # Reusable Plotly chart functions
│   ├── export_model.py          # Script to train & save the model
│   │
│   ├── pages/
│   │   ├── 01_executive_overview.py
│   │   ├── 02_churn_analysis.py
│   │   ├── 03_model_performance.py
│   │   ├── 04_customer_prediction.py
│   │   └── 05_business_insights.py
│   │
│   ├── data/
│   │   └── Telco_customer_churn.xlsx   # IBM dataset
│   │
│   ├── model/
│   │   ├── gradient_boosting_pipeline.joblib
│   │   └── test_results.joblib
│   │
│   ├── .streamlit/
│   │   └── config.toml
│   │
│   ├── requirements.txt
│   └── run_dashboard.bat        # One-click launcher (Windows)
│
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Dashboard** | Streamlit 1.67 |
| **ML** | scikit-learn 1.9 (GradientBoostingClassifier, GridSearchCV) |
| **Data** | pandas 3.x, numpy 2.x |
| **Visualisation** | Plotly 7.x |
| **Dataset** | IBM Telco Customer Churn (7,043 customers, 33 features) |

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vishalscs2463/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r churn_dashboard/requirements.txt
```

### 3. Train & export the model (first time only)
```bash
cd churn_dashboard
python export_model.py
```

### 4. Run the dashboard
```bash
streamlit run app.py
```
> Or on Windows: double-click **`run_dashboard.bat`**

Dashboard opens at → **http://localhost:8501**

---

## ☁️ Deployment

### Streamlit Community Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"** → Connect your GitHub account
3. Select this repo: `Vishalscs2463/telecom-customer-churn-prediction`
4. Set **Main file path**: `churn_dashboard/app.py`
5. Click **Deploy** 🚀

---

## 📁 Dataset

- **Source:** [IBM Sample Data — Telco Customer Churn](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- **Rows:** 7,043 customers
- **Features:** 33 columns (demographics, services, contract, billing)
- **Target:** `Churn Label` (Yes / No)

---

## 🔑 Key Findings

- **Month-to-month contracts** have the highest churn rate (~43%)
- **Fiber optic internet** users churn more than DSL users
- **Tenure < 12 months** is the strongest churn indicator
- Customers with **no tech support / online security** churn 2× more
- **Electronic check** payment users churn more than auto-pay users

---

## 👤 Author

**Vishal** · [@Vishalscs2463](https://github.com/Vishalscs2463)

---

## 📄 License

This project is for educational purposes. Dataset © IBM.
