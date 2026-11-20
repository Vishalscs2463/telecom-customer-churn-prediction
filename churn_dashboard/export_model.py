"""
export_model.py
---------------
Run this script ONCE from the same environment where you trained your model
(i.e., your Jupyter / Colab environment that has sklearn, pandas, etc.).

It will:
  1. Load your dataset
  2. Reproduce your exact preprocessing and GradientBoosting pipeline
  3. Save the trained pipeline to model/gradient_boosting_pipeline.joblib
  4. Save the test-set predictions/probabilities to model/test_results.joblib

Usage (from your notebook or terminal):
    python export_model.py

Make sure the Excel file path below matches where your dataset is located.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

# ── 1. Load & clean data ────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR, "data", "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "..", "Telco_customer_churn.xlsx"),
    os.path.join(_THIS_DIR, "..", "data", "Telco_customer_churn.xlsx"),
    r"d:\Projects\customer churn prediction\churn_dashboard\data\Telco_customer_churn.xlsx",
    r"d:\Projects\customer churn prediction\Telco_customer_churn.xlsx",
]
EXCEL_PATH = next((p for p in _CANDIDATES if os.path.isfile(p)), _CANDIDATES[0])

print(f"Loading dataset from: {EXCEL_PATH} …")
df = pd.read_excel(EXCEL_PATH)

df_clean = df.copy()
df_clean["Total Charges"] = df_clean["Total Charges"].replace("", pd.NA)
df_clean["Total Charges"] = pd.to_numeric(df_clean["Total Charges"], errors="coerce")
df_clean["Total Charges"] = df_clean["Total Charges"].fillna(0)
# Note: "Senior Citizen" in this dataset is stored as "Yes"/"No" strings — keep as-is.

# ── 2. Build feature matrix ─────────────────────────────────────────────────
drop_columns = [
    "CustomerID", "Count", "Churn Label", "Churn Value",
    "Churn Score", "Churn Reason", "CLTV"
]
geo_columns = [
    "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude"
]

model_df = df_clean.drop(columns=drop_columns + geo_columns)

X = model_df
y = df_clean["Churn Label"].map({"No": 0, "Yes": 1})

# ── 3. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# ── 4. Build preprocessor (passthrough num + OHE cat, same as your notebook) ─
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ]
)

# ── 5. Build & tune Gradient Boosting pipeline ───────────────────────────────
gradient_boosting = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)

gradient_boosting_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor_tree), ("model", gradient_boosting)]
)

gb_param_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
}

gb_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running GridSearchCV for Gradient Boosting (this may take a few minutes) …")
gb_grid_search = GridSearchCV(
    estimator=gradient_boosting_pipeline,
    param_grid=gb_param_grid,
    scoring="f1",
    cv=gb_cv,
    n_jobs=-1,
    verbose=1,
)
gb_grid_search.fit(X_train, y_train)

best_gb = gb_grid_search.best_estimator_
print("Best parameters:", gb_grid_search.best_params_)

# ── 6. Evaluate at threshold 0.33 ────────────────────────────────────────────
y_prob_gb_tuned = best_gb.predict_proba(X_test)[:, 1]

final_threshold = 0.33
y_pred_gb_final = (y_prob_gb_tuned >= final_threshold).astype(int)

print("\n── Final Model @ threshold 0.33 ──")
print("Accuracy :", round(accuracy_score(y_test, y_pred_gb_final), 4))
print("Precision:", round(precision_score(y_test, y_pred_gb_final), 4))
print("Recall   :", round(recall_score(y_test, y_pred_gb_final), 4))
print("F1       :", round(f1_score(y_test, y_pred_gb_final), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob_gb_tuned), 4))

# ── 7. ROC & PR curves ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob_gb_tuned)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob_gb_tuned)
cm = confusion_matrix(y_test, y_pred_gb_final)

# ── 8. Feature importance ────────────────────────────────────────────────────
preprocessor_fitted = best_gb.named_steps["preprocessor"]
gb_model = best_gb.named_steps["model"]
feature_names = preprocessor_fitted.get_feature_names_out()

feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": gb_model.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

# Permutation importance (original feature names)
perm_result = permutation_importance(
    best_gb, X_test, y_test, scoring="f1",
    n_repeats=10, random_state=42, n_jobs=-1
)
perm_imp_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance Mean": perm_result.importances_mean,
    "Importance Std": perm_result.importances_std,
}).sort_values("Importance Mean", ascending=False).reset_index(drop=True)

# ── 9. Save everything ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(best_gb, "model/gradient_boosting_pipeline.joblib")
print("\nSaved model to model/gradient_boosting_pipeline.joblib")

test_results = {
    "y_test": y_test,
    "y_pred": y_pred_gb_final,
    "y_prob": y_prob_gb_tuned,
    "cm": cm,
    "fpr": fpr,
    "tpr": tpr,
    "prec_curve": prec_curve,
    "rec_curve": rec_curve,
    "feat_imp_df": feat_imp_df,
    "perm_imp_df": perm_imp_df,
    "X_train_columns": X_train.columns.tolist(),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "threshold": final_threshold,
}
joblib.dump(test_results, "model/test_results.joblib")
print("Saved test results to model/test_results.joblib")

# Save a small sample of training data for dashboard reference
X_train.head(5).to_csv("model/sample_input.csv", index=False)
print("Saved sample_input.csv")

print("\n✅ Export complete! Copy the 'model/' folder into your churn_dashboard/ directory.")
