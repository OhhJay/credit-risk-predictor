# Credit Risk Model Training — Google Colab Notebook
# ================================================
# Copy this into a Colab notebook cell-by-cell.
# After training, download model.joblib and explainer.joblib
# and place them in the ml/ directory of the FastAPI project.

# %% Cell 1: Install dependencies
# !pip install xgboost shap scikit-learn pandas numpy joblib matplotlib seaborn

# %% Cell 2: Generate synthetic healthcare provider credit data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N_SAMPLES = 5000

def generate_synthetic_data(n=N_SAMPLES):
    """Generate synthetic healthcare provider credit data."""
    data = {
        "total_transactions": np.random.randint(5, 500, n),
        "total_purchase_amount": np.random.uniform(50_000, 10_000_000, n),
        "avg_days_to_repay": np.random.uniform(1, 90, n),
        "default_rate": np.random.beta(1, 8, n),  # skewed low (most providers are ok)
        "transaction_frequency_monthly": np.random.uniform(0.5, 30, n),
        "credit_utilization_ratio": np.random.beta(2, 3, n),
        "years_in_operation": np.random.randint(0, 25, n),
        "annual_revenue": np.random.uniform(100_000, 50_000_000, n),
    }

    df = pd.DataFrame(data)

    # Derived: repayment amount (correlated with purchase amount and default rate)
    repay_factor = 1 - df["default_rate"] * np.random.uniform(0.5, 1.5, n)
    df["total_repayment_amount"] = df["total_purchase_amount"] * np.clip(repay_factor, 0.3, 1.0)

    # Outstanding balance
    df["outstanding_balance"] = (
        df["total_purchase_amount"] - df["total_repayment_amount"]
    ) * np.random.uniform(0.1, 1.0, n)

    # Derived features
    df["repayment_ratio"] = df["total_repayment_amount"] / df["total_purchase_amount"]
    df["revenue_per_transaction"] = df["annual_revenue"] / df["total_transactions"]
    df["balance_to_revenue_ratio"] = df["outstanding_balance"] / df["annual_revenue"]

    # Target: credit risk (binary — 1 = high risk, 0 = low/acceptable risk)
    risk_score = (
        0.30 * df["default_rate"]
        + 0.20 * df["credit_utilization_ratio"]
        + 0.15 * np.clip(df["avg_days_to_repay"] / 90, 0, 1)
        + 0.15 * (1 - df["repayment_ratio"])
        + 0.10 * np.clip(1 - df["years_in_operation"] / 10, 0, 1)
        + 0.10 * np.clip(df["balance_to_revenue_ratio"], 0, 1)
    )
    # Add noise and binarize
    risk_score += np.random.normal(0, 0.05, n)
    df["is_high_risk"] = (risk_score > 0.45).astype(int)

    return df

df = generate_synthetic_data()
print(f"Dataset shape: {df.shape}")
print(f"Risk distribution:\n{df['is_high_risk'].value_counts(normalize=True)}")
df.head()


# %% Cell 3: Train-test split and feature setup
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "total_transactions",
    "total_purchase_amount",
    "total_repayment_amount",
    "avg_days_to_repay",
    "default_rate",
    "transaction_frequency_monthly",
    "credit_utilization_ratio",
    "years_in_operation",
    "annual_revenue",
    "outstanding_balance",
    "repayment_ratio",
    "revenue_per_transaction",
    "balance_to_revenue_ratio",
]

X = df[FEATURE_NAMES]
y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# %% Cell 4: Train and compare models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model": model, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}
    print(f"\n{'='*50}")
    print(f"{name} — ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    RocCurveDisplay.from_predictions(y_test, res["y_proba"], name=name, ax=ax)
plt.title("Model Comparison — ROC Curves")
plt.grid(alpha=0.3)
plt.show()


# %% Cell 5: SHAP explainability for best model
import shap

# Select best model (likely XGBoost)
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")

# SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES)

# Single prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], feature_names=FEATURE_NAMES)


# %% Cell 6: Save model and explainer
import joblib

joblib.dump(best_model, "model.joblib")
joblib.dump(explainer, "explainer.joblib")
print("Model and explainer saved!")
print("Download both files and place them in the ml/ directory of your FastAPI project.")

# In Colab, download:
# from google.colab import files
# files.download("model.joblib")
# files.download("explainer.joblib")
