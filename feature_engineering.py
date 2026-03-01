import pandas as pd
import numpy as np
from typing import Dict, List
from app.schemas.schemas import ScoringRequest


# Feature names expected by the model (order matters)
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
    # Derived features
    "repayment_ratio",
    "revenue_per_transaction",
    "balance_to_revenue_ratio",
]


def extract_features(request: ScoringRequest) -> pd.DataFrame:
    """Convert a scoring request into a feature DataFrame for model prediction."""

    # Derived features
    repayment_ratio = (
        request.total_repayment_amount / request.total_purchase_amount
        if request.total_purchase_amount > 0
        else 0.0
    )
    revenue_per_transaction = (
        request.annual_revenue / request.total_transactions
        if request.total_transactions > 0
        else 0.0
    )
    balance_to_revenue_ratio = (
        request.outstanding_balance / request.annual_revenue
        if request.annual_revenue > 0
        else 0.0
    )

    features = {
        "total_transactions": request.total_transactions,
        "total_purchase_amount": request.total_purchase_amount,
        "total_repayment_amount": request.total_repayment_amount,
        "avg_days_to_repay": request.avg_days_to_repay,
        "default_rate": request.default_rate,
        "transaction_frequency_monthly": request.transaction_frequency_monthly,
        "credit_utilization_ratio": request.credit_utilization_ratio,
        "years_in_operation": request.years_in_operation,
        "annual_revenue": request.annual_revenue,
        "outstanding_balance": request.outstanding_balance,
        "repayment_ratio": repayment_ratio,
        "revenue_per_transaction": revenue_per_transaction,
        "balance_to_revenue_ratio": balance_to_revenue_ratio,
    }

    return pd.DataFrame([features], columns=FEATURE_NAMES)


def extract_features_from_transactions(
    transactions: List[Dict], provider_info: Dict
) -> pd.DataFrame:
    """Extract features from raw transaction history stored in DB."""

    if not transactions:
        raise ValueError("No transactions found for this provider")

    df = pd.DataFrame(transactions)

    purchases = df[df["transaction_type"] == "purchase"]
    repayments = df[df["transaction_type"] == "repayment"]

    total_purchase_amount = purchases["amount"].sum() if len(purchases) > 0 else 0
    total_repayment_amount = repayments["amount"].sum() if len(repayments) > 0 else 0

    # Calculate average days to repay for settled transactions
    settled = purchases[purchases["settled_date"].notna() & purchases["due_date"].notna()]
    if len(settled) > 0:
        settled_dates = pd.to_datetime(settled["settled_date"])
        due_dates = pd.to_datetime(settled["due_date"])
        avg_days_to_repay = (settled_dates - due_dates).dt.days.mean()
    else:
        avg_days_to_repay = 0.0

    # Default rate
    defaulted = len(purchases[purchases["status"] == "defaulted"])
    default_rate = defaulted / len(purchases) if len(purchases) > 0 else 0.0

    # Transaction frequency (monthly)
    if len(df) > 1:
        dates = pd.to_datetime(df["transaction_date"])
        months_span = max((dates.max() - dates.min()).days / 30, 1)
        transaction_frequency_monthly = len(df) / months_span
    else:
        transaction_frequency_monthly = 1.0

    # Credit utilization
    outstanding = purchases[purchases["status"].isin(["pending", "overdue"])]["amount"].sum()
    credit_limit_estimate = total_purchase_amount * 1.2 if total_purchase_amount > 0 else 1
    credit_utilization_ratio = min(outstanding / credit_limit_estimate, 1.0)

    request = ScoringRequest(
        total_transactions=len(df),
        total_purchase_amount=total_purchase_amount,
        total_repayment_amount=total_repayment_amount,
        avg_days_to_repay=max(avg_days_to_repay, 0),
        default_rate=default_rate,
        transaction_frequency_monthly=transaction_frequency_monthly,
        credit_utilization_ratio=credit_utilization_ratio,
        years_in_operation=provider_info.get("years_in_operation", 0),
        annual_revenue=provider_info.get("annual_revenue", 0),
        outstanding_balance=outstanding,
    )

    return extract_features(request)
