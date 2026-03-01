import pandas as pd
import numpy as np
from typing import Dict, List
from app.schemas.schemas import ScoringRequest


# Feature names expected by the model — must match training order exactly
FEATURE_NAMES = [
    # Application features
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    # Derived application ratios
    "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS",
    # Payment behavior (installments)
    "avg_days_past_due", "avg_payment_ratio", "late_payment_rate",
    "severe_late_rate", "total_underpayment", "std_days_past_due",
    "std_payment_ratio", "max_days_past_due", "min_payment_ratio",
    "total_installments", "num_contracts",
    # Trends
    "late_rate_trend", "payment_ratio_trend",
    "recent_late_rate", "recent_avg_days_late", "recent_min_payment_ratio",
    # Bureau (external credit)
    "num_bureau_records", "num_active_loans", "num_closed_loans",
    "total_external_debt", "total_external_credit",
    "total_overdue_amount", "max_days_overdue", "has_any_overdue",
    "num_credit_types", "active_loan_ratio", "external_utilization",
    # POS/Cash balance
    "pos_months_count", "pos_dpd_max", "pos_dpd_mean",
    "pos_has_dpd", "pos_completed_count", "pos_active_count",
    # Credit card
    "cc_months_count", "cc_avg_balance", "cc_max_balance",
    "cc_avg_credit_limit", "cc_avg_drawings", "cc_dpd_max",
    "cc_has_dpd", "cc_min_payment_ratio", "cc_utilization",
    # Previous applications
    "prev_app_count", "prev_approved", "prev_refused",
    "prev_avg_credit", "prev_refusal_rate",
]


def extract_features(request: ScoringRequest) -> pd.DataFrame:
    """
    Convert a ScoringRequest into a feature DataFrame for model prediction.
    Maps all 61 fields from the request to the model's expected feature order.
    """
    features = {name: getattr(request, name, 0) for name in FEATURE_NAMES}
    return pd.DataFrame([features], columns=FEATURE_NAMES)