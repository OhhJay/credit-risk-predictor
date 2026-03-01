import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def _low_risk_payload():
    """Provider with strong financials and clean payment history."""
    return {
        "AMT_INCOME_TOTAL": 300000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,
        "DAYS_EMPLOYED": -3000,
        "DAYS_BIRTH": -18000,
        "DAYS_REGISTRATION": -6000,
        "EXT_SOURCE_1": 0.7,
        "EXT_SOURCE_2": 0.75,
        "EXT_SOURCE_3": 0.7,
        "REGION_RATING_CLIENT": 1,
        "CREDIT_TO_INCOME": 1.67,
        "ANNUITY_TO_INCOME": 0.08,
        "CREDIT_TO_GOODS": 1.11,
        "avg_days_past_due": -12.0,
        "avg_payment_ratio": 1.0,
        "late_payment_rate": 0.01,
        "severe_late_rate": 0.0,
        "total_underpayment": 0.0,
        "std_days_past_due": 5.0,
        "std_payment_ratio": 0.0,
        "max_days_past_due": 0,
        "min_payment_ratio": 1.0,
        "total_installments": 60,
        "num_contracts": 4,
        "late_rate_trend": 0.0,
        "payment_ratio_trend": 0.0,
        "recent_late_rate": 0.0,
        "recent_avg_days_late": -10.0,
        "recent_min_payment_ratio": 1.0,
        "num_bureau_records": 6,
        "num_active_loans": 1,
        "num_closed_loans": 5,
        "total_external_debt": 50000,
        "total_external_credit": 800000,
        "total_overdue_amount": 0,
        "max_days_overdue": 0,
        "has_any_overdue": 0,
        "num_credit_types": 2,
        "active_loan_ratio": 0.17,
        "external_utilization": 0.06,
        "pos_months_count": 30,
        "pos_dpd_max": 0,
        "pos_dpd_mean": 0.0,
        "pos_has_dpd": 0,
        "pos_completed_count": 4,
        "pos_active_count": 0,
        "cc_months_count": 24,
        "cc_avg_balance": 20000,
        "cc_max_balance": 50000,
        "cc_avg_credit_limit": 300000,
        "cc_avg_drawings": 15000,
        "cc_dpd_max": 0,
        "cc_has_dpd": 0,
        "cc_min_payment_ratio": 12.0,
        "cc_utilization": 0.07,
        "prev_app_count": 4,
        "prev_approved": 3,
        "prev_refused": 0,
        "prev_avg_credit": 400000,
        "prev_refusal_rate": 0.0,
    }


def _high_risk_payload():
    """Provider with poor financials and bad payment history."""
    return {
        "AMT_INCOME_TOTAL": 80000,
        "AMT_CREDIT": 800000,
        "AMT_ANNUITY": 40000,
        "AMT_GOODS_PRICE": 700000,
        "DAYS_EMPLOYED": -300,
        "DAYS_BIRTH": -10000,
        "DAYS_REGISTRATION": -500,
        "EXT_SOURCE_1": 0.15,
        "EXT_SOURCE_2": 0.2,
        "EXT_SOURCE_3": 0.1,
        "REGION_RATING_CLIENT": 3,
        "CREDIT_TO_INCOME": 10.0,
        "ANNUITY_TO_INCOME": 0.5,
        "CREDIT_TO_GOODS": 1.14,
        "avg_days_past_due": 15.0,
        "avg_payment_ratio": 0.7,
        "late_payment_rate": 0.4,
        "severe_late_rate": 0.15,
        "total_underpayment": 50000,
        "std_days_past_due": 25.0,
        "std_payment_ratio": 0.5,
        "max_days_past_due": 60,
        "min_payment_ratio": 0.2,
        "total_installments": 8,
        "num_contracts": 1,
        "late_rate_trend": 0.2,
        "payment_ratio_trend": -0.3,
        "recent_late_rate": 0.5,
        "recent_avg_days_late": 20.0,
        "recent_min_payment_ratio": 0.3,
        "num_bureau_records": 10,
        "num_active_loans": 7,
        "num_closed_loans": 3,
        "total_external_debt": 600000,
        "total_external_credit": 700000,
        "total_overdue_amount": 50000,
        "max_days_overdue": 90,
        "has_any_overdue": 1,
        "num_credit_types": 4,
        "active_loan_ratio": 0.7,
        "external_utilization": 0.86,
        "pos_months_count": 5,
        "pos_dpd_max": 30,
        "pos_dpd_mean": 5.0,
        "pos_has_dpd": 1,
        "pos_completed_count": 0,
        "pos_active_count": 2,
        "cc_months_count": 6,
        "cc_avg_balance": 180000,
        "cc_max_balance": 200000,
        "cc_avg_credit_limit": 200000,
        "cc_avg_drawings": 50000,
        "cc_dpd_max": 15,
        "cc_has_dpd": 1,
        "cc_min_payment_ratio": 2.0,
        "cc_utilization": 0.9,
        "prev_app_count": 6,
        "prev_approved": 1,
        "prev_refused": 4,
        "prev_avg_credit": 200000,
        "prev_refusal_rate": 0.67,
    }


def test_score_low_risk_provider():
    """Provider with good metrics should get a lower risk score."""
    response = client.post("/api/v1/score", json=_low_risk_payload())
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_category" in data
    assert "feature_importance" in data
    assert data["risk_score"] < 0.5


def test_score_high_risk_provider():
    """Provider with poor metrics should get a higher risk score."""
    response = client.post("/api/v1/score", json=_high_risk_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] > 0.3


def test_score_minimal_features():
    """Only required fields — everything else defaults to 0."""
    payload = {
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 500000,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data


def test_batch_scoring():
    """Batch endpoint should return scores for all providers."""
    payload = {
        "providers": [_low_risk_payload(), _high_risk_payload()]
    }
    response = client.post("/api/v1/score/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    assert len(data["results"]) == 2


def test_high_risk_scores_higher_than_low_risk():
    """High risk provider should score higher than low risk provider."""
    low_resp = client.post("/api/v1/score", json=_low_risk_payload())
    high_resp = client.post("/api/v1/score", json=_high_risk_payload())

    low_score = low_resp.json()["risk_score"]
    high_score = high_resp.json()["risk_score"]

    assert high_score > low_score, (
        f"High risk ({high_score}) should score higher than low risk ({low_score})"
    )