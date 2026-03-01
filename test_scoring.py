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


def test_score_low_risk_provider():
    """Provider with good metrics should get a low risk score."""
    payload = {
        "total_transactions": 200,
        "total_purchase_amount": 5000000,
        "total_repayment_amount": 4950000,
        "avg_days_to_repay": 10,
        "default_rate": 0.01,
        "transaction_frequency_monthly": 15,
        "credit_utilization_ratio": 0.2,
        "years_in_operation": 8,
        "annual_revenue": 10000000,
        "outstanding_balance": 50000,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_category"] == "low"
    assert data["risk_score"] < 0.3
    assert "feature_importance" in data


def test_score_high_risk_provider():
    """Provider with poor metrics should get a high risk score."""
    payload = {
        "total_transactions": 10,
        "total_purchase_amount": 500000,
        "total_repayment_amount": 200000,
        "avg_days_to_repay": 60,
        "default_rate": 0.4,
        "transaction_frequency_monthly": 1,
        "credit_utilization_ratio": 0.9,
        "years_in_operation": 1,
        "annual_revenue": 300000,
        "outstanding_balance": 250000,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_category"] == "high"
    assert data["risk_score"] > 0.7


def test_batch_scoring():
    """Batch endpoint should return scores for all providers."""
    payload = {
        "providers": [
            {
                "total_transactions": 100,
                "total_purchase_amount": 2000000,
                "total_repayment_amount": 1900000,
                "avg_days_to_repay": 12,
                "default_rate": 0.02,
                "transaction_frequency_monthly": 8,
                "credit_utilization_ratio": 0.3,
                "years_in_operation": 5,
                "annual_revenue": 5000000,
                "outstanding_balance": 100000,
            },
            {
                "total_transactions": 20,
                "total_purchase_amount": 800000,
                "total_repayment_amount": 400000,
                "avg_days_to_repay": 45,
                "default_rate": 0.25,
                "transaction_frequency_monthly": 2,
                "credit_utilization_ratio": 0.75,
                "years_in_operation": 2,
                "annual_revenue": 1000000,
                "outstanding_balance": 350000,
            },
        ]
    }
    response = client.post("/api/v1/score/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    assert len(data["results"]) == 2


def test_score_validation_error():
    """Invalid input should return 422."""
    payload = {
        "total_transactions": -1,  # invalid
        "total_purchase_amount": 5000000,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 422
