from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.schemas import (
    ScoringRequest,
    ScoringResponse,
    BatchScoringRequest,
    BatchScoringResponse,
    ProviderScoringRequest,
)
from app.models.models import Provider, Transaction, RiskScore
from app.services.feature_engineering import extract_features_from_transactions

router = APIRouter()


@router.post("/score", response_model=ScoringResponse)
async def score_provider(request: Request, scoring_input: ScoringRequest):
    """
    Score a healthcare provider's creditworthiness using pre-computed features.

    Pass business metrics directly and receive a real-time risk assessment
    with SHAP-based feature explanations.
    """
    model_service = request.app.state.model_service
    return model_service.predict(scoring_input)


@router.post("/score/batch", response_model=BatchScoringResponse)
async def batch_score_providers(request: Request, batch_input: BatchScoringRequest):
    """
    Score multiple providers in a single request.

    Returns risk scores for all providers with individual feature explanations.
    """
    model_service = request.app.state.model_service
    results = [model_service.predict(provider) for provider in batch_input.providers]

    return BatchScoringResponse(results=results, total_scored=len(results))


@router.post("/score/provider", response_model=ScoringResponse)
async def score_from_history(
    request: Request,
    scoring_input: ProviderScoringRequest,
    db: Session = Depends(get_db),
):
    """
    Score a provider using their stored transaction history.

    Automatically computes features from the provider's transaction records
    in the database and generates a risk assessment.
    """
    # Fetch provider
    provider = db.query(Provider).filter(Provider.id == scoring_input.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Fetch transactions
    transactions = (
        db.query(Transaction)
        .filter(Transaction.provider_id == scoring_input.provider_id)
        .all()
    )
    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="No transactions found for this provider. Cannot generate score.",
        )

    # Convert to dicts for feature engineering
    tx_dicts = [
        {
            "amount": t.amount,
            "transaction_type": t.transaction_type,
            "transaction_date": t.transaction_date,
            "due_date": t.due_date,
            "settled_date": t.settled_date,
            "status": t.status,
        }
        for t in transactions
    ]

    provider_info = {
        "years_in_operation": provider.years_in_operation,
        "annual_revenue": provider.annual_revenue or 0,
    }

    # Extract features and score
    features_df = extract_features_from_transactions(tx_dicts, provider_info)
    model_service = request.app.state.model_service

    # Build a ScoringRequest from the computed features
    from app.schemas.schemas import ScoringRequest

    row = features_df.iloc[0]
    scoring_request = ScoringRequest(
        total_transactions=int(row["total_transactions"]),
        total_purchase_amount=float(row["total_purchase_amount"]),
        total_repayment_amount=float(row["total_repayment_amount"]),
        avg_days_to_repay=float(row["avg_days_to_repay"]),
        default_rate=float(row["default_rate"]),
        transaction_frequency_monthly=float(row["transaction_frequency_monthly"]),
        credit_utilization_ratio=float(row["credit_utilization_ratio"]),
        years_in_operation=int(row["years_in_operation"]),
        annual_revenue=float(row["annual_revenue"]),
        outstanding_balance=float(row["outstanding_balance"]),
    )

    result = model_service.predict(scoring_request)

    # Save score to DB
    import json
    risk_score = RiskScore(
        provider_id=scoring_input.provider_id,
        score=result.risk_score,
        risk_category=result.risk_category,
        model_version=result.model_version,
        feature_importance=json.dumps(result.feature_importance),
    )
    db.add(risk_score)
    db.commit()

    return result
