from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.schemas import (
    ScoringRequest,
    ScoringResponse,
    BatchScoringRequest,
    BatchScoringResponse,
)

router = APIRouter()


@router.post("/score", response_model=ScoringResponse)
async def score_provider(request: Request, scoring_input: ScoringRequest):
    """
    Score a provider's creditworthiness using pre-computed features.

    Pass all 61 features directly (application profile, payment behavior,
    bureau data, POS/cash, credit card, and previous application history)
    and receive a real-time risk assessment with SHAP-based feature explanations.

    Features with default values can be omitted if data is unavailable —
    they default to 0 (or sensible defaults) and the model handles missing sources.
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