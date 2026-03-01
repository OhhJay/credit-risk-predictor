from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from uuid import UUID


# ── Provider Schemas ──────────────────────────────────────────────

class ProviderCreate(BaseModel):
    name: str = Field(..., example="MedSupply Corp")
    business_type: str = Field(..., example="medical_equipment")
    registration_number: str = Field(..., example="RC-123456")
    years_in_operation: int = Field(..., ge=0, example=5)
    annual_revenue: Optional[float] = Field(None, example=2500000.00)


class ProviderResponse(BaseModel):
    id: UUID
    name: str
    business_type: str
    registration_number: str
    years_in_operation: int
    annual_revenue: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Transaction Schemas ───────────────────────────────────────────

class TransactionCreate(BaseModel):
    provider_id: UUID
    amount: float = Field(..., gt=0, example=50000.00)
    transaction_type: str = Field(..., pattern="^(purchase|repayment)$", example="purchase")
    transaction_date: datetime
    due_date: Optional[datetime] = None
    settled_date: Optional[datetime] = None
    status: str = Field(default="pending", pattern="^(pending|settled|overdue|defaulted)$")


class TransactionResponse(BaseModel):
    id: UUID
    provider_id: UUID
    amount: float
    transaction_type: str
    transaction_date: datetime
    due_date: Optional[datetime]
    settled_date: Optional[datetime]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Scoring Schemas ───────────────────────────────────────────────

class ScoringRequest(BaseModel):
    """Direct scoring without DB lookup — pass features directly."""
    total_transactions: int = Field(..., ge=0, example=120)
    total_purchase_amount: float = Field(..., ge=0, example=5000000.00)
    total_repayment_amount: float = Field(..., ge=0, example=4800000.00)
    avg_days_to_repay: float = Field(..., ge=0, example=15.5)
    default_rate: float = Field(..., ge=0, le=1, example=0.05)
    transaction_frequency_monthly: float = Field(..., ge=0, example=10.0)
    credit_utilization_ratio: float = Field(..., ge=0, le=1, example=0.65)
    years_in_operation: int = Field(..., ge=0, example=5)
    annual_revenue: float = Field(..., ge=0, example=2500000.00)
    outstanding_balance: float = Field(..., ge=0, example=200000.00)


class ScoringResponse(BaseModel):
    risk_score: float = Field(..., description="Risk score between 0 (low risk) and 1 (high risk)")
    risk_category: str = Field(..., description="low, medium, or high")
    model_version: str
    feature_importance: Dict[str, float] = Field(
        ..., description="SHAP-based feature importance for this prediction"
    )
    scored_at: datetime


class BatchScoringRequest(BaseModel):
    providers: List[ScoringRequest]


class BatchScoringResponse(BaseModel):
    results: List[ScoringResponse]
    total_scored: int


class ProviderScoringRequest(BaseModel):
    """Score a provider using their DB transaction history."""
    provider_id: UUID
