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
    """
    Direct scoring — pass pre-computed features for real-time risk assessment.

    Features are grouped by source:
    - Application: provider profile and financial info
    - Payment behavior: aggregated from installment/transaction history
    - Trends: trajectory of payment behavior over time
    - Bureau: external credit information from other lenders
    - POS/Cash: point-of-sale balance features
    - Credit card: credit card utilization features
    - Previous applications: past credit application history
    """

    # ── Application features (provider profile) ──────────
    AMT_INCOME_TOTAL: float = Field(..., ge=0, example=150000.0, description="Annual revenue / income")
    AMT_CREDIT: float = Field(..., ge=0, example=500000.0, description="Total credit amount")
    AMT_ANNUITY: float = Field(0, ge=0, example=25000.0, description="Periodic payment amount")
    AMT_GOODS_PRICE: float = Field(0, ge=0, example=450000.0, description="Price of goods / services")
    DAYS_EMPLOYED: float = Field(0, example=-1200, description="Days employed (negative = past)")
    DAYS_BIRTH: float = Field(0, example=-15000, description="Days since founded (negative = past)")
    DAYS_REGISTRATION: float = Field(0, example=-5000, description="Days since registration")
    EXT_SOURCE_1: float = Field(0, ge=0, le=1, example=0.5, description="External credit score 1")
    EXT_SOURCE_2: float = Field(0, ge=0, le=1, example=0.55, description="External credit score 2")
    EXT_SOURCE_3: float = Field(0, ge=0, le=1, example=0.5, description="External credit score 3")
    REGION_RATING_CLIENT: int = Field(2, ge=1, le=3, example=2, description="Region risk rating (1-3)")

    # ── Derived application ratios ────────────────────────
    CREDIT_TO_INCOME: float = Field(0, ge=0, example=3.3, description="Credit / Income ratio")
    ANNUITY_TO_INCOME: float = Field(0, ge=0, example=0.17, description="Annuity / Income ratio")
    CREDIT_TO_GOODS: float = Field(0, ge=0, example=1.1, description="Credit / Goods price ratio")

    # ── Payment behavior (from installments) ──────────────
    avg_days_past_due: float = Field(0, example=-8.0, description="Avg days past due (negative=early)")
    avg_payment_ratio: float = Field(1.0, ge=0, example=1.0, description="Median payment ratio")
    late_payment_rate: float = Field(0, ge=0, le=1, example=0.03, description="Fraction of late payments")
    severe_late_rate: float = Field(0, ge=0, le=1, example=0.0, description="Fraction >30 days late")
    total_underpayment: float = Field(0, ge=0, example=0.0, description="Total underpaid amount")
    std_days_past_due: float = Field(0, ge=0, example=10.0, description="Std dev of days past due")
    std_payment_ratio: float = Field(0, ge=0, example=0.0, description="Std dev of payment ratios")
    max_days_past_due: float = Field(0, example=3.0, description="Maximum days past due ever")
    min_payment_ratio: float = Field(1.0, ge=0, example=1.0, description="Minimum payment ratio ever")
    total_installments: int = Field(0, ge=0, example=40, description="Total installment count")
    num_contracts: int = Field(0, ge=0, example=3, description="Number of contracts")

    # ── Trend features ────────────────────────────────────
    late_rate_trend: float = Field(0, example=0.0, description="Change in late rate (positive=worsening)")
    payment_ratio_trend: float = Field(0, example=0.0, description="Change in payment ratio")
    recent_late_rate: float = Field(0, ge=0, le=1, example=0.0, description="Late rate in last 6 payments")
    recent_avg_days_late: float = Field(0, example=-9.0, description="Avg days late in last 6 payments")
    recent_min_payment_ratio: float = Field(1.0, ge=0, example=1.0, description="Min payment ratio last 6")

    # ── Bureau features (external credit) ─────────────────
    num_bureau_records: int = Field(0, ge=0, example=5, description="Total bureau records")
    num_active_loans: int = Field(0, ge=0, example=2, description="Active loans elsewhere")
    num_closed_loans: int = Field(0, ge=0, example=3, description="Closed loans elsewhere")
    total_external_debt: float = Field(0, ge=0, example=200000.0, description="Total debt elsewhere")
    total_external_credit: float = Field(0, ge=0, example=800000.0, description="Total credit elsewhere")
    total_overdue_amount: float = Field(0, ge=0, example=0.0, description="Total overdue elsewhere")
    max_days_overdue: int = Field(0, ge=0, example=0, description="Max days overdue elsewhere")
    has_any_overdue: int = Field(0, ge=0, le=1, example=0, description="Has any overdue elsewhere")
    num_credit_types: int = Field(0, ge=0, example=2, description="Diversity of credit products")
    active_loan_ratio: float = Field(0, ge=0, le=1, example=0.4, description="Fraction of loans active")
    external_utilization: float = Field(0, ge=0, le=1, example=0.25, description="External debt/credit ratio")

    # ── POS/Cash balance features ─────────────────────────
    pos_months_count: int = Field(0, ge=0, example=20, description="Months of POS data")
    pos_dpd_max: int = Field(0, ge=0, example=0, description="Max DPD (POS)")
    pos_dpd_mean: float = Field(0, ge=0, example=0.0, description="Avg DPD (POS)")
    pos_has_dpd: int = Field(0, ge=0, le=1, example=0, description="Has any POS DPD")
    pos_completed_count: int = Field(0, ge=0, example=5, description="Completed POS contracts")
    pos_active_count: int = Field(0, ge=0, example=1, description="Active POS contracts")

    # ── Credit card features ──────────────────────────────
    cc_months_count: int = Field(0, ge=0, example=12, description="Months of CC data")
    cc_avg_balance: float = Field(0, ge=0, example=50000.0, description="Avg CC balance")
    cc_max_balance: float = Field(0, ge=0, example=100000.0, description="Max CC balance")
    cc_avg_credit_limit: float = Field(0, ge=0, example=200000.0, description="Avg credit limit")
    cc_avg_drawings: float = Field(0, ge=0, example=20000.0, description="Avg monthly drawings")
    cc_dpd_max: int = Field(0, ge=0, example=0, description="Max DPD (credit card)")
    cc_has_dpd: int = Field(0, ge=0, le=1, example=0, description="Has any CC DPD")
    cc_min_payment_ratio: float = Field(0, ge=0, example=10.0, description="CC maturity count")
    cc_utilization: float = Field(0, ge=0, le=1, example=0.25, description="CC utilization ratio")

    # ── Previous application features ─────────────────────
    prev_app_count: int = Field(0, ge=0, example=3, description="Number of previous applications")
    prev_approved: int = Field(0, ge=0, example=2, description="Previously approved")
    prev_refused: int = Field(0, ge=0, example=0, description="Previously refused")
    prev_avg_credit: float = Field(0, ge=0, example=300000.0, description="Avg credit from past apps")
    prev_refusal_rate: float = Field(0, ge=0, le=1, example=0.0, description="Past refusal rate")


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