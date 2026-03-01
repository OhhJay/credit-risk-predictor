import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class RiskCategory(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Provider(Base):
    __tablename__ = "providers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    business_type = Column(String, nullable=False)
    registration_number = Column(String, unique=True, nullable=False)
    years_in_operation = Column(Integer, nullable=False)
    annual_revenue = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transactions = relationship("Transaction", back_populates="provider")
    risk_scores = relationship("RiskScore", back_populates="provider")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("providers.id"), nullable=False)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String, nullable=False)  # "purchase" or "repayment"
    transaction_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=True)  # for purchases
    settled_date = Column(DateTime, nullable=True)  # actual repayment date
    status = Column(String, default="pending")  # pending, settled, overdue, defaulted
    created_at = Column(DateTime, default=datetime.utcnow)

    provider = relationship("Provider", back_populates="transactions")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("providers.id"), nullable=False)
    score = Column(Float, nullable=False)  # 0.0 - 1.0
    risk_category = Column(String, nullable=False)  # low, medium, high
    model_version = Column(String, nullable=False)
    feature_importance = Column(String, nullable=True)  # JSON string of SHAP values
    created_at = Column(DateTime, default=datetime.utcnow)

    provider = relationship("Provider", back_populates="risk_scores")
