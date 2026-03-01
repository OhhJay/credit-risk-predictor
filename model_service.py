import joblib
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.core.config import settings
from app.schemas.schemas import ScoringRequest, ScoringResponse
from app.services.feature_engineering import extract_features, FEATURE_NAMES

logger = logging.getLogger(__name__)


class ModelService:
    """Handles ML model loading, prediction, and SHAP explanations."""

    def __init__(self):
        self.model = None
        self.model_version = settings.MODEL_VERSION
        self.explainer = None

    def load_model(self):
        """Load the trained model from disk."""
        model_path = Path(settings.MODEL_PATH)

        if not model_path.exists():
            logger.warning(
                f"Model not found at {model_path}. "
                "Using fallback rule-based scoring. "
                "Train and export a model to enable ML scoring."
            )
            self.model = None
            return

        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path} (version: {self.model_version})")

            # Try to load SHAP explainer
            explainer_path = model_path.parent / "explainer.joblib"
            if explainer_path.exists():
                self.explainer = joblib.load(explainer_path)
                logger.info("SHAP explainer loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, request: ScoringRequest) -> ScoringResponse:
        """Generate a risk score for a single provider."""
        features_df = extract_features(request)

        if self.model is not None:
            score = self._ml_predict(features_df)
            importance = self._get_feature_importance(features_df)
        else:
            score = self._rule_based_predict(request)
            importance = self._rule_based_importance(request)

        risk_category = self._categorize_risk(score)

        return ScoringResponse(
            risk_score=round(score, 4),
            risk_category=risk_category,
            model_version=self.model_version if self.model else "rule-based-v1",
            feature_importance=importance,
            scored_at=datetime.utcnow(),
        )

    def _ml_predict(self, features_df: pd.DataFrame) -> float:
        """Use the trained ML model for prediction."""
        if hasattr(self.model, "predict_proba"):
            # Classification model — probability of high risk (class 1)
            proba = self.model.predict_proba(features_df)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Regression model
            score = float(self.model.predict(features_df)[0])
            return max(0.0, min(1.0, score))

    def _get_feature_importance(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Get SHAP-based feature importance for this prediction."""
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(features_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # class 1 for binary
                importance = dict(zip(FEATURE_NAMES, shap_values[0].tolist()))
                return {k: round(v, 4) for k, v in importance.items()}
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Fallback: model feature importance
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(FEATURE_NAMES, self.model.feature_importances_.tolist()))
            return {k: round(v, 4) for k, v in importance.items()}

        return {}

    def _rule_based_predict(self, request: ScoringRequest) -> float:
        """Fallback rule-based scoring when no ML model is loaded."""
        score = 0.0

        # Default rate is the strongest signal (weight: 0.30)
        score += request.default_rate * 0.30

        # High credit utilization = higher risk (weight: 0.20)
        score += request.credit_utilization_ratio * 0.20

        # Slow repayment increases risk (weight: 0.15)
        repay_risk = min(request.avg_days_to_repay / 90, 1.0)
        score += repay_risk * 0.15

        # Low repayment ratio = higher risk (weight: 0.15)
        if request.total_purchase_amount > 0:
            repayment_ratio = request.total_repayment_amount / request.total_purchase_amount
            score += (1 - min(repayment_ratio, 1.0)) * 0.15
        else:
            score += 0.15

        # Newer businesses are riskier (weight: 0.10)
        tenure_risk = max(1 - (request.years_in_operation / 10), 0)
        score += tenure_risk * 0.10

        # Outstanding balance relative to revenue (weight: 0.10)
        if request.annual_revenue > 0:
            balance_ratio = min(request.outstanding_balance / request.annual_revenue, 1.0)
            score += balance_ratio * 0.10
        else:
            score += 0.10

        return max(0.0, min(1.0, score))

    def _rule_based_importance(self, request: ScoringRequest) -> Dict[str, float]:
        """Return pseudo-importance for rule-based scoring."""
        return {
            "default_rate": 0.30,
            "credit_utilization_ratio": 0.20,
            "avg_days_to_repay": 0.15,
            "repayment_ratio": 0.15,
            "years_in_operation": 0.10,
            "balance_to_revenue_ratio": 0.10,
        }

    def _categorize_risk(self, score: float) -> str:
        if score <= settings.LOW_RISK_THRESHOLD:
            return "low"
        elif score <= settings.HIGH_RISK_THRESHOLD:
            return "medium"
        return "high"
