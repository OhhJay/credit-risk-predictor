from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Credit Risk Scoring API"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/credit_risk"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ML Model
    MODEL_PATH: str = "ml/model.joblib"
    MODEL_VERSION: str = "1.0.0"

    # Google Drive model loading (set file IDs from shared links)
    MODEL_GDRIVE_ID: str = ""
    EXPLAINER_GDRIVE_ID: str = ""

    # Scoring thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    HIGH_RISK_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
