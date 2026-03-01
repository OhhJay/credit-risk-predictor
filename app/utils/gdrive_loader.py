import os
import logging
from pathlib import Path

import gdown

from app.core.config import settings

logger = logging.getLogger(__name__)


def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """
    Download a file from Google Drive using a public shared link.

    How to get the file_id:
    1. Upload model.joblib to Google Drive
    2. Right-click → Share → "Anyone with the link"
    3. Copy link: https://drive.google.com/file/d/FILE_ID/view
    4. The FILE_ID is the long string between /d/ and /view

    Set these as Railway environment variables:
        MODEL_GDRIVE_ID=1ABC...xyz
        EXPLAINER_GDRIVE_ID=1DEF...xyz
    """
    if not file_id:
        return False

    output = Path(output_path)

    if output.exists():
        logger.info(f"File already exists at {output_path}, skipping download")
        return True

    # Ensure directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading from Google Drive: {file_id} → {output_path}")

    try:
        gdown.download(url, output_path, quiet=False)

        if output.exists() and output.stat().st_size > 0:
            size_mb = output.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Downloaded successfully ({size_mb:.1f} MB)")
            return True
        else:
            logger.error("Download produced empty or missing file")
            return False

    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def ensure_models_available():
    """
    Check if models exist locally. If not, try to download from Google Drive.
    Called once on application startup.
    """
    model_path = Path(settings.MODEL_PATH)
    explainer_path = model_path.parent / "explainer.joblib"

    # Try downloading model if not present locally
    if not model_path.exists() and settings.MODEL_GDRIVE_ID:
        logger.info("Model not found locally, downloading from Google Drive...")
        download_from_gdrive(settings.MODEL_GDRIVE_ID, str(model_path))

    # Try downloading explainer if not present locally
    if not explainer_path.exists() and settings.EXPLAINER_GDRIVE_ID:
        logger.info("Explainer not found locally, downloading from Google Drive...")
        download_from_gdrive(settings.EXPLAINER_GDRIVE_ID, str(explainer_path))

    # Report status
    if model_path.exists():
        logger.info(f"✅ Model ready at {model_path}")
    else:
        logger.warning("⚠️  No model available — API will use rule-based fallback")
        logger.warning("   Set MODEL_GDRIVE_ID env var or place model.joblib in ml/")
