from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.v1 import scoring, providers, health
from app.core.config import settings

app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "AI-powered credit scoring engine for evaluating healthcare provider "
        "creditworthiness based on transaction history and business metrics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(providers.router, prefix="/api/v1", tags=["Providers"])
app.include_router(scoring.router, prefix="/api/v1", tags=["Scoring"])

# Serve dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the credit risk scoring dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.on_event("startup")
async def startup_event():
    """Download models from Google Drive if needed, then load into memory."""
    from app.utils.gdrive_loader import ensure_models_available
    from app.services.model_service import ModelService

    ensure_models_available()
    app.state.model_service = ModelService()
    app.state.model_service.load_model()