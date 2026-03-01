from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.core.models import Provider
from app.schemas.schemas import ProviderCreate, ProviderResponse

router = APIRouter()


@router.post("/", response_model=ProviderResponse)
async def create_provider(provider: ProviderCreate, db: Session = Depends(get_db)):
    """Create a new provider."""
    db_provider = Provider(**provider.model_dump())
    db.add(db_provider)
    db.commit()
    db.refresh(db_provider)
    return db_provider


@router.get("/", response_model=List[ProviderResponse])
async def list_providers(db: Session = Depends(get_db)):
    """List all providers."""
    return db.query(Provider).all()


@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: UUID, db: Session = Depends(get_db)):
    """Get a specific provider by ID."""
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider
