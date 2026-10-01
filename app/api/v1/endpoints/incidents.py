"""
Incident Management Endpoints.
Provides endpoints for creating, listing, and inspecting disaster assessment incidents.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.domain import Incident
from app.schemas.domain import IncidentCreate, IncidentRead

router = APIRouter(prefix="/incidents", tags=["Incidents"])


@router.post("/", response_model=IncidentRead, status_code=status.HTTP_201_CREATED)
def create_incident(
    payload: IncidentCreate,
    db: Session = Depends(get_db),
) -> IncidentRead:
    """Create a new disaster incident assessment workspace."""
    incident = Incident(
        title=payload.title,
        disaster_type=payload.disaster_type,
        location_name=payload.location_name,
        notes=payload.notes,
        is_active=True,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


@router.get("/", response_model=List[IncidentRead])
def list_incidents(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
) -> List[IncidentRead]:
    """List all registered disaster incidents."""
    query = db.query(Incident)
    if is_active is not None:
        query = query.filter(Incident.is_active == is_active)
    return query.order_by(Incident.created_at.desc()).all()


@router.get("/{incident_id}", response_model=IncidentRead)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
) -> IncidentRead:
    """Retrieve details of a specific incident by ID."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found."
        )
    return incident
