"""
Simulated Resource Inventory Endpoints.
Allows creating, querying, and updating simulated emergency resources and dispatch assets.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.domain import Resource
from app.schemas.api import ResourceUpdate
from app.schemas.domain import ResourceCreate, ResourceRead

router = APIRouter(prefix="/resources", tags=["Resource Inventory"])


@router.post("/", response_model=ResourceRead, status_code=status.HTTP_201_CREATED)
def create_resource(
    payload: ResourceCreate,
    db: Session = Depends(get_db),
) -> ResourceRead:
    """Register a new simulated emergency resource in the inventory."""
    resource = Resource(
        name=payload.name,
        resource_type=payload.resource_type,
        total_capacity=payload.total_capacity,
        available_capacity=payload.available_capacity,
        simulated_lat=payload.simulated_lat,
        simulated_lon=payload.simulated_lon,
        is_available=payload.is_available,
    )
    db.add(resource)
    db.commit()
    db.refresh(resource)
    return resource


@router.get("/", response_model=List[ResourceRead])
def list_resources(
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    is_available: Optional[bool] = Query(None, description="Filter by availability"),
    db: Session = Depends(get_db),
) -> List[ResourceRead]:
    """List all registered simulated emergency resources."""
    query = db.query(Resource)
    if resource_type is not None:
        query = query.filter(Resource.resource_type.ilike(f"%{resource_type}%"))
    if is_available is not None:
        query = query.filter(Resource.is_available == is_available)
    return query.order_by(Resource.id.asc()).all()


@router.get("/{resource_id}", response_model=ResourceRead)
def get_resource(
    resource_id: int,
    db: Session = Depends(get_db),
) -> ResourceRead:
    """Retrieve details of a specific simulated resource by ID."""
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource with ID {resource_id} not found."
        )
    return resource


@router.patch("/{resource_id}", response_model=ResourceRead)
def update_resource(
    resource_id: int,
    payload: ResourceUpdate,
    db: Session = Depends(get_db),
) -> ResourceRead:
    """Update fields (availability, capacity, coordinates) of an existing simulated resource."""
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource with ID {resource_id} not found."
        )

    # Validate capacity consistency if updating capacity
    new_total = payload.total_capacity if payload.total_capacity is not None else resource.total_capacity
    new_avail = payload.available_capacity if payload.available_capacity is not None else resource.available_capacity
    if new_avail > new_total:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"available_capacity ({new_avail}) cannot exceed total_capacity ({new_total})."
        )

    update_data = payload.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(resource, field_name, value)

    db.commit()
    db.refresh(resource)
    return resource
