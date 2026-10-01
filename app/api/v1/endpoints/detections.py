"""
Detection and Tracking Inspection Endpoints.
Allows querying object detections and multi-object tracks for incidents and media assets.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.domain import Detection, Incident, MediaAsset, Track
from app.schemas.domain import DetectionRead, TrackRead

router = APIRouter(tags=["Detections & Tracking"])


@router.get("/incidents/{incident_id}/detections", response_model=List[DetectionRead])
def list_incident_detections(
    incident_id: int,
    media_asset_id: Optional[int] = Query(None, description="Filter detections by media asset"),
    review_status: Optional[str] = Query(None, description="Filter detections by review status"),
    class_name: Optional[str] = Query(None, description="Filter by class name (e.g. person, vehicle)"),
    db: Session = Depends(get_db),
) -> List[DetectionRead]:
    """Retrieve all object detections associated with a specific incident."""
    # Verify incident exists
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found."
        )

    query = db.query(Detection).join(MediaAsset).filter(MediaAsset.incident_id == incident_id)

    if media_asset_id is not None:
        query = query.filter(Detection.media_asset_id == media_asset_id)
    if review_status is not None:
        query = query.filter(Detection.review_status == review_status)
    if class_name is not None:
        query = query.filter(Detection.class_name.ilike(f"%{class_name}%"))

    return query.order_by(Detection.id.asc()).all()


@router.get("/detections/{detection_id}", response_model=DetectionRead)
def get_detection(
    detection_id: int,
    db: Session = Depends(get_db),
) -> DetectionRead:
    """Retrieve a single detection record by ID."""
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection with ID {detection_id} not found."
        )
    return detection


@router.get("/incidents/{incident_id}/tracks", response_model=List[TrackRead])
def list_incident_tracks(
    incident_id: int,
    is_active: Optional[bool] = Query(None, description="Filter by active tracking status"),
    db: Session = Depends(get_db),
) -> List[TrackRead]:
    """Retrieve multi-frame object tracks associated with a specific incident."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found."
        )

    query = db.query(Track).filter(Track.incident_id == incident_id)
    if is_active is not None:
        query = query.filter(Track.is_active == is_active)

    return query.order_by(Track.id.asc()).all()


@router.get("/tracks/{track_id}", response_model=TrackRead)
def get_track(
    track_id: int,
    db: Session = Depends(get_db),
) -> TrackRead:
    """Retrieve a single track record by ID."""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track with ID {track_id} not found."
        )
    return track
