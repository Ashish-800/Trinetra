"""
Emergency Resource Recommendation Endpoints.
Retrieves advisory resource recommendations with strict human review safety gates.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.domain import Detection, Incident, MediaAsset, ResourceRecommendation
from app.schemas.domain import ResourceRecommendationRead

router = APIRouter(tags=["Resource Recommendations"])


@router.get("/incidents/{incident_id}/recommendations", response_model=List[ResourceRecommendationRead])
def list_incident_recommendations(
    incident_id: int,
    db: Session = Depends(get_db),
) -> List[ResourceRecommendationRead]:
    """
    Retrieve advisory resource allocation recommendations for an incident.
    Every recommendation enforces requires_human_authorization=True.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found."
        )

    recommendations = (
        db.query(ResourceRecommendation)
        .join(Detection)
        .join(MediaAsset)
        .filter(MediaAsset.incident_id == incident_id)
        .order_by(ResourceRecommendation.suitability_score.desc())
        .all()
    )
    return recommendations


@router.get("/recommendations/{recommendation_id}", response_model=ResourceRecommendationRead)
def get_recommendation(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> ResourceRecommendationRead:
    """Retrieve an individual advisory resource recommendation by ID."""
    rec = db.query(ResourceRecommendation).filter(ResourceRecommendation.id == recommendation_id).first()
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ResourceRecommendation with ID {recommendation_id} not found."
        )
    return rec
