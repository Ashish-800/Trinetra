"""
Priority and Uncertainty Assessment Endpoints.
Allows querying explainable priority assessments for incidents and detections.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.domain import Detection, Incident, MediaAsset, PriorityAssessment
from app.schemas.domain import PriorityAssessmentRead

router = APIRouter(tags=["Priority Assessments"])


@router.get("/incidents/{incident_id}/priority-assessments", response_model=List[PriorityAssessmentRead])
def list_incident_priority_assessments(
    incident_id: int,
    db: Session = Depends(get_db),
) -> List[PriorityAssessmentRead]:
    """Retrieve priority assessments for all detections within a specific incident."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found."
        )

    assessments = (
        db.query(PriorityAssessment)
        .join(Detection)
        .join(MediaAsset)
        .filter(MediaAsset.incident_id == incident_id)
        .order_by(PriorityAssessment.urgency_score.desc())
        .all()
    )
    return assessments


@router.get("/priority-assessments/{assessment_id}", response_model=PriorityAssessmentRead)
def get_priority_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
) -> PriorityAssessmentRead:
    """Retrieve an individual priority assessment by ID."""
    assessment = db.query(PriorityAssessment).filter(PriorityAssessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PriorityAssessment with ID {assessment_id} not found."
        )
    return assessment
