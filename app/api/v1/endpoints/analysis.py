"""
Automated Assessment and Workflow Execution Endpoints.
Connects uploaded aerial media directly to the 8-step controlled analysis pipeline.
"""
from pathlib import Path
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_ingestion_service, get_workflow_service
from app.core.config import settings
from app.models.domain import Incident, MediaAsset
from app.schemas.api import AnalysisRequest
from app.services.ingestion import MediaIngestionService
from app.services.workflow import (
    AssessmentWorkflowService,
    IncidentNotFoundError,
    WorkflowError,
    WorkflowExecutionResult,
)

router = APIRouter(prefix="/analysis", tags=["Assessment Analysis"])


@router.post("/start", response_model=WorkflowExecutionResult, status_code=status.HTTP_200_OK)
def start_assessment_analysis(
    payload: AnalysisRequest,
    db: Session = Depends(get_db),
    workflow: AssessmentWorkflowService = Depends(get_workflow_service),
) -> WorkflowExecutionResult:
    """
    Executes the complete 8-step controlled assessment workflow for a registered media asset.
    Includes detection, tracking, simulated location attachment, priority evaluation,
    and advisory resource recommendations.
    """
    # 1. Retrieve registered media asset
    asset = db.query(MediaAsset).filter(MediaAsset.id == payload.media_asset_id).first()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MediaAsset with ID {payload.media_asset_id} not found."
        )

    # 2. Execute assessment workflow
    try:
        result = workflow.run_workflow(
            media_path=asset.file_path,
            incident_id=asset.incident_id,
            telemetry=payload.telemetry,
            sampling_interval_seconds=payload.sampling_interval_seconds,
            detection_confidence_threshold=payload.detection_confidence_threshold,
            media_asset_id=asset.id,
            rescuenet_mask_path=payload.rescuenet_mask_path,
            enable_segmentation=payload.enable_segmentation,
        )
        return result
    except IncidentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except WorkflowError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error executing assessment workflow: {str(e)}"
        )


@router.post("/multimodal", response_model=WorkflowExecutionResult, status_code=status.HTTP_200_OK)
async def execute_multimodal_assessment(
    file: UploadFile = File(..., description="Aerial image or video file for dual-model assessment"),
    incident_id: Optional[int] = Form(None, description="Optional incident ID; created if omitted"),
    confidence_threshold: Optional[float] = Form(None, description="Optional detection confidence threshold"),
    enable_segmentation: bool = Form(True, description="Enable RescueNet environmental segmentation"),
    db: Session = Depends(get_db),
    workflow: AssessmentWorkflowService = Depends(get_workflow_service),
    ingestion: MediaIngestionService = Depends(get_ingestion_service),
) -> WorkflowExecutionResult:
    """
    Executes an end-to-end multimodal aerial assessment in a single request:
    Image/Video Upload -> YOLO Person Detections -> RescueNet Environmental Segmentation
    -> IoU Tracking -> Simulated Location -> Uncertainty & Priority Engine
    -> Advisory Resource Recommendations -> Database Persistence.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = ingestion.SUPPORTED_IMAGE_EXTENSIONS | ingestion.SUPPORTED_VIDEO_EXTENSIONS
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )

    # 1. Resolve or create incident
    if incident_id is not None:
        inc = db.query(Incident).filter(Incident.id == incident_id).first()
        if not inc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident with ID {incident_id} not found."
            )
    else:
        inc = db.query(Incident).filter(Incident.title.contains("Multimodal")).first()
        if not inc:
            inc = Incident(
                title="Multimodal UAV Assessment Incident",
                disaster_type="FLOOD",
                location_name="Coastal Disaster Sector Alpha",
                notes="Automated multimodal UAV assessment combining YOLO and RescueNet segmentation",
            )
            db.add(inc)
            db.commit()
            db.refresh(inc)

    # 2. Store media file safely
    safe_name = f"multimodal_{inc.id}_{uuid.uuid4().hex[:8]}{file_ext}"
    dest_path = settings.UPLOAD_DIR / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded media: {str(e)}"
        )

    # 3. Create MediaAsset record
    is_video = file_ext in ingestion.SUPPORTED_VIDEO_EXTENSIONS
    media_asset = MediaAsset(
        incident_id=inc.id,
        file_path=str(dest_path),
        media_type="VIDEO" if is_video else "IMAGE",
        mime_type=file.content_type or ("video/mp4" if is_video else "image/jpeg"),
        file_size_bytes=dest_path.stat().st_size,
    )
    db.add(media_asset)
    db.commit()
    db.refresh(media_asset)

    # 4. Execute multimodal workflow
    try:
        result = workflow.run_workflow(
            media_path=dest_path,
            incident_id=inc.id,
            media_asset_id=media_asset.id,
            detection_confidence_threshold=confidence_threshold,
            enable_segmentation=enable_segmentation,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal assessment pipeline error: {str(e)}"
        )

