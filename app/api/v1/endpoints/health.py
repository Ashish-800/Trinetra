"""
Health Check and System Diagnostics Endpoint.
"""
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.api import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System Health"])
def get_health() -> HealthResponse:
    """Returns operational status, versioning, and safety enforcement mode."""
    detector_mode = "MOCK_DETECTOR" if settings.USE_MOCK_DETECTOR else f"YOLO ({settings.YOLO_MODEL_NAME})"
    return HealthResponse(
        status="healthy",
        app_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENV,
        safety_constitution="Strictly enforced: Advisory only, zero autonomous physical actuation",
        detector_mode=detector_mode,
    )
