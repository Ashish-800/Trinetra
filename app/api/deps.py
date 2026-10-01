"""
FastAPI Dependency Injection Providers.
Provides scoped database sessions, detectors, and service instances.
"""
from typing import Generator, Optional
from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.detector import BaseDetector, MockDetector, YOLOAerialDetector
from app.services.ingestion import MediaIngestionService
from app.services.location import SimulatedLocationProvider
from app.services.priority_engine import PriorityVerificationEngine
from app.services.recommender import ResourceRecommendationService
from app.services.rescuenet import RescueNetService
from app.services.rescuenet_segmentor import RescueNetSegmentor
from app.services.tracker import AerialIoUTracker
from app.services.workflow import AssessmentWorkflowService


def get_detector() -> BaseDetector:
    """Provides configured detector instance (mock or YOLO based on config)."""
    if settings.USE_MOCK_DETECTOR:
        return MockDetector()
    try:
        weight_path = settings.resolved_yolo_weights_path
        target_classes = ["person"] if settings.YOLO_PERSON_ONLY else None
        return YOLOAerialDetector(
            weights_path=weight_path,
            default_conf_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
            device=settings.YOLO_DEVICE,
            target_classes=target_classes,
            person_only=settings.YOLO_PERSON_ONLY,
        )
    except Exception:
        # Fall back safely to MockDetector if weights are missing
        return MockDetector()


def get_ingestion_service() -> MediaIngestionService:
    """Provides MediaIngestionService instance."""
    return MediaIngestionService()


def get_rescuenet_service() -> RescueNetService:
    """Provides RescueNetService instance for disaster context analysis."""
    return RescueNetService()


_cached_segmentor: Optional[RescueNetSegmentor] = None


def get_rescuenet_segmentor() -> Optional[RescueNetSegmentor]:
    """Provides singleton RescueNetSegmentor when weights are available or enabled in config."""
    global _cached_segmentor
    weights_p = settings.resolved_rescuenet_segmentation_weights_path
    if not settings.ENABLE_RESCUENET_SEGMENTATION and not weights_p.exists():
        return None
    if _cached_segmentor is None:
        try:
            device = settings.RESCUENET_SEGMENTATION_DEVICE
            _cached_segmentor = RescueNetSegmentor(
                weights_path=weights_p if weights_p.exists() else None,
                device=device,
            )
        except Exception:
            return None
    return _cached_segmentor


def get_workflow_service(
    db: Session = Depends(get_db),
    detector: BaseDetector = Depends(get_detector),
    ingestion: MediaIngestionService = Depends(get_ingestion_service),
    rescuenet: RescueNetService = Depends(get_rescuenet_service),
    segmentor: Optional[RescueNetSegmentor] = Depends(get_rescuenet_segmentor),
) -> AssessmentWorkflowService:
    """Provides AssessmentWorkflowService configured with current DB session, detector, RescueNet context, and optional segmentor."""
    return AssessmentWorkflowService(
        db=db,
        detector=detector,
        tracker=AerialIoUTracker(),
        location_estimator=SimulatedLocationProvider(),
        priority_engine=PriorityVerificationEngine(),
        recommender_service=ResourceRecommendationService(),
        ingestion_service=ingestion,
        rescuenet_service=rescuenet,
        segmentor=segmentor,
    )
