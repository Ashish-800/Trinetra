"""
Services package.
Provides modular services for ingestion, detection, tracking, location estimation,
priority scoring, resource recommendation, application workflows, and RescueNet processing.
"""
# Detection
try:
    from app.services.detector import (
        BaseDetector,
        DetectedObject,
        DetectionResult,
        DetectorError,
        InvalidImageError,
        MockDetector,
        ModelWeightsNotFoundError,
        YOLOAerialDetector,
    )
except ImportError:
    pass

# Ingestion
try:
    from app.services.ingestion import (
        CorruptedMediaError,
        ExtractedFrame,
        IngestionResult,
        MediaIngestionError,
        MediaIngestionService,
        MediaNotFoundError,
        UnsupportedMediaFormatError,
    )
except ImportError:
    pass

# Location
from app.services.location import (
    BaseLocationEstimator,
    InvalidCoordinateError,
    LocationError,
    MissingTelemetryError,
    PlanarRayCastingEstimator,
    SimulatedLocationProvider,
)

# Priority & Verification
from app.services.priority_engine import (
    PriorityVerificationEngine,
    PrototypeScoringAssumptions,
)

# Resource Recommendation
from app.services.recommender import (
    ResourceRecommendationService,
    compute_haversine_distance_m,
    get_default_simulated_inventory,
)

# Tracker
from app.services.tracker import (
    AerialIoUTracker,
    BaseTracker,
    TrackedObject,
    compute_iou,
)

# Video Tracking
try:
    from app.services.video_tracking import (
        FrameTrackingResult,
        VideoTrackingPipeline,
        VideoTrackingSummary,
    )
except ImportError:
    pass

# RescueNet
from app.services.rescuenet import (
    RescueNetDatasetError,
    RescueNetService,
)

# Workflow
try:
    from app.services.workflow import (
        AssessmentWorkflowService,
        IncidentNotFoundError,
        WorkflowError,
        WorkflowExecutionResult,
    )
except ImportError:
    pass
