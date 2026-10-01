"""
Application Service: End-to-End Controlled Disaster UAV Assessment Workflow.
Orchestrates:
1. Incident selection or creation
2. Media registration and frame extraction
3. Object detection
4. Multi-object tracking across video frames
5. Simulated location attachment
6. Priority, urgency, and uncertainty evaluation
7. Advisory resource recommendation
8. Complete database persistence
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.domain import (
    Detection,
    Incident,
    MediaAsset,
    PriorityAssessment,
    Resource,
    ResourceRecommendation,
    Track,
)
from app.schemas.api import (
    FrontendDetectionItem,
    FrontendDeviceInfo,
    FrontendIncidentSummary,
    FrontendMediaSummary,
)
from app.schemas.domain import BoundingBoxSchema, IncidentCreate
from app.schemas.location import DroneTelemetryMetadata, LocationSource
from app.schemas.priority import AssessmentInput, HazardSeverity, PriorityAssessmentResult
from app.schemas.rescuenet import RescueNetSceneAnalysis
from app.schemas.resource_engine import IncidentRequirement, RecommendationOutcome
from app.services.detector import BaseDetector, MockDetector
from app.services.ingestion import MediaIngestionService
from app.services.location import BaseLocationEstimator, SimulatedLocationProvider
from app.services.priority_engine import PriorityVerificationEngine
from app.services.recommender import ResourceRecommendationService
from app.services.rescuenet import RescueNetService
from app.services.rescuenet_segmentor import RescueNetSegmentor
from app.services.tracker import AerialIoUTracker, BaseTracker

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Base exception for assessment workflow execution."""
    pass


class IncidentNotFoundError(WorkflowError):
    """Raised when the specified incident ID does not exist."""
    pass


class WorkflowExecutionResult(BaseModel):
    """Structured summary of the complete assessment workflow execution."""
    incident_id: int
    incident_title: str
    media_asset_id: int
    media_type: str
    frames_processed: int
    total_detections_found: int
    unique_person_tracks: int
    locations_attached_count: int
    priority_assessment: PriorityAssessmentResult
    recommendation_outcome: RecommendationOutcome
    saved_records_summary: str
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scene_context: Optional[RescueNetSceneAnalysis] = None

    # Frontend / Command Center Integration Fields
    incident: Optional[FrontendIncidentSummary] = None
    media: Optional[FrontendMediaSummary] = None
    detections: List[FrontendDetectionItem] = Field(default_factory=list)
    device_info: Optional[FrontendDeviceInfo] = None
    safety_disclaimer: str = Field(
        default=(
            "COMMAND CENTER SAFETY NOTICE: All geographic coordinates are simulated UAV projections. "
            "Detections indicate potential persons, not confirmed survivors. "
            "All resource recommendations are strictly advisory (status=PENDING_REVIEW, requires_human_authorization=True)."
        )
    )


class AssessmentWorkflowService:
    """
    Application Service orchestrating the complete 8-step disaster assessment pipeline.
    Enforces the Project Safety Constitution:
    - All recommendations are strictly advisory (human review gate).
    - Coordinates are explicitly labeled source='simulated'.
    - Person detections are labeled 'person' or 'potential survivor', never 'confirmed survivor'.
    """

    def __init__(
        self,
        db: Session,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[BaseTracker] = None,
        location_estimator: Optional[BaseLocationEstimator] = None,
        priority_engine: Optional[PriorityVerificationEngine] = None,
        recommender_service: Optional[ResourceRecommendationService] = None,
        ingestion_service: Optional[MediaIngestionService] = None,
        rescuenet_service: Optional[RescueNetService] = None,
        segmentor: Optional[RescueNetSegmentor] = None,
    ):
        self.db = db
        self.detector = detector or MockDetector()
        self.tracker = tracker or AerialIoUTracker()
        self.location_estimator = location_estimator or SimulatedLocationProvider()
        self.priority_engine = priority_engine or PriorityVerificationEngine()
        self.recommender_service = recommender_service or ResourceRecommendationService()
        self.ingestion_service = ingestion_service or MediaIngestionService()
        self.rescuenet_service = rescuenet_service or RescueNetService()
        self.segmentor = segmentor

    def run_workflow(
        self,
        media_path: Union[str, Path],
        incident_id: Optional[int] = None,
        new_incident: Optional[IncidentCreate] = None,
        telemetry: Optional[DroneTelemetryMetadata] = None,
        sampling_interval_seconds: float = 1.0,
        detection_confidence_threshold: Optional[float] = None,
        hazard_severity: HazardSeverity = HazardSeverity.NONE,
        is_accessible: Optional[bool] = None,
        media_asset_id: Optional[int] = None,
        rescuenet_mask_path: Optional[Union[str, Path]] = None,
        enable_segmentation: Optional[bool] = None,
    ) -> WorkflowExecutionResult:
        """
        Executes the 8-step controlled workflow and persists all results.
        """
        media_file = Path(media_path).resolve()

        # RescueNet Disaster Scene Context Integration
        scene_analysis: Optional[RescueNetSceneAnalysis] = None

        # Path A: Explicit ground-truth mask provided
        if rescuenet_mask_path is not None:
            scene_analysis = self.rescuenet_service.analyze_scene(rescuenet_mask_path)
        # Path B: Optional Deep Learning semantic segmentation inference on input image/media
        elif (
            (enable_segmentation is True or (enable_segmentation is None and settings.ENABLE_RESCUENET_SEGMENTATION))
            and self.segmentor is not None
        ):
            try:
                scene_analysis = self.segmentor.predict_scene_analysis(media_file, image_id=media_file.stem)
            except Exception as e:
                logger.warning("RescueNet segmentation inference failed on %s: %s", media_file, e)

        # Connect scene_analysis to priority/urgency if inferred
        if scene_analysis is not None:
            # Use inferred hazard severity only if manual override was not provided
            if hazard_severity == HazardSeverity.NONE:
                hazard_severity = scene_analysis.inferred_hazard_severity
            # Use inferred accessibility only if not explicitly specified
            if is_accessible is None and scene_analysis.inferred_accessibility is not None:
                is_accessible = (scene_analysis.inferred_accessibility == "accessible")

        # -------------------------------------------------------------
        # Step 1: Create or select an Incident
        # -------------------------------------------------------------
        incident: Optional[Incident] = None
        if incident_id is not None:
            incident = self.db.query(Incident).filter(Incident.id == incident_id).first()
            if not incident:
                raise IncidentNotFoundError(f"Incident with ID {incident_id} not found in database.")
        elif new_incident is not None:
            incident = Incident(
                title=new_incident.title,
                disaster_type=new_incident.disaster_type,
                location_name=new_incident.location_name,
                notes=new_incident.notes,
            )
            self.db.add(incident)
            self.db.commit()
            self.db.refresh(incident)
        else:
            # Default fallback incident for automated batch processing
            incident = Incident(
                title=f"Automated UAV Survey - {media_file.stem}",
                disaster_type="GENERAL",
                location_name="Survey Area Alpha",
                notes="Created automatically by AssessmentWorkflowService",
            )
            self.db.add(incident)
            self.db.commit()
            self.db.refresh(incident)

        # -------------------------------------------------------------
        # Step 2: Register image/video media
        # -------------------------------------------------------------
        ingestion_result = self.ingestion_service.ingest(
            file_path=media_file,
            interval_seconds=sampling_interval_seconds,
        )

        media_asset: Optional[MediaAsset] = None
        if media_asset_id is not None:
            media_asset = self.db.query(MediaAsset).filter(MediaAsset.id == media_asset_id).first()
        if media_asset is None and incident is not None:
            media_asset = self.db.query(MediaAsset).filter(
                MediaAsset.incident_id == incident.id,
                MediaAsset.file_path == str(media_file),
            ).first()

        if media_asset is None:
            media_asset = MediaAsset(
                incident_id=incident.id,
                file_path=str(media_file),
                media_type=ingestion_result.media_type,
                mime_type=ingestion_result.mime_type,
                file_size_bytes=ingestion_result.file_size_bytes,
                width_px=ingestion_result.width_px,
                height_px=ingestion_result.height_px,
                duration_seconds=ingestion_result.duration_seconds,
                frame_count=ingestion_result.total_frames_extracted,
            )
            self.db.add(media_asset)
            self.db.commit()
            self.db.refresh(media_asset)
        else:
            media_asset.media_type = ingestion_result.media_type
            media_asset.mime_type = ingestion_result.mime_type
            media_asset.file_size_bytes = ingestion_result.file_size_bytes
            media_asset.width_px = ingestion_result.width_px
            media_asset.height_px = ingestion_result.height_px
            media_asset.duration_seconds = ingestion_result.duration_seconds
            media_asset.frame_count = ingestion_result.total_frames_extracted
            self.db.commit()
            self.db.refresh(media_asset)

        # -------------------------------------------------------------
        # Steps 3 & 4: Run Detection and Multi-Object Tracking
        # -------------------------------------------------------------
        self.tracker.reset()
        saved_detections: List[Detection] = []
        track_records_map: dict[int, Track] = {}
        unique_person_track_ids: set[int] = set()
        locations_attached_count = 0

        for frame_info in ingestion_result.frames:
            # 3. Detection
            det_res = self.detector.predict(
                image_input=frame_info.output_path,
                conf_threshold=detection_confidence_threshold,
            )

            # 4. Tracking
            if ingestion_result.media_type == "VIDEO":
                tracked_objects = self.tracker.update(
                    detections=det_res.objects,
                    frame_index=frame_info.frame_index,
                    timestamp_seconds=frame_info.timestamp_seconds,
                )
            else:
                # Single static image pass
                from app.services.tracker import TrackedObject
                tracked_objects = [
                    TrackedObject(
                        track_id=idx + 1,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=obj.bbox,
                        frame_index=0,
                        timestamp_seconds=0.0,
                        hits=1,
                        is_provisional=False,
                    )
                    for idx, obj in enumerate(det_res.objects)
                ]

            for tobj in tracked_objects:
                # Synchronize Track entity in DB
                if tobj.track_id not in track_records_map:
                    track_model = Track(
                        incident_id=incident.id,
                        track_label=f"{tobj.class_name}_track_{tobj.track_id:03d}",
                        start_time=datetime.now(timezone.utc),
                    )
                    self.db.add(track_model)
                    self.db.flush()
                    track_records_map[tobj.track_id] = track_model

                db_track = track_records_map[tobj.track_id]

                # ---------------------------------------------------------
                # Step 5: Attach simulated location data
                # ---------------------------------------------------------
                loc_record = self.location_estimator.estimate_location(
                    bbox=tobj.bbox,
                    image_width=frame_info.width_px,
                    image_height=frame_info.height_px,
                    telemetry=telemetry,
                )

                lat = loc_record.latitude if loc_record else None
                lon = loc_record.longitude if loc_record else None
                loc_source = loc_record.source.value if loc_record else "unknown"
                unc_rad = loc_record.accuracy_meters if loc_record else None

                if loc_record is not None:
                    locations_attached_count += 1

                if tobj.class_name in ("person", "potential survivor"):
                    unique_person_track_ids.add(tobj.track_id)

                # Persist Detection Record
                det_model = Detection(
                    media_asset_id=media_asset.id,
                    track_id=db_track.id,
                    class_name=tobj.class_name,
                    confidence=tobj.confidence,
                    bbox_x1=tobj.bbox.x1,
                    bbox_y1=tobj.bbox.y1,
                    bbox_x2=tobj.bbox.x2,
                    bbox_y2=tobj.bbox.y2,
                    frame_number=tobj.frame_index,
                    image_reference=frame_info.output_path,
                    latitude=lat,
                    longitude=lon,
                    location_source=loc_source,
                    uncertainty_radius_m=unc_rad,
                    review_status="PENDING_REVIEW",
                )
                self.db.add(det_model)
                saved_detections.append(det_model)

        self.db.commit()

        # -------------------------------------------------------------
        # Step 6: Calculate Urgency and Verification Status
        # -------------------------------------------------------------
        avg_conf = None
        if saved_detections:
            avg_conf = sum(d.confidence for d in saved_detections) / len(saved_detections)

        priority_result = self.priority_engine.assess(
            AssessmentInput(
                tracked_person_count=len(unique_person_track_ids),
                hazard_severity=hazard_severity,
                is_accessible=is_accessible,
                average_detection_confidence=avg_conf,
                observation_age_minutes=0.0,  # Just processed
                has_missing_telemetry=(telemetry is None),
            )
        )

        # Save Priority Assessments in DB for each detection
        for det_entity in saved_detections:
            assessment_entity = PriorityAssessment(
                detection_id=det_entity.id,
                urgency_score=priority_result.urgency_score,
                uncertainty_score=priority_result.uncertainty_score,
                priority_level=priority_result.composite_priority.value,
                rationale=(
                    f"Assessed {priority_result.urgency_level} urgency ({priority_result.urgency_score:.2f}) "
                    f"with {priority_result.uncertainty_level} uncertainty ({priority_result.uncertainty_score:.2f})."
                ),
            )
            self.db.add(assessment_entity)

        self.db.commit()

        # -------------------------------------------------------------
        # Step 7: Generate Resource Recommendations
        # -------------------------------------------------------------
        # Synthesize primary capability requirement from RescueNet disaster context or incident type
        if scene_analysis:
            if scene_analysis.water_coverage_pct >= 4.0:
                req_caps = ["flood_rescue"]
            elif scene_analysis.debris_coverage_pct >= 8.0 or scene_analysis.destroyed_building_pct >= 0.5:
                req_caps = ["debris_clearing"]
            elif "flood_rescue" in [c.lower() for c in scene_analysis.inferred_required_capabilities]:
                req_caps = ["flood_rescue"]
            elif "debris_clearing" in [c.lower() for c in scene_analysis.inferred_required_capabilities]:
                req_caps = ["debris_clearing"]
            else:
                req_caps = ["ground_search"]
        elif incident.disaster_type.upper() in ("FLOOD", "WATER_SURGE"):
            req_caps = ["flood_rescue"]
        elif incident.disaster_type.upper() in ("COLLAPSE", "EARTHQUAKE"):
            req_caps = ["debris_clearing"]
        else:
            req_caps = ["ground_search"]

        # Pick primary location from first detection if available
        first_loc = None
        if saved_detections and saved_detections[0].latitude is not None:
            from app.schemas.location import LocationRecord
            first_loc = LocationRecord(
                latitude=saved_detections[0].latitude,
                longitude=saved_detections[0].longitude,
                source=LocationSource.SIMULATED,
                accuracy_meters=saved_detections[0].uncertainty_radius_m,
            )

        recommendation_outcome = self.recommender_service.recommend(
            IncidentRequirement(
                incident_id=str(incident.id),
                disaster_type=incident.disaster_type,
                required_capabilities=req_caps,
                needed_capacity=max(1, len(unique_person_track_ids)),
                target_location=first_loc,
                urgency_level=priority_result.urgency_level,
            )
        )

        # -------------------------------------------------------------
        # Step 8: Save Resource Recommendations in Database
        # -------------------------------------------------------------
        # Ensure simulated resources exist in DB if not already present
        for rec_item in recommendation_outcome.recommendations:
            db_res = self.db.query(Resource).filter(Resource.name == rec_item.resource_name).first()
            if not db_res:
                db_res = Resource(
                    name=rec_item.resource_name,
                    resource_type=rec_item.resource_type,
                    total_capacity=6,
                    available_capacity=4,
                    simulated_lat=first_loc.latitude if first_loc else 34.05,
                    simulated_lon=first_loc.longitude if first_loc else -118.25,
                    is_available=True,
                )
                self.db.add(db_res)
                self.db.flush()

            # Attach recommendation to the primary detection if available
            target_det_id = saved_detections[0].id if saved_detections else None
            if target_det_id:
                recom_db = ResourceRecommendation(
                    detection_id=target_det_id,
                    resource_id=db_res.id,
                    suitability_score=rec_item.suitability_score,
                    rationale=rec_item.rationale,
                    status="PENDING_REVIEW",
                    requires_human_authorization=True,
                )
                self.db.add(recom_db)

        self.db.commit()

        summary_text = (
            f"Successfully processed {ingestion_result.media_type} for Incident '{incident.title}' (ID {incident.id}). "
            f"Frames: {ingestion_result.total_frames_extracted} | Detections: {len(saved_detections)} | "
            f"Unique Person Tracks: {len(unique_person_track_ids)} | "
            f"Priority: {priority_result.composite_priority.value} | "
            f"Recommendations: {len(recommendation_outcome.recommendations)} ({recommendation_outcome.status})."
        )

        # -------------------------------------------------------------
        # Step 9: Assemble Frontend / Command Center Integration Models
        # -------------------------------------------------------------
        incident_summary = FrontendIncidentSummary(
            id=incident.id,
            title=incident.title,
            disaster_type=incident.disaster_type,
            location_name=incident.location_name,
            is_active=incident.is_active,
        )

        media_summary = FrontendMediaSummary(
            id=media_asset.id,
            media_type=ingestion_result.media_type,
            file_path=str(media_asset.file_path),
            file_size_bytes=media_asset.file_size_bytes or 0,
            width_px=ingestion_result.width_px,
            height_px=ingestion_result.height_px,
            frames_processed=ingestion_result.total_frames_extracted,
        )

        frontend_detections = [
            FrontendDetectionItem(
                detection_id=d.id,
                track_id=d.track_id,
                label="potential person" if d.class_name.lower() in ("person", "potential survivor") else d.class_name,
                confidence=round(float(d.confidence), 4),
                bbox=BoundingBoxSchema(
                    x1=round(float(d.bbox_x1), 2),
                    y1=round(float(d.bbox_y1), 2),
                    x2=round(float(d.bbox_x2), 2),
                    y2=round(float(d.bbox_y2), 2),
                    confidence=round(float(d.confidence), 4),
                    class_name="potential person" if d.class_name.lower() in ("person", "potential survivor") else d.class_name,
                ),
                frame_number=d.frame_number,
                latitude=round(float(d.latitude), 6) if d.latitude is not None else None,
                longitude=round(float(d.longitude), 6) if d.longitude is not None else None,
                location_source=d.location_source or "simulated",
                uncertainty_radius_m=round(float(d.uncertainty_radius_m), 1) if d.uncertainty_radius_m is not None else None,
                review_status="PENDING_REVIEW",
            )
            for d in saved_detections
        ]

        import torch
        is_cuda = torch.cuda.is_available()
        device_telemetry = FrontendDeviceInfo(
            cuda_available=is_cuda,
            inference_device=f"cuda:{torch.cuda.current_device()}" if is_cuda else "cpu",
            gpu_name=torch.cuda.get_device_name(0) if is_cuda else None,
            detector_model=str(settings.resolved_yolo_weights_path.name),
            segmentation_model=str(settings.resolved_rescuenet_segmentation_weights_path.name),
        )

        return WorkflowExecutionResult(
            incident_id=incident.id,
            incident_title=incident.title,
            media_asset_id=media_asset.id,
            media_type=ingestion_result.media_type,
            frames_processed=ingestion_result.total_frames_extracted,
            total_detections_found=len(saved_detections),
            unique_person_tracks=len(unique_person_track_ids),
            locations_attached_count=locations_attached_count,
            priority_assessment=priority_result,
            recommendation_outcome=recommendation_outcome,
            saved_records_summary=summary_text,
            scene_context=scene_analysis,
            incident=incident_summary,
            media=media_summary,
            detections=frontend_detections,
            device_info=device_telemetry,
        )
