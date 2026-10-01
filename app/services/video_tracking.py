"""
Video Inference and Tracking Pipeline.
Connects Media Ingestion (frame extraction) with Detection and Tracking services.
Preserves frame indices and timestamps, associating detections with persistent track IDs.
"""
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from pydantic import BaseModel, Field
from app.services.detector import BaseDetector, DetectedObject
from app.services.ingestion import MediaIngestionService
from app.services.tracker import AerialIoUTracker, BaseTracker, TrackedObject


class FrameTrackingResult(BaseModel):
    """Output for a single video frame with tracked objects."""
    frame_index: int
    timestamp_seconds: float
    source_frame_path: str
    annotated_frame_path: Optional[str] = None
    detections: List[TrackedObject]
    active_person_count: int


class VideoTrackingSummary(BaseModel):
    """Overall summary of video inference and tracking run."""
    video_source_path: str
    total_frames_extracted: int
    duration_seconds: Optional[float] = None
    frames: List[FrameTrackingResult]
    unique_person_tracks: int = Field(..., description="Count of distinct person track IDs (avoids duplicate frame counts)")
    total_unique_tracks: int = Field(..., description="Total unique object tracks across all classes")


class VideoTrackingPipeline:
    """
    End-to-end pipeline linking video ingestion, detection, and tracking.
    Enforces Safety Rule 1: Tracks do not prove identity, life, or survivor status.
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: Optional[BaseTracker] = None,
        ingestion_service: Optional[MediaIngestionService] = None,
        default_output_dir: Optional[Path] = None,
    ):
        self.detector = detector
        self.tracker = tracker or AerialIoUTracker()
        self.ingestion_service = ingestion_service or MediaIngestionService()
        self.output_dir = default_output_dir or Path("uploads/tracked_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_video(
        self,
        video_path: Union[str, Path],
        sampling_interval_seconds: float = 1.0,
        conf_threshold: Optional[float] = None,
        save_annotated_frames: bool = False,
    ) -> VideoTrackingSummary:
        """
        Executes frame extraction, detection, and tracking across a video file.
        
        Args:
            video_path: Path to the local UAV video.
            sampling_interval_seconds: Frame extraction sampling interval (default 1.0 sec).
            conf_threshold: Optional detection confidence threshold.
            save_annotated_frames: Whether to save annotated frames with tracking IDs to disk.
        """
        v_path = Path(video_path).resolve()
        run_dir = self.output_dir / v_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Ingest video and extract sampled frames
        ingestion_result = self.ingestion_service.ingest(
            file_path=v_path,
            interval_seconds=sampling_interval_seconds,
            custom_output_dir=run_dir / "raw_frames",
        )

        # Reset tracker for the new video stream
        self.tracker.reset()

        frame_results: List[FrameTrackingResult] = []
        unique_person_track_ids: set[int] = set()

        # Step 2: Iterate over extracted frames in temporal order
        for frame_info in ingestion_result.frames:
            det_result = self.detector.predict(
                image_input=frame_info.output_path,
                conf_threshold=conf_threshold,
            )

            # Step 3: Pass detections into the tracker
            tracked_objs = self.tracker.update(
                detections=det_result.objects,
                frame_index=frame_info.frame_index,
                timestamp_seconds=frame_info.timestamp_seconds,
            )

            # Track unique person IDs
            current_frame_person_count = 0
            for tobj in tracked_objs:
                if tobj.class_name in ("person", "potential survivor"):
                    unique_person_track_ids.add(tobj.track_id)
                    current_frame_person_count += 1

            # Step 4: Optional annotation with track IDs
            annotated_frame_path = None
            if save_annotated_frames:
                annotated_path = run_dir / "annotated" / f"track_{frame_info.frame_index:06d}.jpg"
                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                self._annotate_tracked_frame(frame_info.output_path, tracked_objs, annotated_path)
                annotated_frame_path = str(annotated_path)

            frame_results.append(
                FrameTrackingResult(
                    frame_index=frame_info.frame_index,
                    timestamp_seconds=frame_info.timestamp_seconds,
                    source_frame_path=frame_info.output_path,
                    annotated_frame_path=annotated_frame_path,
                    detections=tracked_objs,
                    active_person_count=current_frame_person_count,
                )
            )

        return VideoTrackingSummary(
            video_source_path=str(v_path),
            total_frames_extracted=ingestion_result.total_frames_extracted,
            duration_seconds=ingestion_result.duration_seconds,
            frames=frame_results,
            unique_person_tracks=len(unique_person_track_ids),
            total_unique_tracks=len(self.tracker.get_all_unique_track_ids()),
        )

    def _annotate_tracked_frame(
        self,
        img_path: str,
        tracked_objects: List[TrackedObject],
        output_path: Path,
    ):
        """Draws bounding boxes, track IDs, and confidence scores on frame."""
        img = cv2.imread(img_path)
        if img is None:
            return

        for tobj in tracked_objects:
            x1, y1 = int(tobj.bbox.x1), int(tobj.bbox.y1)
            x2, y2 = int(tobj.bbox.x2), int(tobj.bbox.y2)

            color = (0, 0, 255) if tobj.class_name in ("person", "potential survivor") else (255, 165, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tobj.track_id} {tobj.class_name} ({tobj.confidence:.2f})"
            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_path), img)
