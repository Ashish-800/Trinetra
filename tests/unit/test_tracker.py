"""
Unit tests for Video Inference and Multi-Object Tracking Service (Phase 5).
Uses deterministic sequences and mocked detections (zero GPU, zero network).
"""
from pathlib import Path
import cv2
import numpy as np
import pytest
from app.schemas.domain import BoundingBoxSchema
from app.services.detector import DetectedObject, MockDetector
from app.services.tracker import AerialIoUTracker, compute_iou
from app.services.video_tracking import VideoTrackingPipeline


def test_compute_iou():
    box1 = BoundingBoxSchema(x1=0, y1=0, x2=10, y2=10)
    box2 = BoundingBoxSchema(x1=0, y1=0, x2=10, y2=10)
    assert compute_iou(box1, box2) == 1.0

    # 50% overlap on 10x10 boxes
    box3 = BoundingBoxSchema(x1=5, y1=0, x2=15, y2=10)
    # intersection = 5*10 = 50. union = 100 + 100 - 50 = 150. iou = 50/150 = 0.3333
    assert abs(compute_iou(box1, box3) - (50 / 150)) < 1e-4

    # No overlap
    box4 = BoundingBoxSchema(x1=20, y1=20, x2=30, y2=30)
    assert compute_iou(box1, box4) == 0.0


def test_tracker_preserves_track_id_across_consecutive_frames():
    """Verifies that an object smoothly moving across frames maintains the exact same track_id."""
    tracker = AerialIoUTracker(iou_threshold=0.20)

    # Frame 0: Person detected at (10, 10, 30, 50)
    det_f0 = [
        DetectedObject.create_safe("person", 0.90, BoundingBoxSchema(x1=10, y1=10, x2=30, y2=50))
    ]
    t_f0 = tracker.update(det_f0, frame_index=0, timestamp_seconds=0.0)
    assert len(t_f0) == 1
    track_id_initial = t_f0[0].track_id
    assert t_f0[0].frame_index == 0
    assert t_f0[0].timestamp_seconds == 0.0
    assert t_f0[0].hits == 1

    # Frame 1: Person moves slightly to (12, 11, 32, 51)
    det_f1 = [
        DetectedObject.create_safe("person", 0.88, BoundingBoxSchema(x1=12, y1=11, x2=32, y2=51))
    ]
    t_f1 = tracker.update(det_f1, frame_index=1, timestamp_seconds=1.0)
    assert len(t_f1) == 1
    assert t_f1[0].track_id == track_id_initial  # Same track ID
    assert t_f1[0].frame_index == 1
    assert t_f1[0].timestamp_seconds == 1.0
    assert t_f1[0].hits == 2

    # Frame 2: Person moves to (14, 13, 34, 53)
    det_f2 = [
        DetectedObject.create_safe("person", 0.85, BoundingBoxSchema(x1=14, y1=13, x2=34, y2=53))
    ]
    t_f2 = tracker.update(det_f2, frame_index=2, timestamp_seconds=2.0)
    assert len(t_f2) == 1
    assert t_f2[0].track_id == track_id_initial
    assert t_f2[0].hits == 3


def test_tracker_handles_missed_detections_and_recovers():
    """
    Simulates a temporary occlusion:
    Frame 0: detected
    Frame 1: missed (e.g. occlusion under tree/smoke)
    Frame 2: re-detected nearby -> re-associated with original track ID.
    """
    tracker = AerialIoUTracker(iou_threshold=0.20, max_lost_frames=3)

    # Frame 0
    det0 = [DetectedObject.create_safe("person", 0.85, BoundingBoxSchema(x1=50, y1=50, x2=70, y2=90))]
    t0 = tracker.update(det0, frame_index=0, timestamp_seconds=0.0)
    track_id = t0[0].track_id

    # Frame 1: Missed / zero detections
    t1 = tracker.update([], frame_index=1, timestamp_seconds=0.5)
    assert len(t1) == 0

    # Frame 2: Re-appears overlapping with previous position
    det2 = [DetectedObject.create_safe("person", 0.82, BoundingBoxSchema(x1=52, y1=51, x2=72, y2=91))]
    t2 = tracker.update(det2, frame_index=2, timestamp_seconds=1.0)
    assert len(t2) == 1
    assert t2[0].track_id == track_id  # Recovered original track ID
    assert t2[0].hits == 2


def test_tracker_terminates_track_after_max_lost_frames():
    """
    Verifies that an unobserved track is terminated after max_lost_frames,
    and any subsequent new detection receives a new track ID.
    """
    tracker = AerialIoUTracker(iou_threshold=0.20, max_lost_frames=2)

    # Frame 0: Track 1 starts
    det0 = [DetectedObject.create_safe("person", 0.9, BoundingBoxSchema(x1=10, y1=10, x2=30, y2=40))]
    t0 = tracker.update(det0, frame_index=0, timestamp_seconds=0.0)
    t1_id = t0[0].track_id

    # Frames 1, 2, 3: Object missing for 3 consecutive frames (> max_lost_frames=2)
    tracker.update([], frame_index=1, timestamp_seconds=0.5)
    tracker.update([], frame_index=2, timestamp_seconds=1.0)
    tracker.update([], frame_index=3, timestamp_seconds=1.5)

    # Frame 4: An object appears at the same location. Since track 1 expired, it receives a new track ID.
    det4 = [DetectedObject.create_safe("person", 0.88, BoundingBoxSchema(x1=10, y1=10, x2=30, y2=40))]
    t4 = tracker.update(det4, frame_index=4, timestamp_seconds=2.0)
    assert len(t4) == 1
    assert t4[0].track_id != t1_id  # New ID assigned
    assert t4[0].hits == 1


def test_avoid_counting_every_frame_as_new_person():
    """
    Safety / Accuracy Rule:
    A video sequence of 10 frames detecting 1 person must count as 1 unique person,
    not 10 people!
    """
    tracker = AerialIoUTracker()
    detected_frames = 10

    for i in range(detected_frames):
        det = [
            DetectedObject.create_safe(
                "person",
                0.90,
                BoundingBoxSchema(x1=100 + i, y1=100, x2=130 + i, y2=160),
            )
        ]
        results = tracker.update(det, frame_index=i, timestamp_seconds=float(i))
        assert len(results) == 1
        assert results[0].track_id == 1

    # Total unique tracks across all 10 frames must be exactly 1
    unique_tracks = tracker.get_all_unique_track_ids()
    assert len(unique_tracks) == 1


def test_video_tracking_pipeline_end_to_end(tmp_path: Path):
    """
    Tests the complete VideoTrackingPipeline:
    Takes a synthetic video, extracts frames, detects objects using MockDetector,
    tracks them across frames, and returns a VideoTrackingSummary.
    """
    # 1. Create a 3-frame synthetic video (3 frames at 10 fps, 64x64)
    video_path = tmp_path / "test_survey.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    for idx in range(3):
        frame = np.full((64, 64, 3), 100 + idx * 10, dtype=np.uint8)
        out.write(frame)
    out.release()

    # 2. Configure MockDetector with 1 stationary person
    mock_objs = [
        DetectedObject.create_safe("person", 0.90, BoundingBoxSchema(x1=20, y1=20, x2=40, y2=50))
    ]
    detector = MockDetector(mock_objects=mock_objs)
    pipeline = VideoTrackingPipeline(
        detector=detector,
        tracker=AerialIoUTracker(),
        default_output_dir=tmp_path / "pipeline_runs",
    )

    # 3. Process video
    summary = pipeline.process_video(
        video_path=video_path,
        sampling_interval_seconds=0.1,  # Sample each frame
        save_annotated_frames=True,
    )

    assert summary.total_frames_extracted == 3
    assert len(summary.frames) == 3
    # Exactly 1 unique person tracked across all 3 frames!
    assert summary.unique_person_tracks == 1
    assert summary.total_unique_tracks == 1

    for fr in summary.frames:
        assert fr.active_person_count == 1
        assert len(fr.detections) == 1
        assert fr.detections[0].track_id == 1
        assert fr.annotated_frame_path is not None
        assert Path(fr.annotated_frame_path).exists()
