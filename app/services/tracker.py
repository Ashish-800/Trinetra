"""
Multi-Object Tracking Service for Aerial Video Sequences.
Provides decoupled tracking interfaces, handling missed detections,
track termination, and preventing duplicate entity counts.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.domain import BoundingBoxSchema
from app.services.detector import DetectedObject


class TrackedObject(BaseModel):
    """Detection associated with a temporal tracking ID."""
    track_id: int = Field(..., description="Unique track identifier in video stream")
    class_name: str
    confidence: float
    bbox: BoundingBoxSchema
    frame_index: int
    timestamp_seconds: float
    hits: int = Field(..., description="Number of frames this track has been continuously detected")
    is_provisional: bool = Field(False, description="True if track has not met minimum hit threshold")


class TrackState:
    """Internal lifecycle state of an active or dormant track."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        initial_bbox: BoundingBoxSchema,
        frame_idx: int,
        timestamp: float,
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.current_bbox = initial_bbox
        self.first_frame_idx = frame_idx
        self.last_frame_idx = frame_idx
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.hits = 1
        self.time_since_update = 0
        self.is_active = True

    def update(self, new_bbox: BoundingBoxSchema, frame_idx: int, timestamp: float):
        """Updates track with a new matched detection."""
        self.current_bbox = new_bbox
        self.last_frame_idx = frame_idx
        self.last_timestamp = timestamp
        self.hits += 1
        self.time_since_update = 0
        self.is_active = True

    def mark_missed(self):
        """Marks track as unobserved in the current frame."""
        self.time_since_update += 1


def compute_iou(boxA: BoundingBoxSchema, boxB: BoundingBoxSchema) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = max(0.0, boxA.x2 - boxA.x1) * max(0.0, boxA.y2 - boxA.y1)
    areaB = max(0.0, boxB.x2 - boxB.x1) * max(0.0, boxB.y2 - boxB.y1)

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


class BaseTracker(ABC):
    """Abstract tracking interface decoupling tracking implementations."""

    @abstractmethod
    def update(
        self,
        detections: List[DetectedObject],
        frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        frame_number: Optional[int] = None,
        **kwargs,
    ) -> List[TrackedObject]:
        """Associates incoming frame detections with persistent tracks."""
        pass

    @abstractmethod
    def reset(self):
        """Resets tracker state between video streams."""
        pass

    @abstractmethod
    def get_all_unique_track_ids(self) -> set[int]:
        """Returns all unique track IDs created across the video lifecycle."""
        pass


class AerialIoUTracker(BaseTracker):
    """
    Spatial IoU-based multi-object tracker for aerial video feeds.
    Maintains track persistence, handles temporary occlusions / missed detections,
    and terminates tracks inactive for longer than max_lost_frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_lost_frames: int = 5,
        min_hits_to_confirm: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self._next_track_id = 1
        self._active_tracks: Dict[int, TrackState] = {}
        self._all_created_track_ids: set[int] = set()

    def reset(self):
        """Clears all tracking memory."""
        self._next_track_id = 1
        self._active_tracks.clear()
        self._all_created_track_ids.clear()

    def get_all_unique_track_ids(self) -> set[int]:
        return set(self._all_created_track_ids)

    def update(
        self,
        detections: List[DetectedObject],
        frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        frame_number: Optional[int] = None,
        **kwargs,
    ) -> List[TrackedObject]:
        """
        Associates frame detections to existing tracks using greedy IoU matching.
        Terminates tracks with time_since_update > max_lost_frames.
        """
        f_idx = frame_index if frame_index is not None else (frame_number if frame_number is not None else kwargs.get("frame_index", kwargs.get("frame_number", 0)))
        ts = timestamp_seconds if timestamp_seconds is not None else float(f_idx)
        # Step 1: Match incoming detections to active tracks
        unmatched_detections = list(range(len(detections)))
        matched_tracks: set[int] = set()
        matches: List[tuple[int, int]] = []  # (track_id, det_idx)

        # Build cost / IoU matrix for active tracks
        active_track_ids = [tid for tid, t in self._active_tracks.items() if t.is_active]

        for tid in active_track_ids:
            track = self._active_tracks[tid]
            best_iou = 0.0
            best_det_idx = None

            for d_idx in unmatched_detections:
                det = detections[d_idx]
                # Only match same class (e.g. person with person)
                if det.class_name != track.class_name:
                    continue

                iou = compute_iou(track.current_bbox, det.bbox)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_det_idx is not None:
                matches.append((tid, best_det_idx))
                matched_tracks.add(tid)
                unmatched_detections.remove(best_det_idx)

        # Step 2: Update matched tracks
        for tid, det_idx in matches:
            det = detections[det_idx]
            self._active_tracks[tid].update(det.bbox, f_idx, ts)

        # Step 3: Handle missed detections for unmatched active tracks
        unmatched_track_ids = set(active_track_ids) - matched_tracks
        for tid in unmatched_track_ids:
            self._active_tracks[tid].mark_missed()
            # Track termination condition
            if self._active_tracks[tid].time_since_update > self.max_lost_frames:
                self._active_tracks[tid].is_active = False

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_id = self._next_track_id
            self._next_track_id += 1
            new_track = TrackState(
                track_id=new_id,
                class_name=det.class_name,
                initial_bbox=det.bbox,
                frame_idx=f_idx,
                timestamp=ts,
            )
            self._active_tracks[new_id] = new_track
            self._all_created_track_ids.add(new_id)
            matches.append((new_id, det_idx))

        # Step 5: Construct TrackedObject outputs for current frame
        tracked_results: List[TrackedObject] = []
        for tid, det_idx in matches:
            det = detections[det_idx]
            track = self._active_tracks[tid]
            is_provisional = track.hits < self.min_hits_to_confirm

            tracked_results.append(
                TrackedObject(
                    track_id=tid,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    hits=track.hits,
                    is_provisional=is_provisional,
                )
            )

        return tracked_results
