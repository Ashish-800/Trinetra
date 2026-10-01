"""
Multi-Object Tracking Evaluator.
Measures track consistency, duplicate-alert count under defined rules, and tracking failure modes.
"""
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.schemas.domain import BoundingBoxSchema
from app.services.detector import DetectedObject
from app.services.tracker import AerialIoUTracker, BaseTracker


class TrackingMetrics(BaseModel):
    """Evaluation summary metrics for video multi-object tracking."""
    status: str = "evaluated"
    total_frames_evaluated: int
    ground_truth_target_count: int
    raw_per_frame_detection_alerts: int = Field(
        ..., description="Alert count under naive per-frame detection baseline (fires on every frame detection)"
    )
    tracking_deduplicated_alerts: int = Field(
        ..., description="Alert count under proposed tracking workflow (1 alert per unique track)"
    )
    duplicate_alert_reduction_pct: float = Field(
        ..., description="Percentage of duplicate alert spam eliminated by tracking deduplication"
    )
    track_consistency: float = Field(
        ..., description="Fraction of frames where active targets maintain uninterrupted track ID consistency [0.0 - 1.0]"
    )
    id_switches: int = Field(..., description="Number of identity switch failures across frames")
    false_terminations: int = Field(..., description="Number of times an active target track dropped prematurely")
    phantom_tracks: int = Field(..., description="Number of tracks created from noise or false positive detections")
    test_scenario_description: str


class GroundTruthTrajectory:
    """Represents the ground truth spatial location of a target over time."""
    def __init__(self, target_id: int, trajectory: Dict[int, BoundingBoxSchema]):
        self.target_id = target_id
        self.trajectory = trajectory  # frame_idx -> BoundingBoxSchema


class TrackingEvaluator:
    """
    Evaluator for aerial video multi-object tracking.
    Compares naive per-frame detection against spatial-temporal tracking.
    """

    def __init__(self, tracker: Optional[BaseTracker] = None, iou_match_threshold: float = 0.30):
        self.tracker = tracker or AerialIoUTracker()
        self.iou_match_threshold = iou_match_threshold

    @staticmethod
    def _compute_iou(box_a: BoundingBoxSchema, box_b: BoundingBoxSchema) -> float:
        inter_x1 = max(box_a.x1, box_b.x1)
        inter_y1 = max(box_a.y1, box_b.y1)
        inter_x2 = min(box_a.x2, box_b.x2)
        inter_y2 = min(box_a.y2, box_b.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
        area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)

        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_sequence(
        self,
        frames_detections: List[List[DetectedObject]],
        ground_truth_trajectories: List[GroundTruthTrajectory],
        scenario_description: str = "Synthetic UAV Video Sweep",
    ) -> TrackingMetrics:
        """
        Runs the tracker sequentially over a multi-frame detection stream.
        Records:
        - track consistency across frames
        - duplicate alert count under raw detection baseline vs proposed tracking
        - tracking failure modes (id_switches, false_terminations, phantom_tracks)
        """
        total_frames = len(frames_detections)
        if total_frames == 0:
            return TrackingMetrics(
                status="metric_cannot_yet_be_calculated: no frames provided",
                total_frames_evaluated=0,
                ground_truth_target_count=len(ground_truth_trajectories),
                raw_per_frame_detection_alerts=0,
                tracking_deduplicated_alerts=0,
                duplicate_alert_reduction_pct=0.0,
                track_consistency=0.0,
                id_switches=0,
                false_terminations=0,
                phantom_tracks=0,
                test_scenario_description=scenario_description,
            )

        # Baseline: Raw per-frame detection count
        raw_detection_count = sum(len(frame_dets) for frame_dets in frames_detections)

        # Proposed: Run tracking across frames
        self.tracker.reset()
        all_unique_tracks = set()

        # Mapping: gt_target_id -> last_assigned_tracker_id
        gt_track_assignment: Dict[int, int] = {}
        target_frames_present: Dict[int, int] = {t.target_id: 0 for t in ground_truth_trajectories}
        target_consistent_frames: Dict[int, int] = {t.target_id: 0 for t in ground_truth_trajectories}
        id_switches = 0
        matched_tracker_ids = set()

        for frame_idx, detections in enumerate(frames_detections):
            # Step tracker
            active_tracks = self.tracker.update(detections, frame_number=frame_idx)
            for trk in active_tracks:
                all_unique_tracks.add(trk.track_id)

            # Match active tracks to ground truth for this frame
            frame_gt = {
                t.target_id: t.trajectory[frame_idx]
                for t in ground_truth_trajectories
                if frame_idx in t.trajectory
            }

            for gt_id, gt_box in frame_gt.items():
                target_frames_present[gt_id] += 1
                best_iou = 0.0
                best_trk_id = None

                for trk in active_tracks:
                    iou = self._compute_iou(trk.bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_trk_id = trk.track_id

                if best_iou >= self.iou_match_threshold and best_trk_id is not None:
                    matched_tracker_ids.add(best_trk_id)
                    prev_id = gt_track_assignment.get(gt_id)
                    if prev_id is None:
                        # Initial assignment
                        gt_track_assignment[gt_id] = best_trk_id
                        target_consistent_frames[gt_id] += 1
                    elif prev_id == best_trk_id:
                        # Consistent tracking
                        target_consistent_frames[gt_id] += 1
                    else:
                        # ID Switch failure!
                        id_switches += 1
                        gt_track_assignment[gt_id] = best_trk_id

        # Calculate consistency ratio
        total_gt_frames = sum(target_frames_present.values())
        total_consistent_frames = sum(target_consistent_frames.values())
        consistency_ratio = (
            round(total_consistent_frames / total_gt_frames, 4) if total_gt_frames > 0 else 1.0
        )

        # Deduplicated alerts under proposed workflow
        unique_person_tracks = len(all_unique_tracks)
        
        # Calculate reduction percentage
        if raw_detection_count > 0:
            reduction_pct = round((1.0 - (unique_person_tracks / raw_detection_count)) * 100.0, 2)
        else:
            reduction_pct = 0.0

        # Phantom tracks: tracks created that never matched any ground truth target
        phantom_tracks = max(0, len(all_unique_tracks) - len(matched_tracker_ids))

        # False terminations: when number of unique tracks assigned to a single uninterrupted GT target > 1
        false_terminations = id_switches  # each ID switch represents a dropped track and a new initiation

        return TrackingMetrics(
            status="evaluated",
            total_frames_evaluated=total_frames,
            ground_truth_target_count=len(ground_truth_trajectories),
            raw_per_frame_detection_alerts=raw_detection_count,
            tracking_deduplicated_alerts=unique_person_tracks,
            duplicate_alert_reduction_pct=reduction_pct,
            track_consistency=consistency_ratio,
            id_switches=id_switches,
            false_terminations=false_terminations,
            phantom_tracks=phantom_tracks,
            test_scenario_description=scenario_description,
        )

    @classmethod
    def generate_synthetic_benchmark_sequence(
        cls, num_frames: int = 30, num_targets: int = 2
    ) -> Tuple[List[List[DetectedObject]], List[GroundTruthTrajectory]]:
        """
        Creates a reproducible, deterministic synthetic UAV video trajectory with multiple targets.
        Used for standardized regression testing.
        """
        frames_detections: List[List[DetectedObject]] = []
        gt_trajectories: List[GroundTruthTrajectory] = []

        # Target 1: Person walking diagonally
        t1_boxes: Dict[int, BoundingBoxSchema] = {}
        # Target 2: Person stationary on rooftop
        t2_boxes: Dict[int, BoundingBoxSchema] = {}

        for f in range(num_frames):
            frame_dets = []
            # T1 moves (x + 2px per frame)
            b1 = BoundingBoxSchema(x1=100.0 + f * 2.0, y1=150.0 + f * 1.0, x2=140.0 + f * 2.0, y2=210.0 + f * 1.0)
            t1_boxes[f] = b1
            frame_dets.append(DetectedObject.create_safe("person", 0.90, b1))

            if num_targets >= 2:
                # T2 stationary
                b2 = BoundingBoxSchema(x1=300.0, y1=200.0, x2=340.0, y2=270.0)
                t2_boxes[f] = b2
                frame_dets.append(DetectedObject.create_safe("person", 0.88, b2))

            frames_detections.append(frame_dets)

        gt_trajectories.append(GroundTruthTrajectory(target_id=1, trajectory=t1_boxes))
        if num_targets >= 2:
            gt_trajectories.append(GroundTruthTrajectory(target_id=2, trajectory=t2_boxes))

        return frames_detections, gt_trajectories
