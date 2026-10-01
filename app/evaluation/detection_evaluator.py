"""
Person Detection Evaluator.
Computes precision, recall, F1, AP@50, and processing latency against aerial imagery benchmarks.
Strictly adheres to the Project Safety Constitution: never fabricates metrics.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from app.schemas.domain import BoundingBoxSchema
from app.services.detector import BaseDetector, DetectedObject


class GroundTruthBox(BaseModel):
    """Ground truth target bounding box with category label."""
    class_name: str
    bbox: BoundingBoxSchema
    is_person: bool = True


class DetectionMetrics(BaseModel):
    """Evaluation summary metrics for person detection."""
    status: str = "evaluated"
    dataset_name: str
    sample_count: int
    iou_threshold: float = 0.50
    confidence_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_ground_truth: int
    total_predictions: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    average_precision_50: Optional[float] = Field(
        None, description="Average Precision at IoU 0.50 (AP@50) if PR-curve can be calculated"
    )
    mean_latency_ms: Optional[float] = Field(
        None, description="Average per-image inference and parsing time in milliseconds"
    )
    test_environment: str = Field(
        default="Documented Setup: CPU/Host inference, Python runtime",
        description="Hardware and runtime description"
    )
    notes: Optional[str] = None


class DetectionEvaluator:
    """
    Evaluator for aerial person/pedestrian detection.
    Supports VisDrone 2019-DET format and custom ground-truth bounding box sets.
    """

    VISDRONE_PERSON_CATEGORIES = {1, 2}  # 1: pedestrian, 2: people

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box_a: BoundingBoxSchema, box_b: BoundingBoxSchema) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
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
        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area

    @classmethod
    def load_visdrone_annotations(cls, annotation_path: Union[str, Path]) -> List[GroundTruthBox]:
        """
        Parses VisDrone annotation text file:
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        Filters for category 1 (pedestrian) and 2 (people).
        """
        path = Path(annotation_path)
        if not path.exists():
            return []

        gt_boxes = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        w = float(parts[2])
                        h = float(parts[3])
                        category = int(parts[5])
                        
                        if category in cls.VISDRONE_PERSON_CATEGORIES:
                            gt_boxes.append(
                                GroundTruthBox(
                                    class_name="person",
                                    bbox=BoundingBoxSchema(x1=x, y1=y, x2=x + w, y2=y + h),
                                    is_person=True,
                                )
                            )
                    except (ValueError, IndexError):
                        continue
        return gt_boxes

    def evaluate_predictions(
        self,
        ground_truth_by_image: Dict[str, List[GroundTruthBox]],
        predictions_by_image: Dict[str, List[DetectedObject]],
        confidence_threshold: float = 0.25,
        latency_measurements_ms: Optional[List[float]] = None,
        dataset_name: str = "Benchmark Evaluation Set",
    ) -> DetectionMetrics:
        """
        Evaluates predictions against ground truth bounding boxes.
        Calculates TP, FP, FN, Precision, Recall, F1, and AP@50.
        """
        total_gt = sum(len(boxes) for boxes in ground_truth_by_image.values())
        if total_gt == 0:
            return DetectionMetrics(
                status="metric_cannot_yet_be_calculated: ground truth annotations not available",
                dataset_name=dataset_name,
                sample_count=len(ground_truth_by_image),
                iou_threshold=self.iou_threshold,
                confidence_threshold=confidence_threshold,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                total_ground_truth=0,
                total_predictions=sum(len(p) for p in predictions_by_image.values()),
                notes="No ground truth target boxes available in dataset slice.",
            )

        tp_count = 0
        fp_count = 0
        fn_count = 0

        # For AP calculation: collect (confidence, is_tp) pairs
        detection_eval_records: List[Tuple[float, bool]] = []

        for image_key, gt_boxes in ground_truth_by_image.items():
            preds = predictions_by_image.get(image_key, [])
            # Filter by confidence and sort descending
            valid_preds = [p for p in preds if p.confidence >= confidence_threshold]
            valid_preds.sort(key=lambda x: x.confidence, reverse=True)

            matched_gt_indices = set()

            for pred in valid_preds:
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gt_boxes):
                    if idx in matched_gt_indices:
                        continue
                    iou = self.compute_iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= self.iou_threshold and best_gt_idx != -1:
                    tp_count += 1
                    matched_gt_indices.add(best_gt_idx)
                    detection_eval_records.append((pred.confidence, True))
                else:
                    fp_count += 1
                    detection_eval_records.append((pred.confidence, False))

            # Unmatched ground truth are false negatives
            unmatched_gt = len(gt_boxes) - len(matched_gt_indices)
            fn_count += unmatched_gt

        # Compute standard metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute AP@50 (11-point interpolation)
        ap_50 = self._calculate_ap(detection_eval_records, total_gt)

        mean_latency = None
        if latency_measurements_ms:
            mean_latency = sum(latency_measurements_ms) / len(latency_measurements_ms)

        return DetectionMetrics(
            status="evaluated",
            dataset_name=dataset_name,
            sample_count=len(ground_truth_by_image),
            iou_threshold=self.iou_threshold,
            confidence_threshold=confidence_threshold,
            true_positives=tp_count,
            false_positives=fp_count,
            false_negatives=fn_count,
            total_ground_truth=total_gt,
            total_predictions=tp_count + fp_count,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            average_precision_50=round(ap_50, 4) if ap_50 is not None else None,
            mean_latency_ms=round(mean_latency, 2) if mean_latency is not None else None,
        )

    def evaluate_detector_on_dataset(
        self,
        detector: BaseDetector,
        image_paths: List[Path],
        annotations_dir: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        dataset_name: str = "Held-out Test Slice",
    ) -> DetectionMetrics:
        """
        Runs detector on real or mock images and matches against corresponding VisDrone annotations.
        Records exact execution latency per frame.
        """
        if not image_paths:
            return DetectionMetrics(
                status="metric_cannot_yet_be_calculated: no test images provided",
                dataset_name=dataset_name,
                sample_count=0,
                iou_threshold=self.iou_threshold,
                confidence_threshold=confidence_threshold,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                total_ground_truth=0,
                total_predictions=0,
                notes="Provided image list is empty.",
            )

        gt_by_image: Dict[str, List[GroundTruthBox]] = {}
        preds_by_image: Dict[str, List[DetectedObject]] = {}
        latencies: List[float] = []

        for img_path in image_paths:
            stem = img_path.stem
            # Check for annotation file
            if annotations_dir:
                ann_file = annotations_dir / f"{stem}.txt"
                if ann_file.exists():
                    gt_by_image[stem] = self.load_visdrone_annotations(ann_file)
                else:
                    gt_by_image[stem] = []
            else:
                gt_by_image[stem] = []

            # Measure inference time
            t0 = time.perf_counter()
            try:
                preds = detector.detect(img_path, confidence_threshold=confidence_threshold)
            except Exception:
                preds = []
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            # Filter for person/potential survivor classes
            person_preds = [p for p in preds if p.class_name.lower() in ("person", "pedestrian", "people")]
            preds_by_image[stem] = person_preds

        return self.evaluate_predictions(
            ground_truth_by_image=gt_by_image,
            predictions_by_image=preds_by_image,
            confidence_threshold=confidence_threshold,
            latency_measurements_ms=latencies,
            dataset_name=dataset_name,
        )

    @staticmethod
    def _calculate_ap(detection_records: List[Tuple[float, bool]], total_gt: int) -> Optional[float]:
        """Calculates Average Precision using 11-point interpolation."""
        if total_gt == 0 or not detection_records:
            return 0.0

        # Sort all predictions by confidence descending
        detection_records.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for _, is_tp in detection_records:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt
            precisions.append(precision)
            recalls.append(recall)

        # 11-point interpolated AP
        ap = 0.0
        for recall_threshold in [i / 10.0 for i in range(11)]:
            # Find maximum precision for any recall >= recall_threshold
            qualifying_precisions = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
            max_p = max(qualifying_precisions) if qualifying_precisions else 0.0
            ap += max_p / 11.0

        return ap
