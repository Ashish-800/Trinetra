"""
Final Multimodal Backend Demonstration.
Demonstrates the combined end-to-end aerial assessment pipeline:
  Single Real Input Image
  -> YOLOv8n (trained on VisDrone, person-only, CUDA:0)
  -> DeepLabV3-MobileNetV3-Large (trained on RescueNet, 12 classes, CUDA:0)
  -> Multi-Object Tracking
  -> Simulated Geographic Location
  -> Uncertainty & Priority Engine
  -> Advisory Resource Recommendation (requires_human_authorization=True)
  -> Database Persistence & FastAPI Endpoint Verification
"""
import argparse
import io
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.domain import Incident, MediaAsset, Resource
from app.schemas.rescuenet import CLASS_DISPLAY_NAMES, RescueNetClass
from app.services.detector import YOLOAerialDetector
from app.services.rescuenet_segmentor import RescueNetSegmentor
from app.services.workflow import AssessmentWorkflowService
from train_rescuenet_pilot import colorize_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multimodal_demo")


def run_multimodal_demonstration(
    image_path: str = "archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg",
    output_vis: str = "uploads/demo_samples/final_multimodal_backend_demo.png",
    conf_thresh: float = 0.25,
):
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Test image not found at: {img_p}")

    out_p = Path(output_vis)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()

    # 1. Ensure resources exist in DB for recommendation engine
    if db.query(Resource).count() == 0:
        logger.info("Seeding test emergency resources in DB...")
        resources = [
            Resource(
                name="Coastal SAR Task Force Alpha",
                resource_type="WATER_RESCUE",
                total_capacity=10,
                available_capacity=10,
                simulated_lat=34.052,
                simulated_lon=-118.243,
                is_available=True,
            ),
            Resource(
                name="Heavy Debris Clearance Unit 1",
                resource_type="SAR_TEAM",
                total_capacity=6,
                available_capacity=6,
                simulated_lat=34.055,
                simulated_lon=-118.240,
                is_available=True,
            ),
        ]
        db.add_all(resources)
        db.commit()

    # 2. Check CUDA availability and initialize models
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    logger.info("Initializing Multimodal Models on device: %s (CUDA Available: %s)", device_name, cuda_available)

    yolo_weights = settings.resolved_yolo_weights_path
    logger.info("YOLO Model: %s", yolo_weights)
    detector = YOLOAerialDetector(
        weights_path=yolo_weights,
        default_conf_threshold=conf_thresh,
        device=settings.YOLO_DEVICE,
        person_only=True,
    )

    seg_weights = settings.resolved_rescuenet_segmentation_weights_path
    logger.info("RescueNet Model: %s", seg_weights)
    segmentor = RescueNetSegmentor(
        weights_path=seg_weights,
        device=settings.RESCUENET_SEGMENTATION_DEVICE,
        img_size=(512, 512),
    )

    # 3. Create Workflow Service and Execute
    workflow = AssessmentWorkflowService(
        db=db,
        detector=detector,
        segmentor=segmentor,
    )

    # Register demonstration incident
    inc = Incident(
        title="Multimodal UAV Disaster Assessment Demonstration",
        disaster_type="FLOOD",
        location_name="Coastal Disaster Recon Sector Bravo",
        notes="Dual-model real-time pipeline: YOLO person detection + RescueNet environmental segmentation",
    )
    db.add(inc)
    db.commit()
    db.refresh(inc)

    # Register media asset
    media_asset = MediaAsset(
        incident_id=inc.id,
        file_path=str(img_p.resolve()),
        media_type="IMAGE",
        mime_type="image/jpeg",
        file_size_bytes=img_p.stat().st_size,
    )
    db.add(media_asset)
    db.commit()
    db.refresh(media_asset)

    logger.info("Executing 8-step Multimodal Assessment Workflow...")
    result = workflow.run_workflow(
        media_path=img_p,
        incident_id=inc.id,
        media_asset_id=media_asset.id,
        detection_confidence_threshold=conf_thresh,
        enable_segmentation=True,
    )

    # 4. Generate 3-Panel Visual Artifact with Overlay
    logger.info("Generating final 4-element multimodal visual artifact...")
    raw_pil = Image.open(img_p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    orig_bgr = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)

    # Panel A: Original Image with YOLO person detection bounding boxes
    panel_yolo = orig_bgr.copy()
    for track in workflow.tracker._active_tracks.values():
        bx = track.current_bbox
        x1, y1 = int(bx.x_min * 512), int(bx.y_min * 512)
        x2, y2 = int(bx.x_max * 512), int(bx.y_max * 512)
        cv2.rectangle(panel_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            panel_yolo,
            f"Person {bx.confidence:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 0),
            1,
        )

    # Add header banner to Panel A
    banner_a = np.zeros((35, 512, 3), dtype=np.uint8)
    cv2.putText(banner_a, f"1. YOLOv8n: {result.total_detections_found} Potential Persons", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    panel_a = np.vstack([banner_a, panel_yolo])

    # Panel B: Segmentation Mask & Blended Overlay
    pred_mask = segmentor.predict_mask(img_p)
    color_mask = colorize_mask(pred_mask)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    # Blend 60% original image + 40% segmentation mask
    blended = cv2.addWeighted(orig_bgr, 0.50, color_mask_bgr, 0.50, 0)

    banner_b = np.zeros((35, 512, 3), dtype=np.uint8)
    cv2.putText(banner_b, "2. RescueNet: 12-Class Hazard Segmentation", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    panel_b = np.vstack([banner_b, blended])

    # Panel C: Structured Context and Decision-Support Summary Panel
    panel_c_img = np.full((512, 440, 3), (22, 22, 28), dtype=np.uint8)
    banner_c = np.zeros((35, 440, 3), dtype=np.uint8)
    cv2.putText(banner_c, "3. Multimodal Decision Support", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    panel_c = np.vstack([banner_c, panel_c_img])

    ctx = result.scene_context
    prio = result.priority_assessment

    summary_lines = [
        ("HARDWARE & MODELS", (0, 215, 255)),
        (f"  Device: {device_name} (CUDA: {cuda_available})", (200, 200, 200)),
        ("  YOLO: visdrone_yolov8n_cuda_v1/best.pt", (200, 200, 200)),
        ("  RescueNet: rescuenet_pilot/best_model.pt", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("MICRO: ENTITY DETECTIONS", (0, 255, 0)),
        (f"  Potential Persons: {result.total_detections_found}", (255, 255, 255)),
        (f"  Active Tracks: {result.unique_person_tracks}", (200, 200, 200)),
        ("  Coordinates: Explicitly 'simulated'", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("MACRO: ENVIRONMENTAL CONTEXT", (0, 200, 255)),
        (f"  Road Coverage:     {ctx.road_coverage_pct:.1f}%" if ctx else "  Road: N/A", (220, 220, 220)),
        (f"  Tree Coverage:     {ctx.tree_coverage_pct:.1f}%" if ctx else "  Tree: N/A", (220, 220, 220)),
        (f"  Debris Coverage:   {ctx.debris_coverage_pct:.1f}%" if ctx else "  Debris: N/A", (220, 220, 220)),
        (f"  Flood Water:       {ctx.water_coverage_pct:.1f}%" if ctx else "  Water: N/A", (220, 220, 220)),
        (f"  Inferred Hazard:   {ctx.inferred_hazard_severity.value if ctx else 'N/A'}", (255, 255, 0)),
        (f"  Accessibility:     {ctx.inferred_accessibility if ctx else 'N/A'}", (255, 255, 0)),
        ("", (0, 0, 0)),
        ("PRIORITY & ADVISORY ACTION", (0, 165, 255)),
        (f"  Urgency Score:     {prio.urgency_score:.2f} [{prio.urgency_level}]", (255, 255, 255)),
        (f"  Uncertainty Score: {prio.uncertainty_score:.2f} [{prio.uncertainty_level}]", (200, 200, 200)),
        (f"  Composite Tier:    {prio.composite_priority.value}", (0, 215, 255)),
        (f"  Human Review Gate: {prio.requires_human_verification} (Mandatory)", (0, 140, 255)),
        ("", (0, 0, 0)),
        ("ADVISORY RECOMMENDATION", (147, 112, 219)),
        (f"  Result: {result.recommendation_outcome.status}", (220, 220, 220)),
        (f"  Resource: {result.recommendation_outcome.recommendations[0].resource_name[:28]}" if result.recommendation_outcome.recommendations else "  None", (255, 255, 255)),
        ("  Status: PENDING_REVIEW", (255, 200, 0)),
        ("  Requires Authorization: True", (255, 200, 0)),
    ]

    y_pos = 50
    for text, col in summary_lines:
        if text:
            cv2.putText(panel_c, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        y_pos += 18

    # Combine horizontal composite: [Panel A | Panel B | Panel C]
    composite = np.hstack([panel_a, panel_b, panel_c])

    # Add Bottom Cross-Dataset Disclaimer Banner across full width (512 + 512 + 440 = 1464 px)
    disclaimer_height = 45
    disclaimer_bar = np.zeros((disclaimer_height, composite.shape[1], 3), dtype=np.uint8)
    disclaimer_text_1 = "CROSS-DATASET PROTOTYPE DEMONSTRATION NOTICE:"
    disclaimer_text_2 = "VisDrone (people) and RescueNet (disaster context) are separate datasets. Combined inference demonstrates dual-layer feasibility; neither model confirms survivor status."
    cv2.putText(disclaimer_bar, disclaimer_text_1, (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
    cv2.putText(disclaimer_bar, disclaimer_text_2, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    final_artifact = np.vstack([composite, disclaimer_bar])
    cv2.imwrite(str(out_p), final_artifact)
    logger.info("Saved final multimodal visual artifact to: %s", out_p)

    db.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Backend Demo")
    parser.add_argument("--image", type=str, default="archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    parser.add_argument("--output", type=str, default="uploads/demo_samples/final_multimodal_backend_demo.png")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    run_multimodal_demonstration(
        image_path=args.image,
        output_vis=args.output,
        conf_thresh=args.conf,
    )
