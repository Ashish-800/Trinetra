"""
Demonstration and Manual Testing Script for Phase 9 Application Workflow Service.
Runs the complete 8-step controlled assessment workflow.

Usage:
  # Run end-to-end with automatic synthetic disaster media
  python test_workflow_demo.py

  # Run with custom image or video file
  python test_workflow_demo.py --media path/to/aerial_image.jpg --hazard HIGH --accessible false
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

from app.db.session import SessionLocal, init_db
from app.schemas.domain import IncidentCreate
from app.schemas.location import DroneTelemetryMetadata, LocationSource
from app.schemas.priority import HazardSeverity
from app.services.detector import MockDetector
from app.services.workflow import AssessmentWorkflowService


def generate_sample_disaster_image(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((600, 800, 3), (120, 140, 110), dtype=np.uint8)
    # Simulated flooded road and building
    cv2.rectangle(img, (100, 100), (350, 400), (80, 80, 80), -1)
    cv2.rectangle(img, (0, 450), (800, 600), (180, 120, 50), -1)  # Flood water
    cv2.putText(img, "DISASTER RECON FLIGHT [SIMULATED]", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(output_path), img)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test End-to-End Disaster UAV Assessment Workflow")
    parser.add_argument("--media", type=str, default=None, help="Path to aerial image or video")
    parser.add_argument("--hazard", type=str, default="HIGH", help="Hazard severity: NONE, LOW, MODERATE, HIGH, CRITICAL")
    parser.add_argument("--accessible", type=str, default="false", help="Road accessibility: true, false, or none")
    parser.add_argument("--yolo", action="store_true", help="Use real trained YOLO detector instead of MockDetector")
    parser.add_argument("--rescuenet-mask", type=str, default=None, help="Path to RescueNet segmentation mask for disaster context")
    parser.add_argument("--segmentation", action="store_true", help="Run real trained DeepLabV3 RescueNet semantic segmentation model")
    parser.add_argument("--save-vis", type=str, default="uploads/demo_samples/workflow_demo_visual_comparison.png", help="Path to save visual comparison composite")
    args = parser.parse_args()

    # Initialize SQLite database tables
    init_db()
    db = SessionLocal()

    if args.media:
        media_path = Path(args.media)
    else:
        print("[*] Generating synthetic disaster aerial image...")
        media_path = generate_sample_disaster_image(Path("uploads/demo_samples/workflow_disaster_recon.jpg"))

    hazard_sev = HazardSeverity(args.hazard.upper()) if args.hazard else HazardSeverity.NONE
    access_val = None
    if args.accessible.lower() in ("true", "1"):
        access_val = True
    elif args.accessible.lower() in ("false", "0"):
        access_val = False

    from app.core.config import settings

    if args.yolo:
        from app.services.detector import YOLOAerialDetector
        detector = YOLOAerialDetector(
            weights_path=settings.resolved_yolo_weights_path,
            default_conf_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
            device=settings.YOLO_DEVICE,
            person_only=settings.YOLO_PERSON_ONLY,
        )
    else:
        detector = MockDetector()

    segmentor = None
    if args.segmentation:
        from app.services.rescuenet_segmentor import RescueNetSegmentor
        segmentor = RescueNetSegmentor(
            weights_path=settings.resolved_rescuenet_segmentation_weights_path,
            device=settings.RESCUENET_SEGMENTATION_DEVICE,
        )
        print(f"[*] Initialized RescueNetSegmentor on {segmentor.device} using weights: {settings.resolved_rescuenet_segmentation_weights_path}")

    workflow = AssessmentWorkflowService(
        db=db,
        detector=detector,
        segmentor=segmentor,
    )

    print("=" * 70)
    print("PHASE 9: APPLICATION WORKFLOW SERVICE (CONTROLLED 8-STEP PIPELINE)")
    print("=" * 70)
    print("Executing steps:")
    print("  1. Create or select an incident")
    print("  2. Register image/video media")
    print("  3. Run detection (YOLO potential-person detection)")
    print("  4. Track video detections where applicable")
    print("  5. Attach simulated location data")
    print("  6. Calculate urgency and verification status (fusing environmental context)")
    print("  7. Generate resource recommendations (advisory)")
    print("  8. Save all records in database")
    print("-" * 70)

    result = workflow.run_workflow(
        media_path=media_path,
        new_incident=IncidentCreate(
            title="River Valley Flash Flood Emergency",
            disaster_type="FLOOD",
            location_name="Zone Delta",
            notes="Rapid water ingress following heavy rainfall.",
        ),
        telemetry=DroneTelemetryMetadata(
            drone_latitude=34.0522,
            drone_longitude=-118.2437,
            altitude_agl_m=45.0,
            gimbal_pitch_deg=-60.0,
            gimbal_yaw_deg=15.0,
            source=LocationSource.SIMULATED,
        ),
        hazard_severity=hazard_sev,
        is_accessible=access_val,
        rescuenet_mask_path=args.rescuenet_mask,
        enable_segmentation=args.segmentation,
    )

    print("\n[WORKFLOW EXECUTION RESULT]")
    print(f"Incident:             {result.incident_title} (DB ID: {result.incident_id})")
    print(f"Media Asset ID:       {result.media_asset_id} ({result.media_type})")
    print(f"Frames Processed:     {result.frames_processed}")
    print(f"Detections Found:     {result.total_detections_found} (YOLO person detector)")
    print(f"Unique Person Tracks: {result.unique_person_tracks}")
    print(f"Locations Attached:   {result.locations_attached_count} (Explicitly 'simulated')")

    if result.scene_context:
        sc = result.scene_context
        print("\nRescueNet Environmental Disaster Context:")
        print(f"  * Flood Water Coverage:     {sc.water_coverage_pct:.2f}%")
        print(f"  * Debris Field Coverage:    {sc.debris_coverage_pct:.2f}%")
        print(f"  * Destroyed Buildings:      {sc.destroyed_building_pct:.2f}%")
        print(f"  * Major Structural Damage:  {sc.major_damage_building_pct:.2f}%")
        print(f"  * Road Coverage:            {sc.road_coverage_pct:.2f}%")
        print(f"  * Tree Coverage:            {sc.tree_coverage_pct:.2f}%")
        print(f"  * Inferred Hazard Severity: {sc.inferred_hazard_severity.value}")
        print(f"  * Inferred Accessibility:   {sc.inferred_accessibility}")
        print(f"  * Inferred Capabilities:    {', '.join(sc.inferred_required_capabilities)}")
        print(f"  * Contributing Factors:     {'; '.join(sc.contributing_hazard_factors)}")

    print(f"\nPriority Assessment:")
    print(f"  * Urgency Score:      {result.priority_assessment.urgency_score:.2f} [{result.priority_assessment.urgency_level}]")
    print(f"  * Uncertainty Score:  {result.priority_assessment.uncertainty_score:.2f} [{result.priority_assessment.uncertainty_level}]")
    print(f"  * Composite Priority: {result.priority_assessment.composite_priority.value}")
    print(f"  * Human Review Gate:  {result.priority_assessment.requires_human_verification} (Mandatory)")

    print(f"\nResource Recommendations ({result.recommendation_outcome.status}):")
    for idx, rec in enumerate(result.recommendation_outcome.recommendations, 1):
        print(f"  [{idx}] {rec.resource_name} (Suitability: {rec.suitability_score:.2f})")
        print(f"      Rationale: {rec.rationale}")
        print(f"      Distance:  {rec.distance_note}")

    print("\nProcessing Summary:")
    print(f"  {result.saved_records_summary}")

    # Generate 3-panel visual comparison if segmentation was performed
    if args.segmentation and segmentor is not None and args.save_vis:
        out_vis_path = Path(args.save_vis)
        out_vis_path.parent.mkdir(parents=True, exist_ok=True)

        from train_rescuenet_pilot import colorize_mask
        from PIL import Image

        raw_pil = Image.open(media_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        orig_np = np.array(raw_pil)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        # Draw detected YOLO boxes on the original image panel
        for track in workflow.tracker._active_tracks.values():
            bx = track.current_bbox
            x1, y1 = int(bx.x_min * 512), int(bx.y_min * 512)
            x2, y2 = int(bx.x_max * 512), int(bx.y_max * 512)
            cv2.rectangle(orig_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_bgr, f"Person {bx.confidence:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Predict segmentation mask
        pred_mask = segmentor.predict_mask(media_path)
        color_mask = colorize_mask(pred_mask)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Create info summary panel
        info_panel = np.full((512, 420, 3), (25, 25, 30), dtype=np.uint8)
        cv2.putText(info_panel, "RESCUENET CONTEXT SUMMARY", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
        cv2.line(info_panel, (15, 42), (405, 42), (100, 100, 100), 1)

        lines = [
            f"YOLO Person Detections: {result.total_detections_found}",
            f"Inferred Hazard: {result.scene_context.inferred_hazard_severity.value if result.scene_context else 'N/A'}",
            f"Terrain Accessibility: {result.scene_context.inferred_accessibility if result.scene_context else 'N/A'}",
            f"Urgency Score: {result.priority_assessment.urgency_score:.2f}",
            f"Composite Priority: {result.priority_assessment.composite_priority.value}",
            "",
            "Key Coverages (RescueNet 12-Class):",
            f"  Flood Water:  {result.scene_context.water_coverage_pct:.1f}%" if result.scene_context else "",
            f"  Debris Field: {result.scene_context.debris_coverage_pct:.1f}%" if result.scene_context else "",
            f"  Destroyed:    {result.scene_context.destroyed_building_pct:.1f}%" if result.scene_context else "",
            f"  Major Damage: {result.scene_context.major_damage_building_pct:.1f}%" if result.scene_context else "",
            f"  Roads:        {result.scene_context.road_coverage_pct:.1f}%" if result.scene_context else "",
            f"  Tree/Veg:     {result.scene_context.tree_coverage_pct:.1f}%" if result.scene_context else "",
            "",
            "Advisory Recommendation:",
            f"  {result.recommendation_outcome.recommendations[0].resource_name[:35]}" if result.recommendation_outcome.recommendations else "  No resources available",
            "  status: PENDING_REVIEW",
            "  auth required: True",
        ]
        y_pos = 70
        for line in lines:
            if line:
                cv2.putText(info_panel, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            y_pos += 22

        composite = np.hstack([orig_bgr, color_mask_bgr, info_panel])
        cv2.imwrite(str(out_vis_path), composite)
        print(f"\n[VISUALIZATION SAVED] 3-Panel Composite saved to: {out_vis_path}")

    print("=" * 70)
    print("NOTE ON CROSS-DATASET EVALUATION:")
    print("VisDrone (drone person detection) and RescueNet (disaster semantic segmentation)")
    print("were collected as separate datasets. Applying the segmentation model to a VisDrone")
    print("image represents a cross-dataset prototype demonstration of dual-layer inference")
    print("(micro person detection + macro environmental context), not geographically aligned ground truth.")
    print("=" * 70)
    print("[SUCCESS] End-to-end workflow successfully executed and persisted!")
    print("=" * 70)

    db.close()


if __name__ == "__main__":
    main()
