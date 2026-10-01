"""
Demonstration and Manual Testing Script for Phase 5 Video Inference & Tracking.

Usage:
  # Run with automatic synthetic video (zero external files required)
  python test_video_tracking_demo.py

  # Run with a real UAV video and a YOLO model weights file
  python test_video_tracking_demo.py --video path/to/drone_flight.mp4 --weights weights/yolov8n.pt --interval 1.0
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from app.services.detector import MockDetector, YOLOAerialDetector
from app.services.tracker import AerialIoUTracker
from app.services.video_tracking import VideoTrackingPipeline


def create_demo_tracking_video(output_path: Path) -> Path:
    """
    Creates a synthetic 4-second video (40 frames at 10 fps) simulating a UAV flight
    over terrain with 2 moving individuals and 1 stationary object.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 640, 480
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_idx in range(40):
        # Background
        frame = np.full((height, width, 3), (120, 140, 110), dtype=np.uint8)

        # Person 1 moving left to right
        p1_x = int(100 + frame_idx * 6)
        p1_y = 200
        cv2.rectangle(frame, (p1_x, p1_y), (p1_x + 30, p1_y + 60), (50, 50, 200), -1)

        # Person 2 moving diagonally
        p2_x = int(450 - frame_idx * 4)
        p2_y = int(150 + frame_idx * 3)
        cv2.rectangle(frame, (p2_x, p2_y), (p2_x + 30, p2_y + 60), (200, 50, 50), -1)

        cv2.putText(
            frame,
            f"Frame {frame_idx:03d} (T+{frame_idx/fps:.1f}s) - Synthetic Flight",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        out.write(frame)

    out.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test Video Inference and Multi-Object Tracking")
    parser.add_argument("--video", type=str, default=None, help="Path to UAV video file")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLO weights (.pt). If omitted, uses MockDetector.")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds (default: 0.5s)")
    parser.add_argument("--output-dir", type=str, default="uploads/tracking_demo_runs", help="Output directory")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        print("[*] No --video provided. Generating synthetic 4-second UAV video...")
        video_path = create_demo_tracking_video(Path("uploads/demo_samples/synthetic_tracking_flight.mp4"))

    # Select Detector
    if args.weights and Path(args.weights).exists():
        print(f"[*] Initializing YOLOAerialDetector with weights: {args.weights}")
        detector = YOLOAerialDetector(weights_path=args.weights, default_conf_threshold=0.30)
    else:
        print("[*] Using deterministic MockDetector (no GPU/weights required).")
        detector = MockDetector()

    tracker = AerialIoUTracker(iou_threshold=0.20, max_lost_frames=4)
    pipeline = VideoTrackingPipeline(
        detector=detector,
        tracker=tracker,
        default_output_dir=Path(args.output_dir),
    )

    print("=" * 70)
    print("PHASE 5: VIDEO INFERENCE & TRACKING DEMO")
    print("=" * 70)
    print(f"Video Source:      {video_path}")
    print(f"Sampling Interval: {args.interval} seconds per frame")
    print("-" * 70)

    summary = pipeline.process_video(
        video_path=video_path,
        sampling_interval_seconds=args.interval,
        save_annotated_frames=True,
    )

    print("\n[EXECUTION SUMMARY]")
    print(f"Duration:                {summary.duration_seconds:.1f} seconds")
    print(f"Sampled Frames:          {summary.total_frames_extracted}")
    print(f"Unique Person Tracks:    {summary.unique_person_tracks} (Avoids duplicate count inflation!)")
    print(f"Total Unique Tracks:     {summary.total_unique_tracks}")
    print("\nSampled Frame Breakdown:")
    for fr in summary.frames:
        track_ids = [f"ID:{d.track_id} ({d.class_name})" for d in fr.detections]
        print(f"  -> Frame {fr.frame_index:04d} (T+{fr.timestamp_seconds:.2f}s): {len(fr.detections)} detections -> {', '.join(track_ids) if track_ids else 'None'}")

    print("\nAnnotated frames saved under:")
    print(f"  {Path(args.output_dir) / video_path.stem / 'annotated'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
