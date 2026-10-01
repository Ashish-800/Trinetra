"""
Demonstration and Manual Testing Script for Phase 3 Media Ingestion.

Usage:
  python test_ingestion_demo.py
  python test_ingestion_demo.py --image path/to/aerial.jpg --video path/to/drone_flight.mp4
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from app.services.ingestion import MediaIngestionService


def create_demo_assets(target_dir: Path) -> tuple[Path, Path]:
    """Creates a synthetic demonstration image and short video if none are supplied."""
    target_dir.mkdir(parents=True, exist_ok=True)
    img_path = target_dir / "synthetic_aerial.jpg"
    video_path = target_dir / "synthetic_drone_sweep.mp4"

    # 1. Generate synthetic image (800x600 with simulated terrain grid)
    img = np.full((600, 800, 3), (120, 150, 110), dtype=np.uint8)
    # Draw simulated ground features
    cv2.putText(img, "UAV Aerial Survey [Simulated Demo]", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.rectangle(img, (200, 200), (350, 400), (80, 80, 80), -1)  # Simulated structure
    cv2.circle(img, (500, 300), 20, (50, 50, 200), -1)  # Simulated marker
    cv2.imwrite(str(img_path), img)

    # 2. Generate synthetic 3-second video (30 frames at 10 fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 640, 480
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(30):
        frame = np.full((height, width, 3), (100 + i * 3, 130, 110), dtype=np.uint8)
        cv2.putText(frame, f"Flight Telemetry Frame {i} - T+{i/fps:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    return img_path, video_path


def main():
    parser = argparse.ArgumentParser(description="Test Media Ingestion Service with Image and Video")
    parser.add_argument("--image", type=str, default=None, help="Path to aerial image file")
    parser.add_argument("--video", type=str, default=None, help="Path to UAV video file")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds for video (default: 1.0)")
    args = parser.parse_args()

    demo_dir = Path("uploads/demo_samples")
    if args.image and args.video:
        img_path = Path(args.image)
        video_path = Path(args.video)
    else:
        print("[*] No custom paths provided. Generating synthetic demonstration assets...")
        img_path, video_path = create_demo_assets(demo_dir)

    service = MediaIngestionService(default_output_dir=Path("uploads/extracted_frames"))

    print("\n" + "=" * 65)
    print("1. TESTING IMAGE INGESTION")
    print("=" * 65)
    print(f"Target Image: {img_path}")
    img_result = service.ingest(img_path)
    print(f"Media Type:   {img_result.media_type}")
    print(f"Dimensions:   {img_result.width_px} x {img_result.height_px} px")
    print(f"File Size:    {img_result.file_size_bytes} bytes")
    print(f"Frames:       {img_result.total_frames_extracted} extracted")
    for f in img_result.frames:
        print(f"  -> Frame {f.frame_index} (T+{f.timestamp_seconds:.2f}s): {f.output_path}")

    print("\n" + "=" * 65)
    print("2. TESTING VIDEO INGESTION (STREAMING FRAME EXTRACTION)")
    print("=" * 65)
    print(f"Target Video: {video_path}")
    print(f"Sampling:     1 frame every {args.interval} seconds")
    vid_result = service.ingest(video_path, interval_seconds=args.interval)
    print(f"Media Type:   {vid_result.media_type}")
    print(f"Dimensions:   {vid_result.width_px} x {vid_result.height_px} px")
    print(f"Duration:     {vid_result.duration_seconds} seconds")
    print(f"Extracted:    {vid_result.total_frames_extracted} frames")
    for f in vid_result.frames:
        print(f"  -> Frame {f.frame_index:04d} (T+{f.timestamp_seconds:.2f}s): {f.output_path}")

    print("\n" + "=" * 65)
    print("[SUCCESS] Media Ingestion tested successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
