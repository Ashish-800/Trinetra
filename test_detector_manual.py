"""
Manual Testing Script for Phase 4 Detection Service.

Usage Examples:
1. Run with a real YOLO weights file on a VisDrone sample image:
   python test_detector_manual.py --weights weights/yolov8n.pt --image archive/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg

2. Run with custom confidence threshold and output path:
   python test_detector_manual.py --weights weights/yolov8n.pt --image path/to/aerial.jpg --conf 0.40 --output uploads/my_result.jpg
"""
import argparse
import sys
from pathlib import Path
from app.services.detector import (
    DetectorError,
    InvalidImageError,
    ModelWeightsNotFoundError,
    YOLOAerialDetector,
)


def find_sample_image(archive_dir: Path) -> Path | None:
    """Finds the first available image in the extracted VisDrone dataset, if present."""
    val_images = archive_dir / "VisDrone" / "VisDrone2019-DET-val" / "images"
    if val_images.exists():
        for p in val_images.glob("*.jpg"):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Manual Test for YOLO Aerial Detection Service")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolov8n.pt",
        help="Path to YOLO weights file (.pt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to aerial test image. If not supplied, automatically selects from archive/VisDrone/.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0.0 - 1.0, default: 0.35)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uploads/annotated_detection.jpg",
        help="Path where annotated output image with bounding boxes will be saved",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if args.image:
        image_path = Path(args.image)
    else:
        # Try finding a real aerial image from user's VisDrone dataset
        sample = find_sample_image(Path("archive"))
        if sample:
            image_path = sample
            print(f"[*] Auto-detected sample image from VisDrone: {image_path}")
        else:
            print("[!] No --image specified and no sample found in archive/VisDrone/.")
            print("    Please provide --image path/to/image.jpg")
            sys.exit(1)

    print("=" * 65)
    print("PHASE 4 DETECTION SERVICE: MANUAL TEST")
    print("=" * 65)
    print(f"Weights Path:         {weights_path}")
    print(f"Input Image:          {image_path}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"Output Annotated:     {args.output}")
    print("-" * 65)

    detector = YOLOAerialDetector(
        weights_path=weights_path,
        default_conf_threshold=args.conf,
    )

    try:
        result = detector.predict(
            image_input=image_path,
            conf_threshold=args.conf,
            save_annotated_path=args.output,
        )
    except ModelWeightsNotFoundError as e:
        print(f"[FAIL] Weights Error: {e}")
        print("\nNote: By project safety rules, model weights are NOT downloaded automatically.")
        print("To run this manual test with a real model, download or place 'yolov8n.pt' into the 'weights/' folder:")
        print("  mkdir weights")
        print("  curl -L -o weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
        sys.exit(1)
    except InvalidImageError as e:
        print(f"[FAIL] Image Error: {e}")
        sys.exit(1)
    except DetectorError as e:
        print(f"[FAIL] Detection Error: {e}")
        sys.exit(1)

    print(f"[SUCCESS] Model '{result.model_name}' executed successfully.")
    print(f"Image Resolution:   {result.image_width} x {result.image_height} px")
    print(f"Total Detections:   {result.total_detections}")
    print(f"Potential Persons:  {result.person_count}")
    print("\nDetected Objects:")
    for idx, obj in enumerate(result.objects, 1):
        b = obj.bbox
        print(f"  [{idx}] Class: '{obj.class_name}' | Conf: {obj.confidence:.2f} | BBox: [{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}]")

    if result.annotated_image_path:
        print(f"\nAnnotated image saved to: {result.annotated_image_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
