"""
Quantitative Validation Evaluation of Trained RescueNet Segmentation Model.
Evaluates DeepLabV3-MobileNetV3-Large on the real RescueNet validation split (449 pairs).
Calculates mIoU, per-class IoU, pixel accuracy, class distributions, confusion matrix,
and exports side-by-side visual comparisons (Original | Ground Truth | Prediction).
"""
import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from app.schemas.rescuenet import CLASS_DISPLAY_NAMES, RescueNetClass
from app.services.rescuenet_segmentor import (
    DEFAULT_TARGET_SIZE,
    NUM_RESCUENET_CLASSES,
    RescueNetDataset,
    create_segmentation_model,
)
from train_rescuenet_pilot import FastConfusionMatrix, RESCUENET_PALETTE, colorize_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_rescuenet")


def evaluate_validation_split(
    checkpoint_path: str = "runs/rescuenet_pilot/best_model.pt",
    output_dir: str = "runs/rescuenet_pilot/validation_evaluation",
    batch_size: int = 4,
    img_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    num_visualizations: int = 12,
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # 1. Load validation dataset
    logger.info("Loading RescueNet validation split...")
    val_dataset = RescueNetDataset(split="val", img_size=img_size)
    num_samples = len(val_dataset)
    logger.info("Found %d real validation image/mask pairs.", num_samples)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Build model & load checkpoint
    logger.info("Loading checkpoint from: %s", ckpt_p)
    model = create_segmentation_model(num_classes=NUM_RESCUENET_CLASSES, pretrained_backbone=False)
    ckpt = torch.load(ckpt_p, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_info = ckpt.get("epoch", "unknown")
        val_loss_ckpt = ckpt.get("val_loss", "unknown")
        logger.info("Loaded state dict (Trained Epoch: %s, Checkpoint Val Loss: %s)", epoch_info, val_loss_ckpt)
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    # 3. Quantitative Evaluation Loop
    cm = FastConfusionMatrix(num_classes=NUM_RESCUENET_CLASSES)
    gt_pixel_counts = np.zeros(NUM_RESCUENET_CLASSES, dtype=np.int64)
    pred_pixel_counts = np.zeros(NUM_RESCUENET_CLASSES, dtype=np.int64)

    start_eval_time = time.time()
    logger.info("Starting quantitative inference over %d validation samples...", num_samples)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)["out"]
            else:
                outputs = model(images)["out"]

            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int64)
            targets_np = targets.cpu().numpy().astype(np.int64)

            cm.update(preds, targets_np)

            # Accumulate pixel counts
            for c in range(NUM_RESCUENET_CLASSES):
                gt_pixel_counts[c] += np.sum(targets_np == c)
                pred_pixel_counts[c] += np.sum(preds == c)

            if batch_idx % 25 == 0 or batch_idx == len(val_loader):
                logger.info("Processed [%d/%d] batches", batch_idx, len(val_loader))

    eval_duration = time.time() - start_eval_time
    pixel_acc, mean_iou, per_class_iou = cm.compute_metrics()

    total_gt_pixels = int(gt_pixel_counts.sum())
    total_pred_pixels = int(pred_pixel_counts.sum())

    # Build detailed per-class metrics
    class_metrics = {}
    for c in range(NUM_RESCUENET_CLASSES):
        cls_enum = RescueNetClass(c)
        name = CLASS_DISPLAY_NAMES[cls_enum]
        tp = int(cm.matrix[c, c])
        fn = int(cm.matrix[c, :].sum() - tp)
        fp = int(cm.matrix[:, c].sum() - tp)
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        gt_freq_pct = float(gt_pixel_counts[c] / total_gt_pixels * 100.0) if total_gt_pixels > 0 else 0.0
        pred_freq_pct = float(pred_pixel_counts[c] / total_pred_pixels * 100.0) if total_pred_pixels > 0 else 0.0

        class_metrics[name] = {
            "class_id": c,
            "iou": round(iou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "gt_pixel_count": int(gt_pixel_counts[c]),
            "gt_frequency_pct": round(gt_freq_pct, 2),
            "pred_pixel_count": int(pred_pixel_counts[c]),
            "pred_frequency_pct": round(pred_freq_pct, 2),
        }

    # Identify best and worst performing classes (with gt_pixel_count > 0)
    evaluated_classes = [c for c, m in class_metrics.items() if m["gt_pixel_count"] > 0]
    sorted_by_iou = sorted(evaluated_classes, key=lambda c: class_metrics[c]["iou"], reverse=True)
    best_classes = sorted_by_iou[:3]
    worst_classes = sorted_by_iou[-3:]

    # Save Confusion Matrix
    cm_npy_path = out_dir / "confusion_matrix.npy"
    np.save(str(cm_npy_path), cm.matrix)
    logger.info("Saved raw confusion matrix array to: %s", cm_npy_path)

    # 4. Generate Visual Comparisons on Diverse Real Validation Samples
    logger.info("Generating side-by-side visual comparisons for real validation samples...")
    # Choose sample indices spaced throughout the validation set
    sample_indices = np.linspace(0, num_samples - 1, num=max(10, num_visualizations), dtype=int).tolist()
    # Add key specific interesting validation indices if available
    vis_records = []

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for idx in sample_indices:
        img_p, mask_p = val_dataset.samples[idx]
        stem = img_p.stem

        raw_img = Image.open(img_p).convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
        raw_mask = Image.open(mask_p).convert("L").resize(img_size, Image.Resampling.NEAREST)

        mask_np = np.array(raw_mask, dtype=np.int64)

        # Preprocess
        img_arr = np.array(raw_img, dtype=np.float32) / 255.0
        inp_tensor = torch.from_numpy(img_arr.transpose((2, 0, 1))).unsqueeze(0).to(device)
        inp_tensor = (inp_tensor - mean_t) / std_t

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred_logits = model(inp_tensor)["out"]
            else:
                pred_logits = model(inp_tensor)["out"]
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        orig_bgr = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        gt_color = cv2.cvtColor(colorize_mask(mask_np), cv2.COLOR_RGB2BGR)
        pred_color = cv2.cvtColor(colorize_mask(pred_mask), cv2.COLOR_RGB2BGR)

        # Add top banner text to each panel
        def add_panel_banner(img: np.ndarray, text: str, color=(0, 255, 0)) -> np.ndarray:
            banner = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            return np.vstack([banner, img])

        panel_orig = add_panel_banner(orig_bgr, f"Original Image: {stem}.jpg", (255, 255, 255))
        panel_gt = add_panel_banner(gt_color, "Ground Truth Mask", (0, 255, 255))
        panel_pred = add_panel_banner(pred_color, "DeepLabV3 Predicted Mask", (0, 255, 0))

        composite = np.hstack([panel_orig, panel_gt, panel_pred])
        out_vis_path = out_dir / f"val_sample_{stem}_comparison.png"
        cv2.imwrite(str(out_vis_path), composite)

        present_gt_classes = [CLASS_DISPLAY_NAMES[RescueNetClass(c)] for c in np.unique(mask_np) if c in CLASS_DISPLAY_NAMES]
        present_pred_classes = [CLASS_DISPLAY_NAMES[RescueNetClass(c)] for c in np.unique(pred_mask) if c in CLASS_DISPLAY_NAMES]

        vis_records.append({
            "image_id": stem,
            "file": str(out_vis_path),
            "gt_classes": present_gt_classes,
            "pred_classes": present_pred_classes,
        })

    logger.info("Saved %d validation visual comparisons in: %s", len(vis_records), out_dir)

    # 5. Summarize Results & Save JSON Report
    results = {
        "checkpoint": str(ckpt_p),
        "device": str(device),
        "validation_samples": num_samples,
        "input_resolution": list(img_size),
        "evaluation_time_seconds": round(eval_duration, 2),
        "overall_metrics": {
            "mean_iou": round(mean_iou, 4),
            "pixel_accuracy": round(pixel_acc, 4),
            "total_pixels_evaluated": total_gt_pixels,
        },
        "per_class_metrics": class_metrics,
        "best_performing_classes": [{"class": c, "iou": class_metrics[c]["iou"]} for c in best_classes],
        "worst_performing_classes": [{"class": c, "iou": class_metrics[c]["iou"]} for c in worst_classes],
        "confusion_matrix_path": str(cm_npy_path),
        "visual_comparisons_dir": str(out_dir),
        "visual_samples_count": len(vis_records),
    }

    report_json_path = out_dir / "validation_evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved complete JSON evaluation report to: %s", report_json_path)

    return results


def print_evaluation_summary(results: Dict):
    metrics = results["overall_metrics"]
    class_metrics = results["per_class_metrics"]

    print("\n" + "=" * 80)
    print("RESCUENET VALIDATION EVALUATION REPORT (FULL 449 VALIDATION SAMPLES)")
    print("=" * 80)
    print(f"Checkpoint Evaluated: {results['checkpoint']}")
    print(f"Inference Device:     {results['device']}")
    print(f"Validation Pairs:     {results['validation_samples']}")
    print(f"Input Resolution:     {results['input_resolution'][0]}x{results['input_resolution'][1]}")
    print(f"Evaluation Time:      {results['evaluation_time_seconds']}s")
    print("-" * 80)
    print(f"OVERALL MEAN IoU (mIoU): {metrics['mean_iou']:.4f} ({metrics['mean_iou'] * 100:.2f}%)")
    print(f"OVERALL PIXEL ACCURACY:  {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy'] * 100:.2f}%)")
    print("-" * 80)
    print(f"{'Class Name':<30} | {'IoU':<8} | {'Precision':<10} | {'Recall':<8} | {'GT Freq %':<10} | {'Pred Freq %':<10}")
    print("-" * 80)
    for name, m in class_metrics.items():
        print(f"{name:<30} | {m['iou']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['gt_frequency_pct']:<10.2f} | {m['pred_frequency_pct']:<10.2f}")
    print("-" * 80)
    print("BEST PERFORMING CLASSES:")
    for item in results["best_performing_classes"]:
        print(f"  * {item['class']}: IoU = {item['iou']:.4f}")
    print("WORST PERFORMING CLASSES:")
    for item in results["worst_performing_classes"]:
        print(f"  * {item['class']}: IoU = {item['iou']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained RescueNet Segmentation Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="runs/rescuenet_pilot/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="runs/rescuenet_pilot/validation_evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-vis", type=int, default=12)
    args = parser.parse_args()

    res = evaluate_validation_split(
        checkpoint_path=args.checkpoint,
        output_dir=args.out_dir,
        batch_size=args.batch_size,
        num_visualizations=args.num_vis,
    )
    print_evaluation_summary(res)
