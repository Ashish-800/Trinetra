"""
RescueNet Semantic Segmentation Pilot Training Script.
Trains DeepLabV3-MobileNetV3-Large on the complete RescueNet train split for 5 epochs.
Utilizes mixed precision (FP16/AMP) on NVIDIA GeForce RTX 2050 (4 GB VRAM).
Tracks training loss, validation loss, pixel accuracy, mean IoU, and VRAM telemetry.
Saves checkpoints and generates visual segmentation evaluations.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import settings
from app.services.rescuenet_segmentor import (
    DEFAULT_TARGET_SIZE,
    NUM_RESCUENET_CLASSES,
    RescueNetDataset,
    create_segmentation_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 12-Class Visual Palette (RGB)
RESCUENET_PALETTE = {
    0: (0, 0, 0),          # Background: Black
    1: (165, 42, 42),      # Debris: Brown
    2: (0, 191, 255),      # Water: Sky Blue
    3: (0, 255, 0),        # Building No Damage: Green
    4: (255, 215, 0),      # Building Minor Damage: Gold
    5: (255, 140, 0),      # Building Major Damage: Orange
    6: (255, 0, 0),        # Building Total Destruction: Red
    7: (128, 0, 128),      # Vehicle: Purple
    8: (128, 128, 128),    # Road: Gray
    9: (34, 139, 34),      # Tree: Forest Green
    10: (0, 255, 255),     # Pool: Cyan
    11: (244, 164, 96),    # Sand: Sandy Brown
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D integer mask [H, W] (0..11) to an RGB color image [H, W, 3]."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in RESCUENET_PALETTE.items():
        color_img[mask == class_id] = color
    return color_img


class FastConfusionMatrix:
    """Efficient running confusion matrix for segmentation metrics across batches."""

    def __init__(self, num_classes: int = NUM_RESCUENET_CLASSES):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        valid = (targets >= 0) & (targets < self.num_classes)
        p = preds[valid].astype(np.int64)
        t = targets[valid].astype(np.int64)
        bins = np.bincount(self.num_classes * t + p, minlength=self.num_classes ** 2)
        self.matrix += bins.reshape((self.num_classes, self.num_classes))

    def compute_metrics(self) -> Tuple[float, float, Dict[int, float]]:
        total = self.matrix.sum()
        if total == 0:
            return 0.0, 0.0, {}

        diag = np.diag(self.matrix)
        pixel_acc = float(diag.sum() / total)

        union = self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - diag
        valid_classes = union > 0
        ious = np.zeros(self.num_classes, dtype=np.float64)
        ious[valid_classes] = diag[valid_classes] / (union[valid_classes] + 1e-7)

        per_class_iou = {c: float(ious[c]) for c in range(self.num_classes) if valid_classes[c]}
        mean_iou = float(np.mean([ious[c] for c in range(self.num_classes) if valid_classes[c]])) if per_class_iou else 0.0
        return pixel_acc, mean_iou, per_class_iou


def train_pilot(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    img_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    save_dir: str = "runs/rescuenet_pilot",
    resume_path: Optional[str] = None,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    eval_vis_dir = save_path / "eval_visualizations"
    eval_vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    logger.info("Initializing Pilot Training on device: %s", device)
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info("GPU: %s | Total VRAM: %.0f MB", gpu_name, vram_mb)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Dataset & Loaders
    logger.info("Loading RescueNet datasets (train and val splits)...")
    train_dataset = RescueNetDataset(split="train", img_size=img_size)
    val_dataset = RescueNetDataset(split="val", img_size=img_size)
    logger.info("Train dataset: %d pairs | Val dataset: %d pairs", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=is_cuda,
    )

    # 2. Model & Optimizer
    logger.info("Building DeepLabV3-MobileNetV3-Large model (12 classes)...")
    model = create_segmentation_model(num_classes=NUM_RESCUENET_CLASSES, pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    history = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "architecture": "DeepLabV3-MobileNetV3-Large",
        "num_classes": NUM_RESCUENET_CLASSES,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epoch_metrics": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "total_training_time_s": 0.0,
    }

    best_val_loss = float("inf")
    start_epoch = 1

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        history["best_epoch"] = ckpt.get("epoch", 0)
        history["best_val_loss"] = round(best_val_loss, 4)
        logger.info(
            "Resuming training from checkpoint %s at Epoch %d (Best Val Loss: %.4f)",
            resume_path,
            start_epoch,
            best_val_loss,
        )

        metrics_file = save_path / "metrics_history.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    history = json.load(f)
            except Exception:
                pass

    start_total_time = time.time()

    logger.info("Starting Pilot Training from Epoch %d to %d...", start_epoch, epochs)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_steps = 0

        logger.info("--- Epoch %d/%d Training ---", epoch, epochs)
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=is_cuda):
                outputs = model(images)["out"]
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            train_steps += 1

            if batch_idx % 300 == 0 or batch_idx == len(train_loader):
                avg_so_far = running_train_loss / train_steps
                logger.info(
                    "Epoch %d [%d/%d batches] - Step Loss: %.4f | Running Avg Loss: %.4f",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    avg_so_far,
                )

        train_loss = running_train_loss / max(1, train_steps)

        # 3. Validation Evaluation
        logger.info("Running validation evaluation for Epoch %d...", epoch)
        model.eval()
        running_val_loss = 0.0
        val_steps = 0
        cm = FastConfusionMatrix(num_classes=NUM_RESCUENET_CLASSES)

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)["out"]
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)["out"]
                    loss = criterion(outputs, targets)

                running_val_loss += loss.item()
                val_steps += 1

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()
                cm.update(preds, targets_np)

        val_loss = running_val_loss / max(1, val_steps)
        pixel_acc, mean_iou, per_class_iou = cm.compute_metrics()
        epoch_time = time.time() - epoch_start

        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if is_cuda else 0.0

        logger.info(
            "Epoch %d Summary: Train Loss: %.4f | Val Loss: %.4f | Pixel Acc: %.4f | mIoU: %.4f | Time: %.1fs | Peak VRAM: %.1f MB",
            epoch,
            train_loss,
            val_loss,
            pixel_acc,
            mean_iou,
            epoch_time,
            peak_vram_mb,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "pixel_accuracy": round(pixel_acc, 4),
            "mean_iou": round(mean_iou, 4),
            "epoch_time_s": round(epoch_time, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "per_class_iou": {str(k): round(v, 4) for k, v in per_class_iou.items()},
        }
        history["epoch_metrics"].append(epoch_record)

        # Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            best_ckpt_path = save_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "pixel_acc": pixel_acc,
                    "mean_iou": mean_iou,
                },
                best_ckpt_path,
            )
            logger.info("Saved new BEST checkpoint to: %s (Val Loss: %.4f)", best_ckpt_path, val_loss)

    # 4. Save Final Checkpoint
    final_ckpt_path = save_path / "final_model.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "pixel_acc": pixel_acc,
            "mean_iou": mean_iou,
        },
        final_ckpt_path,
    )
    logger.info("Saved FINAL checkpoint to: %s", final_ckpt_path)

    total_time_s = time.time() - start_total_time
    history["total_training_time_s"] = round(total_time_s, 1)

    metrics_path = save_path / "metrics_history.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved metrics history to: %s", metrics_path)

    # 5. Visual Segmentation Evaluation on Selected Validation Images
    logger.info("Generating visual segmentation evaluation on validation images...")
    eval_stems = ["10781", "10793", "10842"]
    vis_summary = []

    model.eval()
    for stem in eval_stems:
        img_p = Path("RescueNet/val/val-org-img") / f"{stem}.jpg"
        lab_p = Path("RescueNet/val/val-label-img") / f"{stem}_lab.png"
        if not img_p.exists() or not lab_p.exists():
            continue

        raw_img = Image.open(img_p).convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
        raw_lab = Image.open(lab_p).resize(img_size, Image.Resampling.NEAREST)
        lab_arr = np.array(raw_lab, dtype=np.int64)

        # Predict
        input_tensor = (
            torch.from_numpy(np.array(raw_img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            .unsqueeze(0)
            .to(device)
        )
        # Normalize
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean_t) / std_t

        with torch.no_grad():
            if is_cuda:
                with torch.amp.autocast("cuda"):
                    pred_logits = model(input_tensor)["out"]
            else:
                pred_logits = model(input_tensor)["out"]
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Verification of class ID bounds
        unique_pred_ids = sorted(list(set(pred_mask.flatten().tolist())))
        assert min(unique_pred_ids) >= 0 and max(unique_pred_ids) <= 11, "Prediction contains invalid class IDs!"

        # Colorize
        gt_color = colorize_mask(lab_arr)
        pred_color = colorize_mask(pred_mask)
        orig_np = np.array(raw_img)

        # Combine into side-by-side composite: [Original | Ground Truth | Prediction]
        composite = np.hstack([orig_np, gt_color, pred_color])
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

        out_vis_file = eval_vis_dir / f"eval_{stem}_comparison.png"
        cv2.imwrite(str(out_vis_file), composite_bgr)

        # Save individual prediction mask as well
        pred_mask_file = eval_vis_dir / f"pred_{stem}_color.png"
        cv2.imwrite(str(pred_mask_file), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

        logger.info("Saved visual evaluation for %s to: %s (Classes: %s)", stem, out_vis_file, unique_pred_ids)
        vis_summary.append({"image_id": stem, "composite_path": str(out_vis_file), "predicted_classes": unique_pred_ids})

    logger.info("Pilot training and visual evaluation successfully completed in %.1f minutes.", total_time_s / 60.0)
    return history, vis_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RescueNet Pilot Semantic Segmentation Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--save-dir", type=str, default="runs/rescuenet_pilot", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train_pilot(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        resume_path=args.resume,
    )
