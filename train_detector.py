"""
Phase 14: VisDrone UAV Object Detector Training & Evaluation Script.
Configured defensively for CPU execution without machine lockup.
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("PHASE 14 — VISDRONE UAV DETECTOR TRAINING")
    print("=" * 70)
    
    # 1. Environment Confirmation
    print("\n--- STEP 1: Environment Confirmation ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    device = "cuda:0" if cuda_avail else "cpu"
    print(f"Selected Compute Device: {device}")
    
    # 2. Paths
    root_dir = Path(__file__).resolve().parent
    weights_path = root_dir / "weights" / "yolov8n.pt"
    data_yaml = root_dir / "visdrone.yaml"
    output_project = root_dir / "runs" / "detect"
    run_name = "visdrone_yolov8n_v1"
    
    if not weights_path.exists():
        print(f"Error: Base weights missing at {weights_path}")
        sys.exit(1)
        
    print(f"Base Weights: {weights_path}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Target Output: {output_project / run_name}")
    
    # 3. Model Training
    print("\n--- STEP 4: Commencing Training Run ---")
    model = YOLO(str(weights_path))
    
    start_train_time = time.time()
    
    # Training configuration adhering to CPU limitations:
    # Target 50 epochs with early-stopping patience=5
    # Batch size 16, imgsz 640
    # fraction=0.03 (~194 train images with ~8,700 annotations) ensures completion on CPU
    train_results = model.train(
        data=str(data_yaml),
        epochs=50,
        patience=5,
        imgsz=640,
        batch=16,
        device=device,
        project=str(output_project),
        name=run_name,
        exist_ok=True,
        fraction=0.03,
        workers=2,
        plots=True,
        save=True,
        deterministic=True,
        seed=42,
        verbose=True,
    )
    
    total_train_duration = time.time() - start_train_time
    print(f"\nTraining completed in {total_train_duration:.2f}s ({total_train_duration/60:.2f} min)")
    
    # 4. Checkpoints
    best_ckpt = output_project / run_name / "weights" / "best.pt"
    last_ckpt = output_project / run_name / "weights" / "last.pt"
    print(f"\nBest Checkpoint: {best_ckpt} (Exists: {best_ckpt.exists()})")
    print(f"Final Checkpoint: {last_ckpt} (Exists: {last_ckpt.exists()})")
    
    # 5. Model Evaluation (STEP 5)
    print("\n--- STEP 5: Model Evaluation on Validation Set ---")
    val_model = YOLO(str(best_ckpt))
    
    t_val_start = time.time()
    val_metrics = val_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        device=device,
        plots=True,
        verbose=True,
    )
    val_duration = time.time() - t_val_start
    print(f"Validation completed in {val_duration:.2f}s")
    
    # Extract Per-Class Metrics
    print("\n================ DETAILED EVALUATION METRICS ================")
    # Overall
    box_p = float(val_metrics.box.p.mean()) if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > 0 else float(val_metrics.box.mp)
    box_r = float(val_metrics.box.r.mean()) if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > 0 else float(val_metrics.box.mr)
    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    
    print(f"Overall Metrics: Precision={box_p:.4f} | Recall={box_r:.4f} | mAP50={map50:.4f} | mAP50-95={map50_95:.4f}")
    
    class_names = ["person", "vehicle", "other"]
    print("\nPer-Class Breakdown:")
    for cls_idx, cls_name in enumerate(class_names):
        try:
            p_cls = float(val_metrics.box.p[cls_idx])
            r_cls = float(val_metrics.box.r[cls_idx])
            map50_cls = float(val_metrics.box.all_ap[cls_idx, 0])
            map50_95_cls = float(val_metrics.box.all_ap[cls_idx].mean())
            print(f"  Class {cls_idx} ({cls_name:7s}): Precision={p_cls:.4f} | Recall={r_cls:.4f} | mAP50={map50_cls:.4f} | mAP50-95={map50_95_cls:.4f}")
        except Exception as e:
            print(f"  Class {cls_idx} ({cls_name:7s}): Could not extract individual metrics ({e})")
            
    # 6. Qualitative Inference (STEP 6)
    print("\n--- STEP 6: Qualitative Inference on 5 Validation Images ---")
    val_images_dir = root_dir / "archive" / "VisDrone" / "VisDrone2019-DET-val" / "images"
    test_images = sorted(list(val_images_dir.glob("*.jpg")))[:5]
    
    qual_dir = output_project / run_name / "qualitative_predictions"
    qual_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img_p in enumerate(test_images, 1):
        t0_inf = time.time()
        inf_res = val_model(str(img_p), conf=0.25, verbose=False)[0]
        inf_ms = (time.time() - t0_inf) * 1000
        
        boxes = inf_res.boxes
        num_dets = len(boxes)
        classes_found = [class_names[int(c)] if int(c) < len(class_names) else str(int(c)) for c in boxes.cls.tolist()]
        confs = [round(float(c), 3) for c in boxes.conf.tolist()]
        conf_range = f"{min(confs):.3f} - {max(confs):.3f}" if confs else "N/A"
        
        # Save annotated image
        annotated = inf_res.plot()
        out_save_path = qual_dir / f"pred_{img_p.name}"
        cv2.imwrite(str(out_save_path), annotated)
        
        # Count classes
        class_summary = {}
        for c in classes_found:
            class_summary[c] = class_summary.get(c, 0) + 1
            
        print(f"[{idx}] {img_p.name} | Total Detections: {num_dets} | Classes: {class_summary} | Conf Range: {conf_range} | Latency: {inf_ms:.1f}ms")
        print(f"     Saved annotated image to: {out_save_path}")

    print("\n" + "=" * 70)
    print("PHASE 14 COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
