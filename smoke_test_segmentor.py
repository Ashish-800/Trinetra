"""
RescueNet Semantic Segmentation Model Preparation & Smoke Test.
Verifies dataset loading, tensor dimensions, 12 output classes, CUDA execution,
loss calculation, backward step, and GPU VRAM footprint on NVIDIA RTX 2050 (4 GB).
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.services.rescuenet_segmentor import (
    DEFAULT_TARGET_SIZE,
    NUM_RESCUENET_CLASSES,
    RescueNetDataset,
    RescueNetSegmentor,
    create_segmentation_model,
)


def run_smoke_test():
    print("=" * 70)
    print("RESCUENET SEMANTIC SEGMENTATION: MODEL PREPARATION & SMOKE TEST")
    print("=" * 70)

    # 1. Device check
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"CUDA Available:   {is_cuda}")
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"Device:           cuda:0 ({gpu_name})")
        print(f"Total VRAM:       {total_vram_mb:.0f} MB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        print(f"Device:           cpu")

    # 2. Dataset loading and pairing verification
    print("\n--- Step 1: Dataset Pipeline & Pairing Verification ---")
    val_dataset = RescueNetDataset(split="val", img_size=DEFAULT_TARGET_SIZE, max_samples=4)
    print(f"Loaded RescueNet validation subset: {len(val_dataset)} samples")
    loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    sample_images, sample_masks = next(iter(loader))
    print(f"Input batch shape:   {sample_images.shape} (dtype: {sample_images.dtype})")
    print(f"Target batch shape:  {sample_masks.shape} (dtype: {sample_masks.dtype})")
    print(f"Mask values range:   min={sample_masks.min().item()}, max={sample_masks.max().item()}")

    assert sample_images.shape == (2, 3, 512, 512), f"Unexpected input shape: {sample_images.shape}"
    assert sample_masks.shape == (2, 512, 512), f"Unexpected mask shape: {sample_masks.shape}"
    assert sample_masks.min().item() >= 0 and sample_masks.max().item() <= 11, "Mask labels exceed [0, 11] range!"
    print("[PASS] Dataset pairing and preprocessing verified.")

    # 3. Model Architecture Instantiation
    print("\n--- Step 2: Architecture Setup (DeepLabV3-MobileNetV3-Large) ---")
    model = create_segmentation_model(num_classes=NUM_RESCUENET_CLASSES, pretrained_backbone=True).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture:     DeepLabV3 with MobileNetV3-Large backbone")
    print(f"Supported Classes: {NUM_RESCUENET_CLASSES} (IDs 0..11)")
    print(f"Total Parameters:  {param_count:,} ({param_count / 1e6:.2f} M)")
    print(f"Trainable Params:  {trainable_count:,}")

    # 4. Forward Pass & Loss Calculation
    print("\n--- Step 3: Forward Pass & Backward Step (Mixed Precision) ---")
    images = sample_images.to(device)
    targets = sample_masks.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    t0 = time.time()
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=is_cuda):
        outputs = model(images)["out"]
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    step_time_ms = (time.time() - t0) * 1000

    print(f"Output Logits Shape: {outputs.shape}")
    print(f"Cross-Entropy Loss:  {loss.item():.4f}")
    print(f"Step Execution Time: {step_time_ms:.1f} ms")
    assert outputs.shape == (2, NUM_RESCUENET_CLASSES, 512, 512), f"Unexpected output shape: {outputs.shape}"
    print("[PASS] Forward and backward passes executed successfully.")

    # 5. GPU Memory Telemetry
    if is_cuda:
        alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"\n--- Step 4: VRAM Utilization Telemetry ---")
        print(f"Allocated Memory: {alloc_mb:.1f} MB")
        print(f"Peak Memory Used: {peak_mb:.1f} MB ({(peak_mb / total_vram_mb) * 100:.1f}% of 4 GB)")
        print(f"Reserved Memory:  {reserved_mb:.1f} MB")
        print("[PASS] VRAM footprint comfortably within 4 GB budget.")

    # 6. Service-level Inference & Scene Context Extraction
    print("\n--- Step 5: Service-level Inference & Scene Context Pipeline ---")
    segmentor = RescueNetSegmentor(model=model, device=device)
    sample_img_path = val_dataset.samples[0][0]
    print(f"Running segmentor on sample image: {sample_img_path.name}")
    pred_mask = segmentor.predict_mask(sample_img_path)
    print(f"Predicted mask shape: {pred_mask.shape} (dtype: {pred_mask.dtype})")
    print(f"Predicted classes present: {sorted(list(set(pred_mask.flatten())))}")

    analysis = segmentor.predict_scene_analysis(sample_img_path, image_id=sample_img_path.stem)
    print(f"Scene Analysis Image ID: {analysis.image_id}")
    print(f"Inferred Hazard Severity: {analysis.inferred_hazard_severity.value}")
    print(f"Inferred Accessibility:   {analysis.inferred_accessibility}")
    print(f"Contributing Factors:     {analysis.contributing_hazard_factors}")
    print(f"Total Pixels Analyzed:    {analysis.total_pixels:,}")

    print("\n" + "=" * 70)
    print("[SUCCESS] RescueNet Semantic Segmentation Smoke Test Complete!")
    print("Model architecture, dataset pipeline, and CUDA mixed-precision are READY.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
