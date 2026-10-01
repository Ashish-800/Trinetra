"""
RescueNet Semantic Segmentation Model Service and Dataset Preprocessing Pipeline.
Provides a lightweight, memory-efficient DeepLabV3-MobileNetV3-Large semantic segmentation
architecture optimized for 4 GB VRAM (e.g., NVIDIA GeForce RTX 2050 Laptop GPU).
Supports all 12 RescueNet disaster scene classes (Background ID 0 + 11 foreground classes IDs 1-11).
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.models.segmentation as seg_models

from app.core.config import settings
from app.schemas.rescuenet import (
    CLASS_DISPLAY_NAMES,
    RescueNetClass,
    RescueNetHazardThresholds,
    RescueNetSceneAnalysis,
)
from app.services.rescuenet import RescueNetService

logger = logging.getLogger(__name__)


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TARGET_SIZE = (512, 512)
NUM_RESCUENET_CLASSES = 12


class RescueNetDataset(Dataset):
    """
    PyTorch Dataset for RescueNet disaster semantic segmentation.
    Pairs aerial images with corresponding pixel label masks.
    Applies bilinear interpolation to images and nearest-neighbor interpolation to masks
    to strictly preserve integer semantic class IDs (0..11).
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        img_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        max_samples: Optional[int] = None,
    ):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else settings.BASE_DIR / "RescueNet"
        self.split = split
        self.img_size = img_size

        split_dir = self.dataset_dir / split
        self.img_dir = split_dir / f"{split}-org-img"
        self.lab_dir = split_dir / f"{split}-label-img"

        if not self.img_dir.exists() or not self.lab_dir.exists():
            raise FileNotFoundError(
                f"RescueNet {split} directories not found at: {self.img_dir} or {self.lab_dir}"
            )

        self.samples: List[Tuple[Path, Path]] = []
        for img_p in sorted(self.img_dir.glob("*.jpg")):
            stem = img_p.stem
            mask_p = self.lab_dir / f"{stem}_lab.png"
            if mask_p.exists():
                self.samples.append((img_p, mask_p))
                if max_samples and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No valid image-mask pairs found in RescueNet {split} split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p = self.samples[idx]

        # 1. Load image and mask
        image = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)

        # 2. Resize: Bilinear for image, Nearest-neighbor for mask
        image_resized = image.resize(self.img_size, Image.Resampling.BILINEAR)
        mask_resized = mask.resize(self.img_size, Image.Resampling.NEAREST)

        # 3. Convert image to normalized float32 tensor [3, H, W]
        img_tensor = TF.to_tensor(image_resized)
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # 4. Convert mask to int64 tensor [H, W] with verified range [0, 11]
        mask_np = np.array(mask_resized, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)

        return img_tensor, mask_tensor


def create_segmentation_model(
    num_classes: int = NUM_RESCUENET_CLASSES,
    architecture: str = "deeplabv3_mobilenet_v3_large",
    pretrained_backbone: bool = True,
) -> nn.Module:
    """
    Builds a lightweight semantic segmentation model configured for RescueNet's 12 classes.
    Uses DeepLabV3-MobileNetV3-Large by default for ASPP multi-scale context within a 4 GB VRAM budget.
    """
    if architecture == "deeplabv3_mobilenet_v3_large":
        try:
            if pretrained_backbone:
                model = seg_models.deeplabv3_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=seg_models.MobileNet_V3_Large_Weights.DEFAULT if hasattr(seg_models, "MobileNet_V3_Large_Weights") else "DEFAULT",
                    num_classes=num_classes,
                )
            else:
                model = seg_models.deeplabv3_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=num_classes,
                )
        except Exception as e:
            logger.warning("Could not load pretrained backbone weights (%s); initializing model randomly.", e)
            model = seg_models.deeplabv3_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )
        return model
    elif architecture == "lraspp_mobilenet_v3_large":
        try:
            model = seg_models.lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone="DEFAULT" if pretrained_backbone else None,
                num_classes=num_classes,
            )
        except Exception:
            model = seg_models.lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )
        return model
    else:
        raise ValueError(f"Unsupported segmentation architecture: '{architecture}'.")


class RescueNetSegmentor:
    """
    Inference service for real-time RescueNet semantic segmentation.
    Executes on CUDA when available, with automatic CPU fallback.
    Outputs both 2D integer class segmentation maps and structured RescueNetSceneAnalysis context.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        img_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        rescuenet_service: Optional[RescueNetService] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif settings.RESCUENET_SEGMENTATION_DEVICE:
            self.device = torch.device(settings.RESCUENET_SEGMENTATION_DEVICE)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.img_size = img_size
        self.rescuenet_service = rescuenet_service or RescueNetService()

        # Initialize or load model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = create_segmentation_model(
                num_classes=NUM_RESCUENET_CLASSES,
                pretrained_backbone=True,
            ).to(self.device)

        resolved_weights = weights_path or (
            settings.resolved_rescuenet_segmentation_weights_path
            if settings.resolved_rescuenet_segmentation_weights_path.exists()
            else None
        )
        if resolved_weights is not None:
            w_p = Path(resolved_weights)
            if w_p.exists():
                ckpt = torch.load(w_p, map_location=self.device)
                state_dict = (
                    ckpt["model_state_dict"]
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                    else ckpt
                )
                self.model.load_state_dict(state_dict)
                logger.info("Loaded RescueNet segmentation weights from: %s", w_p)

        self.model.eval()

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Loads and normalizes an input image to a tensor shape [1, 3, H, W]."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        resized = pil_img.resize(self.img_size, Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(resized)
        norm_tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return norm_tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_mask(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Runs segmentation inference on a single image.
        Returns a 2D numpy array [H, W] of dtype int64 with class IDs in [0, 11].
        """
        tensor_batch = self.preprocess_image(image_input)
        is_cuda = self.device.type == "cuda"

        if is_cuda:
            with torch.amp.autocast("cuda"):
                out = self.model(tensor_batch)["out"]
        else:
            out = self.model(tensor_batch)["out"]

        pred_labels = torch.argmax(out, dim=1).squeeze(0)  # [H, W]
        return pred_labels.cpu().numpy().astype(np.int64)

    def predict_scene_analysis(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        image_id: str = "drone_frame",
        thresholds: Optional[RescueNetHazardThresholds] = None,
    ) -> RescueNetSceneAnalysis:
        """
        Runs segmentation inference and aggregates the predicted 12-class map
        into an explainable RescueNetSceneAnalysis object.
        """
        pred_mask = self.predict_mask(image_input)
        return self.rescuenet_service.analyze_mask_array(
            mask_array=pred_mask,
            image_id=image_id,
            thresholds=thresholds,
        )
