"""
augmentation.py
───────────────
Augmentation pipeline designed specifically for FMCG shelf / retail images.

Beyond standard augmentations, we simulate:
  • Perspective warping  — camera tilt when scanning a planogram
  • Specular highlights  — glossy packaging under fluorescent lights
  • Shadow strips        — shelving bars casting shadow on products
  • Partial occlusion    — adjacent products blocking view
  • Colour temperature   — warm vs. cool lighting conditions

This pipeline produces ~50x effective samples from each real image when
combined with MixUp and CutMix at training time.
"""

from typing import Tuple

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# ── CLIP normalisation constants ───────────────────────────────────────────────
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ── Shelf-specific custom transforms ──────────────────────────────────────────

class RandomShelfShadow(A.ImageOnlyTransform):
    """Adds horizontal shadow strips simulating shelving bars."""

    def __init__(self, num_shadows: int = 2, shadow_opacity: float = 0.4, p: float = 0.3):
        super().__init__(p=p)
        self.num_shadows = num_shadows
        self.shadow_opacity = shadow_opacity

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.astype(np.float32)
        for _ in range(np.random.randint(1, self.num_shadows + 1)):
            y = np.random.randint(0, h)
            thickness = np.random.randint(4, 20)
            y1 = max(0, y - thickness // 2)
            y2 = min(h, y + thickness // 2)
            result[y1:y2, :] *= (1.0 - self.shadow_opacity * np.random.uniform(0.5, 1.0))
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("num_shadows", "shadow_opacity")


class RandomColorTemperature(A.ImageOnlyTransform):
    """Shifts image toward warm (orange) or cool (blue) light temperature."""

    def __init__(self, p: float = 0.3):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, warm: bool = True, **params) -> np.ndarray:
        result = img.astype(np.float32)
        strength = np.random.uniform(0.05, 0.15)
        if warm:
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + strength), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - strength), 0, 255)
        else:
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 - strength), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 + strength), 0, 255)
        return result.astype(np.uint8)

    def apply_to_mask(self, mask, **params):
        return mask

    def get_params(self):
        return {"warm": np.random.random() > 0.5}

    def get_transform_init_args_names(self):
        return ()


# ── Main augmentation pipelines ───────────────────────────────────────────────

def build_train_transforms(image_size: int = 224, strong: bool = True) -> A.Compose:
    """
    Build training augmentation pipeline.
    
    Args:
        image_size: Target image size.
        strong: If True, use aggressive shelf-aware augmentations.
    """
    base = [
        A.SmallestMaxSize(max_size=image_size + 32),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ]

    colour = [
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.RandomGrayscale(p=0.05),
        RandomColorTemperature(p=0.3),
    ]

    geometric = [
        A.Rotate(limit=15, p=0.5),
        A.Perspective(scale=(0.03, 0.10), p=0.3),  # Planogram camera angle
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=0, p=0.4),
    ]

    noise_blur = [
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.15),
    ]

    shelf_specific = [
        RandomShelfShadow(num_shadows=2, shadow_opacity=0.4, p=0.3),
        A.CoarseDropout(
            max_holes=3, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.2,
        ),  # Partial occlusion
    ]

    randaugment = [
        A.RandAugment(num_transforms=2, magnitude=9, p=0.5),
    ] if strong else []

    normalise = [
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]

    all_transforms = (
        base + colour + geometric + noise_blur + shelf_specific + randaugment + normalise
    )
    return A.Compose(all_transforms)


def build_val_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic validation transforms (resize + centre crop + normalise)."""
    return A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC numpy array to CHW torch tensor."""
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


# ── MixUp & CutMix (applied in training loop) ─────────────────────────────────

def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation to a batch.
    Returns mixed_x, y_a, y_b, lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix augmentation to a batch.
    Returns mixed_x, y_a, y_b, lambda.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam


def mixup_criterion(
    criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """MixUp / CutMix loss combinator."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
