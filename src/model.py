"""
model.py
────────
Two-phase training architecture:
  Phase 1 — Linear probe: frozen CLIP backbone, train MLP classifier only.
  Phase 2 — Fine-tune: unfreeze last N transformer blocks, very low LR.

This approach prevents catastrophic forgetting of CLIP's rich visual features
while adapting the representation to FMCG-specific categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Optional


class CLIPClassifier(nn.Module):
    """
    CLIP ViT backbone with a 2-layer MLP classification head.

    The backbone is kept frozen during Phase 1 (linear probe) and partially
    unfrozen during Phase 2 (fine-tuning), preventing catastrophic forgetting.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "ViT-L-14",
        pretrained: str = "openai",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_bn: bool = True,
    ):
        super().__init__()
        # Load CLIP model (vision encoder only)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        self.visual = clip_model.visual  # Vision transformer
        embedding_dim = self.visual.output_dim

        # Freeze backbone initially
        self.freeze_backbone()

        # Classification head: 2-layer MLP
        layers = [nn.Linear(embedding_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        ])
        self.classifier = nn.Sequential(*layers)

        # Store config
        self.backbone_name = backbone
        self.num_classes = num_classes

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1: linear probe)."""
        for param in self.visual.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreeze the last N transformer blocks for fine-tuning (Phase 2).
        Also unfreezes the final LayerNorm and projection layer.
        """
        # First freeze everything
        self.freeze_backbone()

        # ViT transformer blocks are in self.visual.transformer.resblocks
        if hasattr(self.visual, "transformer"):
            blocks = self.visual.transformer.resblocks
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze final norm and projection
        for name, param in self.visual.named_parameters():
            if "ln_post" in name or "proj" in name:
                param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → class logits."""
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.visual.parameters())
        ):
            features = self.visual(x)  # (B, D)
            features = F.normalize(features, dim=-1)  # L2-normalise
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised CLIP features (no classifier)."""
        features = self.visual(x)
        return F.normalize(features, dim=-1)


# ── Utility: count parameters ─────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
