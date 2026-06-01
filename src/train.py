"""
train.py
────────
Two-phase training loop for FMCG CLIP classifier.

Phase 1 — Linear Probe (30 epochs):
  • Backbone fully frozen
  • Only the 2-layer MLP head is trained
  • High learning rate (1e-3), fast convergence

Phase 2 — Fine-tuning (20 epochs):
  • Unfreeze last 2 transformer blocks
  • Very low learning rate (5e-6) to avoid catastrophic forgetting
  • Cosine annealing with warmup

Both phases use MixUp / CutMix and label smoothing for regularisation.

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from sklearn.metrics import accuracy_score

from augmentation import mixup_data, cutmix_data, mixup_criterion
from dataset import build_datasets, build_dataloaders
from model import CLIPClassifier, count_parameters

console = Console()


# ── Training utilities ─────────────────────────────────────────────────────────

def get_cosine_scheduler_with_warmup(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_fraction: float = 0.01,
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_fraction + 0.5 * (1 - min_lr_fraction) * (
            1 + np.cos(np.pi * progress)
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Single epoch train / eval ──────────────────────────────────────────────────

def train_epoch(
    model, loader, criterion, optimizer, device, cfg, use_mixup: bool = True
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    mixup_alpha = cfg.training.mixup_alpha
    cutmix_alpha = cfg.training.cutmix_alpha

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Choose augmentation strategy randomly
        r = np.random.rand()
        if use_mixup and r < 0.5 and mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        elif use_mixup and r < 0.75 and cutmix_alpha > 0:
            images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "preds": all_preds,
        "labels": all_labels,
    }


# ── Phase runner ───────────────────────────────────────────────────────────────

def run_phase(
    phase_name: str,
    model,
    train_loader,
    val_loader,
    cfg_phase,
    cfg,
    device,
    checkpoint_dir: Path,
) -> float:
    """Train one phase. Returns best validation accuracy."""
    console.print(f"\n[bold cyan]━━ {phase_name} ━━[/bold cyan]")
    params = count_parameters(model)
    console.print(f"   Trainable: {params['trainable']:,} / {params['total']:,} params")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_phase.lr,
        weight_decay=cfg_phase.weight_decay,
    )
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg_phase.epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    early_stop = EarlyStopping(patience=cfg.training.patience)

    best_val_acc = 0.0
    best_ckpt = checkpoint_dir / f"{phase_name.lower().replace(' ', '_')}_best.pt"

    for epoch in range(1, cfg_phase.epochs + 1):
        t0 = time.time()
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg,
            use_mixup=(epoch > 5),  # Warm up before applying MixUp
        )
        val_stats = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        improved = val_stats["acc"] > best_val_acc
        if improved:
            best_val_acc = val_stats["acc"]
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_acc": best_val_acc},
                best_ckpt,
            )

        icon = "✅" if improved else "  "
        console.print(
            f"  Ep {epoch:02d}/{cfg_phase.epochs} │ "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.3f} │ "
            f"val_acc={val_stats['acc']:.3f} │ lr={lr:.2e} │ {elapsed:.1f}s {icon}"
        )

        if early_stop.step(val_stats["acc"]):
            console.print(f"  [yellow]Early stopping triggered at epoch {epoch}[/]")
            break

    console.print(f"\n  [green]Best val acc: {best_val_acc:.1%}[/green]  →  {best_ckpt}")
    # Reload best checkpoint
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FMCG CLIP classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default=None, help="Override config image_dir")
    parser.add_argument("--annotations", default=None, help="Override config annotations_csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.image_dir = args.data_dir
    if args.annotations:
        cfg.data.annotations_csv = args.annotations

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold]Device:[/] {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    console.print("[bold]Loading datasets...[/]")
    train_ds, val_ds, test_ds, class_names = build_datasets(
        annotations_csv=cfg.data.annotations_csv,
        image_size=cfg.data.image_size,
        test_split=cfg.data.test_split,
        val_split=cfg.data.val_split,
    )
    console.print(
        f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | "
        f"Classes: {len(class_names)}"
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    console.print("\n[bold]Building model...[/]")
    model = CLIPClassifier(
        num_classes=len(class_names),
        backbone=cfg.model.backbone,
        hidden_dim=cfg.model.classifier.hidden_dim,
        dropout=cfg.model.classifier.dropout,
        use_bn=cfg.model.classifier.use_bn,
    ).to(device)

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Linear probe ──────────────────────────────────────────────
    model.freeze_backbone()
    phase1_acc = run_phase(
        "Phase 1 — Linear Probe",
        model, train_loader, val_loader,
        cfg.training.phase1, cfg, device, checkpoint_dir,
    )

    # ── Phase 2: Fine-tune ─────────────────────────────────────────────────
    model.unfreeze_last_n_blocks(n=cfg.training.phase2.unfreeze_layers)
    phase2_acc = run_phase(
        "Phase 2 — Fine-tuning",
        model, train_loader, val_loader,
        cfg.training.phase2, cfg, device, checkpoint_dir,
    )

    # ── Final test evaluation ──────────────────────────────────────────────
    console.print("\n[bold cyan]━━ Final Test Evaluation ━━[/bold cyan]")
    criterion = nn.CrossEntropyLoss()
    test_stats = eval_epoch(model, test_loader, criterion, device)
    console.print(f"\n  🎯 [bold green]Test Accuracy: {test_stats['acc']:.1%}[/bold green]")

    # Save final model
    final_ckpt = checkpoint_dir / "final_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "config": OmegaConf.to_container(cfg),
        "test_accuracy": test_stats["acc"],
    }, final_ckpt)
    console.print(f"  💾 Final model saved to {final_ckpt}")

    # Summary
    console.print(f"""
┌──────────────────────────────────────┐
│  Training Summary                    │
│  Phase 1 (linear probe):  {phase1_acc:.1%}      │
│  Phase 2 (fine-tune):     {phase2_acc:.1%}      │
│  Final test accuracy:     {test_stats['acc']:.1%}      │
└──────────────────────────────────────┘
""")
    target = 0.95
    if test_stats["acc"] >= target:
        console.print(f"  ✅ [bold green]Target of {target:.0%} ACHIEVED![/bold green]")
    else:
        console.print(f"  ⚠️  [yellow]Target of {target:.0%} not yet reached. Consider more augmentation or data.[/yellow]")


if __name__ == "__main__":
    main()
