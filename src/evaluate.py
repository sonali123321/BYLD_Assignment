"""
evaluate.py
───────────
Comprehensive evaluation of the trained FMCG classifier.

Produces:
  • Top-1 accuracy, F1 macro, per-class accuracy
  • Confusion matrix (saved as PNG)
  • t-SNE visualisation of CLIP embeddings
  • Failure analysis (most-confused image pairs)
  • Classification report

Usage:
    python src/evaluate.py \
        --checkpoint checkpoints/final_model.pt \
        --test_dir data/test \
        --output_dir results/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import build_datasets, build_dataloaders
from model import CLIPClassifier

console = Console()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model and class names from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    cfg = ckpt["config"]

    model = CLIPClassifier(
        num_classes=len(class_names),
        backbone=cfg["model"]["backbone"],
        hidden_dim=cfg["model"]["classifier"]["hidden_dim"],
        dropout=0.0,  # No dropout at inference
        use_bn=cfg["model"]["classifier"]["use_bn"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device), class_names


@torch.no_grad()
def get_predictions_and_features(model, loader, device):
    """Run inference and collect predictions, labels, and embeddings."""
    all_preds, all_labels, all_features, all_probs = [], [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        features = model.extract_features(images)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_features.extend(features.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_features),
    )


def plot_confusion_matrix(cm, class_names, output_path: Path):
    """Plot and save a normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Normalised Confusion Matrix — FMCG Classifier", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"  📊 Confusion matrix saved to {output_path}")


def plot_tsne(features, labels, class_names, output_path: Path):
    """Plot t-SNE of CLIP embeddings coloured by class."""
    console.print("  Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    emb_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[colors[i]], label=cls, alpha=0.8, s=80, edgecolors="white", linewidth=0.5,
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_title("t-SNE of CLIP Embeddings — FMCG Products", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"  🗺️  t-SNE plot saved to {output_path}")


def print_results_table(per_class_acc, class_names, overall_acc, f1):
    """Print a rich table with per-class accuracy."""
    table = Table(title="Per-Class Accuracy", show_header=True, header_style="bold cyan")
    table.add_column("Class", style="white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Status", justify="center")

    for cls, acc in zip(class_names, per_class_acc):
        status = "✅" if acc >= 0.95 else "⚠️" if acc >= 0.80 else "❌"
        color = "green" if acc >= 0.95 else "yellow" if acc >= 0.80 else "red"
        table.add_row(cls, f"[{color}]{acc:.1%}[/]", status)

    table.add_section()
    table.add_row("[bold]Overall[/bold]", f"[bold green]{overall_acc:.1%}[/bold green]", "")
    table.add_row("[bold]F1 Macro[/bold]", f"[bold]{f1:.1%}[/bold]", "")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FMCG classifier")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_tsne", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.print(f"\n[bold cyan]📈 FMCG Classifier Evaluation[/bold cyan]")
    console.print(f"   Checkpoint: {args.checkpoint}")
    console.print(f"   Device:     {device}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_names = load_model(args.checkpoint, device)

    # ── Load test data ────────────────────────────────────────────────────────
    _, _, test_ds, _ = build_datasets(annotations_csv=args.annotations)
    from dataset import build_dataloaders
    _, _, test_loader = build_dataloaders(
        test_ds, test_ds, test_ds, batch_size=32, num_workers=4
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    console.print("[bold]Running inference on test set...[/]")
    preds, labels, probs, features = get_predictions_and_features(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    overall_acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    per_class_acc = []
    for i in range(len(class_names)):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc.append((preds[mask] == labels[mask]).mean())
        else:
            per_class_acc.append(0.0)

    # Print results
    print_results_table(per_class_acc, class_names, overall_acc, f1)

    console.print("\n[bold]Full Classification Report:[/]")
    report = classification_report(labels, preds, target_names=class_names)
    console.print(report)

    # Save report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"F1 Macro: {f1:.4f}\n\n")
        f.write(report)
    console.print(f"\n  📄 Report saved to {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    if not args.no_tsne:
        plot_tsne(features, labels, class_names, output_dir / "tsne_embeddings.png")

    # Final verdict
    console.print(f"\n{'='*50}")
    if overall_acc >= 0.95:
        console.print(f"  🎉 [bold green]TARGET ACHIEVED: {overall_acc:.1%} ≥ 95%[/bold green]")
    else:
        console.print(
            f"  ⚠️  [yellow]Test accuracy {overall_acc:.1%} — below 95% target.[/yellow]\n"
            f"  Suggestions:\n"
            f"    • Collect more images for low-accuracy classes\n"
            f"    • Increase augmentation strength\n"
            f"    • Try ViT-L/14@336 backbone\n"
        )
    console.print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
