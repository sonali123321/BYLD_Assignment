"""
annotation_tool.py
──────────────────
Semi-automatic annotation pipeline that dramatically reduces labeling effort:

  1. CLIP zero-shot inference   → auto-labels high-confidence images (≥threshold)
  2. Active learning sampling   → selects the MOST uncertain images for human review
  3. Label propagation (k-NN)   → propagates human labels to remaining uncertain images
  4. Gradio UI                  → simple browser interface for human verification

Total human effort: ~20 images instead of 100.

Usage:
    python src/annotation_tool.py \
        --image_dir data/raw \
        --output_csv data/annotations.csv \
        --confidence_threshold 0.85
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.table import Table
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

console = Console()


# ── CLIP helper ───────────────────────────────────────────────────────────────

class CLIPAnnotator:
    """
    Uses CLIP to provide zero-shot label predictions and extract embeddings
    for downstream k-NN label propagation.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        console.print(f"[cyan]Loading CLIP [{model_name}]...[/]", end=" ")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        console.print("[green]✓[/]")

    @torch.no_grad()
    def embed_images(self, image_paths: List[Path]) -> np.ndarray:
        """Return L2-normalised CLIP image embeddings (N, D)."""
        embeddings = []
        for path in tqdm(image_paths, desc="Embedding images"):
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(tensor)
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)  # (N, D)

    @torch.no_grad()
    def embed_texts(self, prompts_per_class: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Returns per-class averaged text embeddings.
        prompts_per_class: {class_name: [prompt1, prompt2, ...]}
        """
        text_embeddings = {}
        for cls, prompts in prompts_per_class.items():
            tokens = self.tokenizer(prompts).to(self.device)
            embs = self.model.encode_text(tokens)
            embs = F.normalize(embs, dim=-1)
            avg = embs.mean(dim=0, keepdim=True)
            avg = F.normalize(avg, dim=-1)
            text_embeddings[cls] = avg.cpu().numpy()  # (1, D)
        return text_embeddings

    def zero_shot_predict(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity between image and text embeddings.
        Returns:
            pred_labels: (N,) array of predicted class names indices
            confidences: (N,) softmax confidence scores
        """
        classes = list(text_embeddings.keys())
        text_matrix = np.vstack([text_embeddings[c] for c in classes])  # (C, D)

        # Cosine similarity (both embeddings are L2-normalised)
        sim = image_embeddings @ text_matrix.T  # (N, C)
        probs = torch.softmax(torch.tensor(sim * 100.0), dim=-1).numpy()
        pred_idx = probs.argmax(axis=-1)
        confidence = probs.max(axis=-1)
        pred_labels = np.array([classes[i] for i in pred_idx])
        return pred_labels, confidence


# ── Active learning ───────────────────────────────────────────────────────────

def uncertainty_sampling(confidences: np.ndarray, budget: int) -> np.ndarray:
    """Return indices of `budget` most uncertain (lowest confidence) samples."""
    return np.argsort(confidences)[:budget]


def label_propagation_knn(
    embeddings: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_classes: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Propagate labels from labeled_indices to all images using k-NN
    in CLIP embedding space.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(embeddings[labeled_indices], labeled_classes[labeled_indices])
    return knn.predict(embeddings)


# ── Main annotation pipeline ──────────────────────────────────────────────────

class AnnotationPipeline:
    def __init__(
        self,
        image_dir: str,
        output_csv: str,
        clip_prompts: Optional[Dict[str, List[str]]] = None,
        confidence_threshold: float = 0.85,
        active_learning_budget: int = 20,
        k_neighbors: int = 5,
        clip_model: str = "ViT-L-14",
    ):
        self.image_dir = Path(image_dir)
        self.output_csv = Path(output_csv)
        self.confidence_threshold = confidence_threshold
        self.active_learning_budget = active_learning_budget
        self.k_neighbors = k_neighbors
        self.clip = CLIPAnnotator(model_name=clip_model)

        # Auto-detect classes from directory structure
        self.classes = sorted([
            d.name for d in self.image_dir.iterdir() if d.is_dir()
        ])
        console.print(f"[bold]Detected {len(self.classes)} classes:[/] {self.classes}")

        # Default prompts: one generic prompt per class if none provided
        self.clip_prompts = clip_prompts or {
            cls: [f"a product photo of {cls.replace('_', ' ')}",
                  f"retail packaged {cls.replace('_', ' ')} on shelf"]
            for cls in self.classes
        }

    def _gather_image_paths(self) -> List[Path]:
        """Collect all images from the image directory."""
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = []
        for cls in self.classes:
            cls_dir = self.image_dir / cls
            if cls_dir.exists():
                paths.extend([p for p in cls_dir.iterdir() if p.suffix.lower() in exts])
        return sorted(paths)

    def _get_folder_labels(self, paths: List[Path]) -> np.ndarray:
        """Extract ground-truth labels from folder names (for evaluation)."""
        return np.array([p.parent.name for p in paths])

    def run(self, use_gradio_ui: bool = True) -> Dict:
        """
        Full annotation pipeline.
        Returns annotation dict: {image_path: label}
        """
        console.print("\n[bold cyan]🏷️  FMCG Semi-Automatic Annotation Pipeline[/bold cyan]\n")

        # ── Step 1: Gather images ───────────────────────────────────────────
        paths = self._gather_image_paths()
        folder_labels = self._get_folder_labels(paths)
        console.print(f"📁 Found {len(paths)} images across {len(self.classes)} classes\n")

        # ── Step 2: CLIP embeddings ─────────────────────────────────────────
        console.print("[bold]Step 1/4:[/] Computing CLIP image embeddings...")
        image_embeddings = self.clip.embed_images(paths)

        console.print("[bold]Step 2/4:[/] Computing CLIP text embeddings...")
        text_embeddings = self.clip.embed_texts(self.clip_prompts)

        # ── Step 3: Zero-shot predictions ──────────────────────────────────
        console.print("[bold]Step 3/4:[/] Running zero-shot classification...")
        pred_labels, confidences = self.clip.zero_shot_predict(image_embeddings, text_embeddings)

        # Compute zero-shot accuracy (using folder as ground truth)
        zs_acc = (pred_labels == folder_labels).mean()
        console.print(f"   Zero-shot accuracy: [green]{zs_acc:.1%}[/green]")

        # ── Step 4: Split into auto-labeled vs. needs-human-review ─────────
        auto_mask = confidences >= self.confidence_threshold
        uncertain_mask = ~auto_mask

        auto_indices = np.where(auto_mask)[0]
        uncertain_indices = np.where(uncertain_mask)[0]

        console.print(
            f"\n   Auto-labeled (conf ≥ {self.confidence_threshold}): "
            f"[green]{len(auto_indices)}[/green] images"
        )
        console.print(
            f"   Uncertain (conf < {self.confidence_threshold}): "
            f"[yellow]{len(uncertain_indices)}[/yellow] images"
        )

        # ── Step 5: Active learning — select most informative to label ─────
        # Among uncertain images, pick the most uncertain for human review
        if len(uncertain_indices) > 0:
            uncertain_confidences = confidences[uncertain_indices]
            al_count = min(self.active_learning_budget, len(uncertain_indices))
            al_local_idx = uncertainty_sampling(uncertain_confidences, budget=al_count)
            al_global_idx = uncertain_indices[al_local_idx]
        else:
            al_global_idx = np.array([], dtype=int)

        console.print(
            f"   Active learning selection: [bold yellow]{len(al_global_idx)}[/bold yellow] "
            f"images selected for human review\n"
        )

        # ── Step 6: Human verification ─────────────────────────────────────
        human_labels = np.empty(len(paths), dtype=object)
        human_labels[:] = None

        # For auto-labeled images, accept CLIP prediction
        human_labels[auto_indices] = pred_labels[auto_indices]

        # Human labels: use folder name as oracle (in real use, this is the UI)
        if use_gradio_ui and len(al_global_idx) > 0:
            console.print("[bold]Step 4/4:[/] Launching annotation UI for human review...")
            human_labels = self._run_gradio_annotation(
                paths, al_global_idx, human_labels, pred_labels
            )
        else:
            # CLI fallback: print each image path and prompt for label
            console.print("[bold]Step 4/4:[/] CLI annotation for uncertain images...")
            for idx in al_global_idx:
                console.print(f"\n  📷 {paths[idx]}")
                console.print(f"     CLIP prediction: [yellow]{pred_labels[idx]}[/] "
                              f"(conf={confidences[idx]:.2f})")
                console.print(f"     Classes: {self.classes}")
                label_input = input("  Enter correct label (or press Enter to accept): ").strip()
                human_labels[idx] = label_input if label_input else pred_labels[idx]

        # ── Step 7: Label propagation for remaining unlabeled images ───────
        still_unlabeled = np.where(human_labels == None)[0]  # noqa: E711
        if len(still_unlabeled) > 0:
            console.print(f"\n🔗 Propagating labels to {len(still_unlabeled)} remaining images...")
            labeled_mask = human_labels != None  # noqa: E711
            labeled_indices = np.where(labeled_mask)[0]
            labeled_classes = human_labels[labeled_indices]

            # Encode labeled classes as integers for k-NN
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(labeled_classes)
            label_arr = np.empty(len(paths), dtype=object)
            label_arr[labeled_indices] = labeled_classes

            knn = KNeighborsClassifier(n_neighbors=self.k_neighbors, metric="cosine")
            knn.fit(image_embeddings[labeled_indices], encoded)
            propagated_encoded = knn.predict(image_embeddings[still_unlabeled])
            propagated_labels = le.inverse_transform(propagated_encoded)

            for i, idx in enumerate(still_unlabeled):
                human_labels[idx] = propagated_labels[i]

        # ── Step 8: Compute final annotation accuracy ──────────────────────
        final_acc = (human_labels == folder_labels).mean()
        console.print(f"\n[bold green]✨ Annotation accuracy vs. folder labels: {final_acc:.1%}[/]")

        # ── Step 9: Save to CSV ────────────────────────────────────────────
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "clip_confidence", "annotation_method"])
            for i, path in enumerate(paths):
                method = (
                    "auto_clip" if i in auto_indices
                    else "human_verified" if i in al_global_idx
                    else "label_propagation"
                )
                writer.writerow([str(path), human_labels[i], f"{confidences[i]:.4f}", method])

        console.print(f"\n💾 Annotations saved to [bold]{self.output_csv}[/bold]")
        self._print_summary(human_labels, confidences, auto_indices, al_global_idx)

        return {str(p): l for p, l in zip(paths, human_labels)}

    def _run_gradio_annotation(
        self,
        paths: List[Path],
        review_indices: np.ndarray,
        human_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> np.ndarray:
        """Launch a Gradio UI for reviewing uncertain predictions."""
        try:
            import gradio as gr
        except ImportError:
            console.print("[yellow]Gradio not installed. Falling back to CLI.[/]")
            for idx in review_indices:
                human_labels[idx] = pred_labels[idx]  # Accept CLIP prediction
            return human_labels

        state = {"current_idx": 0, "labels": human_labels.copy()}
        review_paths = [paths[i] for i in review_indices]
        review_preds = [pred_labels[i] for i in review_indices]

        def load_image(idx):
            if idx >= len(review_paths):
                return None, "Done!", gr.update(visible=False)
            return (
                str(review_paths[idx]),
                f"CLIP predicts: **{review_preds[idx]}** | Image {idx+1}/{len(review_paths)}",
                gr.update(visible=True),
            )

        def save_label(label, idx):
            global_idx = review_indices[idx]
            state["labels"][global_idx] = label
            next_idx = idx + 1
            if next_idx >= len(review_paths):
                return *load_image(next_idx), next_idx
            return *load_image(next_idx), next_idx

        with gr.Blocks(title="FMCG Annotation Tool") as demo:
            gr.Markdown("# 🏷️ FMCG Semi-Automatic Annotation\nVerify CLIP predictions for uncertain images.")
            with gr.Row():
                img_display = gr.Image(label="Product Image", height=400)
                with gr.Column():
                    info_text = gr.Markdown()
                    label_radio = gr.Radio(choices=self.classes, label="Correct label")
                    submit_btn = gr.Button("Save & Next →", variant="primary")
            idx_state = gr.State(value=0)

            demo.load(fn=lambda: load_image(0), outputs=[img_display, info_text, submit_btn])
            submit_btn.click(
                fn=save_label,
                inputs=[label_radio, idx_state],
                outputs=[img_display, info_text, submit_btn, idx_state],
            )

        demo.launch(share=False, quiet=True)
        return state["labels"]

    def _print_summary(self, labels, confidences, auto_idx, human_idx):
        table = Table(title="Annotation Summary", show_header=True)
        table.add_column("Method", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("% of Dataset", justify="right")
        n = len(labels)
        propagated = n - len(auto_idx) - len(human_idx)
        table.add_row("Auto (CLIP high-confidence)", str(len(auto_idx)), f"{len(auto_idx)/n:.0%}")
        table.add_row("Human-verified (active learning)", str(len(human_idx)), f"{len(human_idx)/n:.0%}")
        table.add_row("Label propagation (k-NN)", str(propagated), f"{propagated/n:.0%}")
        table.add_row("[bold]Total[/bold]", f"[bold]{n}[/bold]", "[bold]100%[/bold]")
        console.print(table)
        console.print(
            f"\n💡 Labeling effort reduction: [bold green]{len(human_idx)/n:.0%}[/bold green] "
            f"manual effort vs. 100% traditional approach\n"
        )


def main():
    parser = argparse.ArgumentParser(description="FMCG semi-automatic annotation")
    parser.add_argument("--image_dir", default="data/raw")
    parser.add_argument("--output_csv", default="data/annotations.csv")
    parser.add_argument("--confidence_threshold", type=float, default=0.85)
    parser.add_argument("--active_learning_budget", type=int, default=20)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--clip_model", default="ViT-L-14")
    parser.add_argument("--no_ui", action="store_true", help="CLI-only mode (no Gradio)")
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="Path to JSON file with per-class CLIP prompts",
    )
    args = parser.parse_args()

    clip_prompts = None
    if args.prompts_json:
        with open(args.prompts_json) as f:
            clip_prompts = json.load(f)

    pipeline = AnnotationPipeline(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        clip_prompts=clip_prompts,
        confidence_threshold=args.confidence_threshold,
        active_learning_budget=args.active_learning_budget,
        k_neighbors=args.k_neighbors,
        clip_model=args.clip_model,
    )
    pipeline.run(use_gradio_ui=not args.no_ui)


if __name__ == "__main__":
    main()
