"""
data_collection.py
──────────────────
Downloads FMCG product images from the web using DuckDuckGo image search.
Organises images into per-class subdirectories and applies basic quality
filtering (size, aspect ratio, format validation).

Usage:
    python src/data_collection.py \
        --categories "cola_can,chips_lays,maggi_noodles" \
        --images_per_class 20 \
        --output_dir data/raw
"""

import argparse
import hashlib
import io
import os
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from duckduckgo_search import DDGS

console = Console()

# ── Search queries per FMCG category ──────────────────────────────────────────
SEARCH_QUERIES = {
    "cola_can_red": [
        "Coca-Cola can product photo white background",
        "Coke 330ml can FMCG retail",
        "red cola can front view",
    ],
    "cola_can_blue": [
        "Pepsi can product photo white background",
        "Pepsi 330ml can retail shelf",
        "blue pepsi can front view",
    ],
    "chips_lays_classic": [
        "Lays classic potato chips packet product photo",
        "Lays original flavor snack bag",
        "yellow Lays chips front view white background",
    ],
    "chips_lays_masala": [
        "Lays masala chips packet India",
        "Lays spicy masala flavor bag product photo",
    ],
    "maggi_noodles_masala": [
        "Maggi 2-minute noodles packet product photo",
        "Maggi masala noodles pack white background",
        "Nestle Maggi noodles retail pack",
    ],
    "tide_detergent_1kg": [
        "Tide detergent powder 1kg box product photo",
        "Tide laundry detergent retail pack blue",
    ],
    "amul_butter_500g": [
        "Amul butter 500g carton product photo",
        "Amul dairy butter box retail India",
    ],
    "britannia_biscuits": [
        "Britannia Good Day biscuits packet product photo",
        "Britannia cookies retail pack",
    ],
    "parle_g_biscuits": [
        "Parle-G biscuit packet product photo",
        "Parle G glucose biscuits retail pack India",
    ],
    "surf_excel_detergent": [
        "Surf Excel detergent powder packet product photo",
        "Surf Excel blue laundry detergent retail pack",
    ],
}


class FMCGImageDownloader:
    """Downloads and validates FMCG product images from the web."""

    def __init__(
        self,
        output_dir: str,
        images_per_class: int = 20,
        min_size: int = 128,
        max_size: int = 2048,
        min_aspect: float = 0.3,
        max_aspect: float = 3.0,
        request_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.images_per_class = images_per_class
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        )

    def _is_valid_image(self, img: Image.Image) -> bool:
        """Check image quality constraints."""
        w, h = img.size
        if w < self.min_size or h < self.min_size:
            return False
        if w > self.max_size or h > self.max_size:
            return False
        aspect = w / h
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False
        return True

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download a single image with validation."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return img if self._is_valid_image(img) else None
        except Exception:
            return None

    def _get_image_hash(self, img: Image.Image) -> str:
        """Perceptual hash to deduplicate images."""
        img_small = img.resize((16, 16), Image.LANCZOS).convert("L")
        pixels = list(img_small.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return hashlib.md5(bits.encode()).hexdigest()

    def download_class(self, class_name: str, queries: List[str]) -> int:
        """Download images for a single class. Returns count of saved images."""
        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        saved, seen_hashes = 0, set()

        with DDGS() as ddgs:
            for query in queries:
                if saved >= self.images_per_class:
                    break
                try:
                    results = ddgs.images(
                        query,
                        max_results=min(50, (self.images_per_class - saved) * 3),
                        type_image="photo",
                    )
                    for result in results:
                        if saved >= self.images_per_class:
                            break
                        url = result.get("image", "")
                        if not url:
                            continue
                        img = self._download_image(url)
                        if img is None:
                            continue
                        img_hash = self._get_image_hash(img)
                        if img_hash in seen_hashes:
                            continue  # Skip duplicate
                        seen_hashes.add(img_hash)
                        # Save as JPEG
                        ext = ".jpg"
                        fname = class_dir / f"{class_name}_{saved:04d}{ext}"
                        img.save(fname, "JPEG", quality=95)
                        saved += 1
                        time.sleep(self.request_delay)
                except Exception as e:
                    console.print(f"[yellow]Warning: query '{query}' failed: {e}[/]")

        return saved

    def run(self, categories: List[str]) -> dict:
        """Download images for all requested categories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats = {}

        console.print(f"\n[bold cyan]📦 Downloading FMCG images[/bold cyan]")
        console.print(
            f"   Categories: {len(categories)} | Target per class: {self.images_per_class}\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=len(categories))
            for cat in categories:
                progress.update(task, description=f"[cyan]{cat}[/]")
                queries = SEARCH_QUERIES.get(cat, [f"{cat} product photo white background"])
                n = self.download_class(cat, queries)
                stats[cat] = n
                progress.advance(task)
                console.print(f"  ✅ {cat}: {n} images")

        total = sum(stats.values())
        console.print(
            f"\n[bold green]✨ Done! Downloaded {total} images across {len(categories)} categories.[/bold green]"
        )
        return stats


def main():
    parser = argparse.ArgumentParser(description="Download FMCG product images")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(SEARCH_QUERIES.keys()),
        help="Comma-separated list of category names",
    )
    parser.add_argument("--images_per_class", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (s)")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    downloader = FMCGImageDownloader(
        output_dir=args.output_dir,
        images_per_class=args.images_per_class,
        request_delay=args.delay,
    )
    downloader.run(categories)


if __name__ == "__main__":
    main()
