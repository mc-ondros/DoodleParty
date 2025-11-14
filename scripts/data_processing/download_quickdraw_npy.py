#!/usr/bin/env python3
"""Download QuickDraw data plus inappropriate sketches for moderation training."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Iterable

import numpy as np


QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
APPENDIX_BASE_URL = "https://raw.githubusercontent.com/studiomoniker/Quickdraw-appendix/master"

# Positive (inappropriate) categories default to penis drawings from Quickdraw-appendix.
POSITIVE_CATEGORIES = ["penis"]

# Combine legacy safe shapes with a few recognizable objects for negatives.
LEGACY_NEGATIVE = [
    "circle",
    "rectangle",
    "triangle",
    "star",
    "line",
    "square",
    "diamond",
    "heart",
    "plus",
    "cross",
    "crescent",
    "octagon",
    "pentagon",
    "hexagon",
    "spiral",
    "cloud",
    "moon",
    "sun",
    "zig-zag",
    "check",
    "X",
]
DEFAULT_NEGATIVE = sorted(
    set(LEGACY_NEGATIVE + ["dog", "cat", "house", "tree", "car", "flower", "star"])
)


def log(message: str, quiet: bool = False) -> None:
    """Print helper that respects --quiet."""
    if not quiet:
        print(message)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_quickdraw_category(category: str, output_dir: str, *, force: bool, quiet: bool) -> bool:
    """Download a single QuickDraw category (.npy)."""
    ensure_output_dir(output_dir)
    url = f"{QUICKDRAW_BASE_URL}/{category}.npy"
    output_path = os.path.join(output_dir, f"{category}.npy")

    if os.path.exists(output_path) and not force:
        log(f"  ✓ {category}.npy already exists, skipping", quiet)
        return True

    try:
        log(f"  Downloading {category}...", quiet)
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log(f"    ✓ saved {category}.npy ({size_mb:.1f} MB)", quiet)
        return True
    except urllib.error.HTTPError as exc:
        log(f"    ✗ HTTP {exc.code} for {category}", quiet)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for CLI
        log(f"    ✗ Failed to download {category}: {exc}", quiet)

    if os.path.exists(output_path):
        os.remove(output_path)
    return False


def ndjson_to_bitmap(strokes, size: int = 128) -> np.ndarray:
    """Convert stroke lists (Quickdraw-appendix style) into bitmap arrays."""
    from PIL import Image, ImageDraw  # Lazy import so QuickDraw-only users are unaffected

    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)

    for stroke in strokes:
        x_coords, y_coords = stroke
        if len(x_coords) < 2:
            continue

        for idx in range(len(x_coords) - 1):
            x1 = int(x_coords[idx] * size / 256)
            y1 = int(y_coords[idx] * size / 256)
            x2 = int(x_coords[idx + 1] * size / 256)
            y2 = int(y_coords[idx + 1] * size / 256)
            draw.line([x1, y1, x2, y2], fill=0, width=2)

    return np.array(image, dtype=np.uint8)


def download_inappropriate_category(
    category: str,
    output_dir: str,
    *,
    max_samples: int,
    force: bool,
    quiet: bool,
) -> bool:
    """Fetch Quickdraw-appendix NDJSON and convert it to numpy bitmaps."""

    ensure_output_dir(output_dir)
    output_path = os.path.join(output_dir, f"{category}.npy")

    if os.path.exists(output_path) and not force:
        log(f"  ✓ {category}.npy already exists, skipping", quiet)
        return True

    sources = [
        f"{APPENDIX_BASE_URL}/{category}-raw.ndjson",
        f"{APPENDIX_BASE_URL}/{category}-simplified.ndjson",
    ]

    temp_file = os.path.join(output_dir, f"{category}_temp.ndjson")
    for url in sources:
        try:
            log(f"  Downloading {url.split('/')[-1]}...", quiet)
            urllib.request.urlretrieve(url, temp_file)

            bitmaps = []
            with open(temp_file, "r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if idx >= max_samples:
                        break

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    drawing = payload.get("drawing")
                    if drawing:
                        bitmaps.append(ndjson_to_bitmap(drawing).flatten())

            np.save(output_path, np.array(bitmaps, dtype=np.uint8))
            if os.path.exists(temp_file):
                os.remove(temp_file)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            log(
                f"    ✓ {category}.npy created with {len(bitmaps)} samples ({size_mb:.1f} MB)",
                quiet,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for CLI
            log(f"    ✗ Failed via {url}: {exc}", quiet)
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    log(f"  ✗ Could not retrieve {category} from Quickdraw-appendix", quiet)
    return False


def describe_categories(label: str, categories: Iterable[str], quiet: bool) -> None:
    items = list(categories)
    if items:
        log(f"{label} ({len(items)}): {items}", quiet)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download QuickDraw and Quickdraw-appendix datasets for moderation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full binary dataset (default settings)
  python download_quickdraw_npy.py --output-dir data/raw

  # Only inappropriate drawings
  python download_quickdraw_npy.py --output-dir data/raw --positive-only

  # Override QuickDraw categories
  python download_quickdraw_npy.py --categories cat dog house

  # Preview without downloading
  python download_quickdraw_npy.py --dry-run
        """,
    )

    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory for .npy files")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Override QuickDraw (appropriate) categories",
    )
    parser.add_argument(
        "--inappropriate-categories",
        type=str,
        nargs="*",
        default=POSITIVE_CATEGORIES,
        help="Quickdraw-appendix categories to download",
    )
    parser.add_argument(
        "--appropriate-categories",
        type=str,
        nargs="*",
        default=DEFAULT_NEGATIVE,
        help="QuickDraw categories to download when --categories is not provided",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Download only inappropriate (positive) categories",
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Download only appropriate (negative) categories",
    )
    parser.add_argument(
        "--max-inappropriate",
        type=int,
        default=20000,
        help="Max samples to convert per inappropriate category",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument("--dry-run", action="store_true", help="List actions without downloading")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")

    args = parser.parse_args()
    quiet = args.quiet

    quickdraw_categories = args.categories or args.appropriate_categories
    inappropriate_categories = args.inappropriate_categories

    if args.positive_only:
        quickdraw_categories = []
    if args.negative_only:
        inappropriate_categories = []

    log("=" * 70, quiet)
    log("Content Moderation Dataset Downloader", quiet)
    log("=" * 70, quiet)
    log(f"Output directory: {args.output_dir}\n", quiet)

    describe_categories("Inappropriate", inappropriate_categories, quiet)
    describe_categories("Appropriate", quickdraw_categories, quiet)

    if args.dry_run:
        log("\nDRY RUN complete — no files downloaded.", quiet)
        return 0

    ensure_output_dir(args.output_dir)

    inappropriate_success = 0
    if inappropriate_categories:
        log("\n[1/2] Downloading inappropriate categories", quiet)
        for category in inappropriate_categories:
            if download_inappropriate_category(
                category,
                args.output_dir,
                max_samples=args.max_inappropriate,
                force=args.force_download,
                quiet=quiet,
            ):
                inappropriate_success += 1

    appropriate_success = 0
    if quickdraw_categories:
        log("\n[2/2] Downloading appropriate QuickDraw categories", quiet)
        for category in quickdraw_categories:
            if download_quickdraw_category(
                category,
                args.output_dir,
                force=args.force_download,
                quiet=quiet,
            ):
                appropriate_success += 1

    total_inappropriate = len(inappropriate_categories)
    total_appropriate = len(quickdraw_categories)
    total = total_inappropriate + total_appropriate

    log("\n" + "=" * 70, quiet)
    log("Download Summary", quiet)
    log("=" * 70, quiet)
    if total_inappropriate:
        log(
            f"Inappropriate: {inappropriate_success}/{total_inappropriate} successful",
            quiet,
        )
    if total_appropriate:
        log(f"Appropriate: {appropriate_success}/{total_appropriate} successful", quiet)
    if total:
        log(f"Total: {inappropriate_success + appropriate_success}/{total} successful", quiet)

    failures = (total_inappropriate - inappropriate_success) + (total_appropriate - appropriate_success)
    if failures:
        log("\n⚠ Some downloads failed. Re-run to retry missing categories.", quiet)
        return 1

    log("\n✓ All datasets downloaded successfully!", quiet)
    log("Next step: python scripts/training/train.py", quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
