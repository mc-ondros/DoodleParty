#!/usr/bin/env python3
"""
Download QuickDraw dataset as numpy files.

Downloads from Google's QuickDraw dataset in NumPy bitmap format (28x28 pre-processed).
The official numpy format is hosted at: https://quickdraw.withgoogle.com/data

This script downloads and organizes the dataset for training the binary classifier
(penis vs safe drawings).
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


# Official Google QuickDraw dataset URL
QUICKDRAW_BASE_URL = "https://quickdraw.withgoogle.com/data/numpy_bitmap"

# Categories for binary classification
# Positive class: penis (offensive content)
POSITIVE_CATEGORIES = [
    "penis"
]

# Negative categories: common safe shapes (22 categories)
NEGATIVE_CATEGORIES = [
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
    "X"
]


def download_category(category: str, output_dir: str, verbose: bool = True) -> bool:
    """
    Download a single QuickDraw category.
    
    Args:
        category: Category name (e.g., 'penis', 'circle')
        output_dir: Directory to save the .npy file
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct download URL
    url = f"{QUICKDRAW_BASE_URL}/{category}.npy"
    output_path = os.path.join(output_dir, f"{category}.npy")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        if verbose:
            print(f"✓ {category}.npy already exists")
        return True
    
    try:
        if verbose:
            print(f"⬇ Downloading {category}...")
        
        # Download with progress callback
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                if block_num % 50 == 0 and verbose:
                    print(f"   {percent}% ({downloaded}/{total_size} bytes)")
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        
        if verbose:
            file_size = os.path.getsize(output_path)
            print(f"✓ Downloaded {category}.npy ({file_size:,} bytes)")
        
        return True
    
    except urllib.error.HTTPError as e:
        print(f"✗ Error downloading {category}: HTTP {e.code}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    except Exception as e:
        print(f"✗ Error downloading {category}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download QuickDraw dataset for DoodleParty',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all categories (1 positive + 21 negative)
  python download_quickdraw_npy.py --output-dir data/raw
  
  # Download only positive category
  python download_quickdraw_npy.py --output-dir data/raw --positive-only
  
  # Download specific categories
  python download_quickdraw_npy.py --output-dir data/raw --categories penis circle star
  
  # Show what would be downloaded without downloading
  python download_quickdraw_npy.py --output-dir data/raw --dry-run
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded .npy files (default: data/raw)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        help='Specific categories to download (overrides default selection)'
    )
    
    parser.add_argument(
        '--positive-only',
        action='store_true',
        help='Download only positive (offensive) categories'
    )
    
    parser.add_argument(
        '--negative-only',
        action='store_true',
        help='Download only negative (safe) categories'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Determine categories to download
    if args.categories:
        categories = args.categories
    elif args.positive_only:
        categories = POSITIVE_CATEGORIES
    elif args.negative_only:
        categories = NEGATIVE_CATEGORIES
    else:
        categories = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("QuickDraw Dataset Downloader for DoodleParty")
        print("=" * 60)
        print(f"\nCategories to download: {len(categories)}")
        print(f"Output directory: {args.output_dir}")
        print(f"Total size estimate: ~{len(categories) * 25}MB\n")
    
    if args.dry_run:
        if verbose:
            print("DRY RUN - Categories to download:")
            for cat in categories:
                print(f"  - {cat}")
        return 0
    
    # Download categories
    successful = 0
    failed = 0
    
    for category in categories:
        if download_category(category, args.output_dir, verbose):
            successful += 1
        else:
            failed += 1
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Download complete: {successful} successful, {failed} failed")
        
        if successful > 0:
            print(f"\nDataset location: {os.path.abspath(args.output_dir)}")
            # List files
            npy_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
            print(f"Files downloaded: {len(npy_files)}")
            for f in npy_files[:5]:
                print(f"  - {f}")
            if len(npy_files) > 5:
                print(f"  ... and {len(npy_files) - 5} more")
        
        print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
