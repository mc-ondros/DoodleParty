"""
Download QuickDraw dataset from Google Cloud Storage (numpy bitmap format).
Uses pre-processed 28x28 grayscale images - much smaller than ndjson format.

Source: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
"""

import os
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# GCS bucket with numpy bitmap format (pre-processed, ~100-200MB per category)
GCS_BUCKET = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap'

# List of QuickDraw categories to download
CATEGORIES = [
    "airplane",
    "apple",
    "banana",
    "cat",
    "dog",
    "flower",
    "house",
    "tree",
    "star",
    "circle",
    "square",
    "triangle",
    "cloud",
    "sun",
    "moon",
    "bird",
    "fish",
    "arm",
    "boomerang",
    "drill",
    "pencil",
]


def download_category(category, output_dir):
    """
    Download a QuickDraw category as numpy bitmap (28x28 grayscale images).
    
    Args:
        category: Category name (e.g., "airplane")
        output_dir: Directory to save the .npy file
    
    Returns:
        Path to downloaded file or None if failed
    """
    output_file = output_dir / f"{category}.npy"
    
    # Skip if already downloaded
    if output_file.exists():
        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"  ✓ Already downloaded ({file_size:.1f}MB)")
        return output_file
    
    url = f"{GCS_BUCKET}/{category}.npy"
    
    try:
        print(f"  Connecting to {url}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / 1024 / 1024
        
        print(f"  Downloading {total_size_mb:.1f}MB...")
        
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     unit_divisor=1024, desc=category, leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify downloaded file
        if output_file.exists():
            try:
                data = np.load(output_file)
                print(f"  ✓ Downloaded: {data.shape[0]} images ({total_size_mb:.1f}MB)")
                return output_file
            except Exception as e:
                print(f"  ✗ File corrupted: {str(e)}")
                output_file.unlink()
                return None
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  ✗ Category not found (404)")
        else:
            print(f"  ✗ HTTP Error {e.response.status_code}")
        return None
    except requests.exceptions.Timeout:
        print(f"  ✗ Download timeout")
        if output_file.exists():
            output_file.unlink()
        return None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        if output_file.exists():
            output_file.unlink()
        return None


def main():
    print("=" * 75)
    print('QUICKDRAW NUMPY BITMAP DOWNLOADER')
    print("=" * 75)
    print('Format: 28x28 grayscale images (pre-processed)')
    print('Source: https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap')
    print()
    
    output_dir = Path('/home/mcvaj/ML/data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}\n")
    
    downloaded = []
    failed = []
    
    for idx, category in enumerate(CATEGORIES, 1):
        print(f"[{idx}/{len(CATEGORIES)}] {category.upper()}")
        result = download_category(category, output_dir)
        
        if result:
            downloaded.append(category)
        else:
            failed.append(category)
        print()
    
    # Summary
    print("=" * 75)
    print('DOWNLOAD SUMMARY')
    print("=" * 75)
    print(f"Successfully downloaded: {len(downloaded)}/{len(CATEGORIES)}")
    print(f"Failed: {len(failed)}/{len(CATEGORIES)}")
    
    if downloaded:
        print('\n✓ Downloaded categories:')
        for cat in downloaded:
            filepath = output_dir / f"{cat}.npy"
            if filepath.exists():
                size = filepath.stat().st_size / 1024 / 1024
                data = np.load(filepath)
                print(f"  • {cat}: {data.shape[0]:,} images ({size:.1f}MB)")
    
    if failed:
        print('\n✗ Failed categories:')
        for cat in failed:
            print(f"  • {cat}")
    
    print(f"\n✓ All files saved to: {output_dir}")
    print('\nNext step: Run dataset preparation')
    print("  python src/dataset.py --classes " + " ".join(downloaded[:5]))
    
    return len(downloaded) > 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
