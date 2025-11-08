"""
Download QuickDraw dataset from Google Cloud Storage (GCS).
Uses the publicly available raw dataset from:
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/raw
"""

import os
import urllib.request
import urllib.error
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw
import socket

# GCS public bucket URL - raw ndjson files
GCS_BUCKET = 'https://storage.googleapis.com/quickdraw_dataset/full/raw/'

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


def strokes_to_bitmap(drawing, size=28):
    """Convert QuickDraw stroke data to 28x28 bitmap."""
    try:
        if not drawing:
            return None
        
        img = Image.new('L', (256, 256), color=255)
        draw = ImageDraw.Draw(img)
        
        for stroke in drawing:
            if not stroke or len(stroke) < 2:
                continue
            
            xs = stroke[0]
            ys = stroke[1]
            
            if not xs or not ys:
                continue
            
            n = min(len(xs), len(ys))
            points = [(int(xs[i]), int(ys[i])) for i in range(n)]
            
            if len(points) > 1:
                draw.line(points, fill=0, width=2)
        
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        bitmap = np.array(img)
        
        return bitmap
    except Exception:
        return None


def download_category(category, max_samples=5000):
    """
    Download a QuickDraw category from GCS.
    
    Args:
        category: Category name (e.g., "airplane")
        max_samples: Max samples to load
    
    Returns:
        Array of 28x28 images
    """
    output_file = Path(f"/tmp/{category}.ndjson")
    
    url = f"{GCS_BUCKET}{category}.ndjson"
    print(f"Downloading {category} from GCS...")
    print(f"  URL: {url}")
    
    try:
        # Create request with User-Agent header
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)')
        
        # Set socket timeout
        socket.setdefaulttimeout(60)
        
        print(f"  Connecting...")
        with urllib.request.urlopen(request) as response:
            total_size = response.headers.get('content-length')
            if total_size:
                total_size = int(total_size) / 1024 / 1024
                print(f"  File size: {total_size:.1f}MB")
            
            with open(output_file, 'wb') as out_file:
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                        print(f"  Downloaded: {downloaded / 1024 / 1024:.1f}MB")
        
        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"  ✓ Downloaded {file_size:.1f}MB")
        
    except urllib.error.HTTPError as e:
        print(f"  ✗ HTTP Error {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return None
    
    if not output_file.exists():
        return None
    
    # Parse NDJSON and convert to images
    print(f"  Processing drawings...")
    images = []
    errors = 0
    
    try:
        with open(output_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    drawing = data.get('drawing', [])
                    
                    image = strokes_to_bitmap(drawing)
                    if image is not None:
                        images.append(image)
                    else:
                        errors += 1
                    
                    if (idx + 1) % 5000 == 0:
                        print(f"    Processed {idx + 1} samples...")
                        
                except Exception as e:
                    errors += 1
                    continue
        
        print(f"  ✓ Loaded {len(images)} samples ({errors} errors)")
        
    except Exception as e:
        print(f"  ✗ Error processing file: {str(e)}")
        return None
    finally:
        # Clean up
        if output_file.exists():
            output_file.unlink()
    
    if len(images) == 0:
        return None
    
    return np.array(images, dtype=np.uint8)


def main():
    print("=" * 70)
    print('DOWNLOADING QUICKDRAW DATASET FROM GOOGLE CLOUD STORAGE')
    print("=" * 70)
    print()
    
    output_dir = Path('/home/mcvaj/ML/data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path('/home/mcvaj/ML/data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = {}
    total_samples = 0
    
    for category in CATEGORIES:
        print(f"\n[{CATEGORIES.index(category) + 1}/{len(CATEGORIES)}] {category.upper()}")
        print("-" * 70)
        
        images = download_category(category, max_samples=10000)
        
        if images is not None and len(images) > 0:
            all_images[category] = images
            total_samples += len(images)
            
            # Save individual category file
            output_file = output_dir / f'{category}.npy'
            np.save(output_file, images)
            print(f"  ✓ Saved to {output_file}\n")
        else:
            print(f"  ✗ Failed to download {category}\n")
    
    print("\n" + "=" * 70)
    print('DOWNLOAD SUMMARY')
    print("=" * 70)
    print(f"Downloaded {len(all_images)} categories")
    print(f"Total samples: {total_samples}")
    
    for category, images in sorted(all_images.items()):
        print(f"  {category}: {len(images)} samples")
    
    print("\n✓ All files saved to:", output_dir)
    
    return all_images


if __name__ == '__main__':
    all_images = main()
