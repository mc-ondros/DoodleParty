"""
Download QuickDraw categories from Google's official dataset.
"""

import os
import urllib.request
import urllib.error
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw
import socket

# List of QuickDraw categories to download as negatives
CATEGORIES = [
    "house",
    "tree", 
    "cat",
    "dog",
    "flower",
    "star",
    "circle",
    "square",
    "triangle",
    "cloud",
    "sun",
    "moon",
    "bird",
    "fish",
    "apple",
    "arm",
    "banana",
    "boomerang",
    "drill",
    "pencil"
]

# Try multiple mirrors - updated with correct paths
BASE_URLS = [
    "https://quickdraw.google.com/data/full/simplified/",          # Official source
    "https://storage.googleapis.com/quickdraw_dataset/full/ndjson/",  # GCS with ndjson
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


def download_category(category, max_samples=2000):
    """
    Download a QuickDraw category from available mirrors.
    
    Args:
        category: Category name (e.g., "house")
        max_samples: Max samples to load (Google provides ~100K per category)
    
    Returns:
        Array of 28x28 images
    """
    output_file = Path(f"/tmp/{category}.ndjson")
    
    # Try each mirror until one works
    for base_url in BASE_URLS:
        url = f"{base_url}{category}.ndjson"
        print(f"Downloading {category} from {base_url}...")
        
        try:
            # Create request with timeout
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            
            # Set socket timeout globally for this request
            socket.setdefaulttimeout(30)
            with urllib.request.urlopen(request) as response:
                with open(output_file, 'wb') as out_file:
                    out_file.write(response.read())
            
            file_size = output_file.stat().st_size / 1024 / 1024
            print(f"  ✓ Downloaded {file_size:.1f}MB")
            break
        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP Error {e.code}: {e.reason}")
            continue
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:80]}")
            continue
    else:
        # All mirrors failed
        print(f"  ✗ Could not download {category} from any mirror")
        return None
    
    if not output_file.exists():
        return None
    
    # Parse NDJSON
    images = []
    errors = 0
    
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
                
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1} samples...")
                    
            except Exception as e:
                errors += 1
                continue
    
    print(f"  ✓ Loaded {len(images)} samples ({errors} errors)")
    
    # Clean up
    output_file.unlink()
    
    return np.array(images, dtype=np.uint8)


def main():
    print("=" * 60)
    print("DOWNLOADING QUICKDRAW CATEGORIES FOR NEGATIVE SAMPLES")
    print("=" * 60)
    
    output_dir = Path('/home/mcvaj/ML/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_negatives = []
    
    for category in CATEGORIES:
        images = download_category(category, max_samples=2000)
        if images is not None and len(images) > 0:
            all_negatives.append(images)
            print(f"  Total so far: {sum(len(img_array) for img_array in all_negatives)} samples\n")
    
    if not all_negatives:
        print("✗ Failed to download any categories!")
        return
    
    # Combine all negatives
    X_negative_real = np.concatenate(all_negatives, axis=0)
    print(f"\n✓ Total real negative samples: {len(X_negative_real)}")
    
    # Save
    np.save(output_dir / 'quickdraw_negatives.npy', X_negative_real)
    print(f"✓ Saved to {output_dir / 'quickdraw_negatives.npy'}")
    
    return X_negative_real


if __name__ == "__main__":
    X_negative_real = main()
