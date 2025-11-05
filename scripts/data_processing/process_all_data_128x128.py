"""
Process ALL data (penis and QuickDraw) to 128x128 with consistent formatting.
Ensures:
- Same 128x128 output size
- Same stroke width (12px on 256x256 canvas)
- Same colors (black background, white strokes)
- Proper bounding box fitting with padding
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def raw_strokes_to_bitmap_128(drawing, size=128):
    """
    Convert raw NDJSON strokes to 128x128 bitmap with consistent processing.
    
    Args:
        drawing: List of strokes in QuickDraw format
                 Each stroke is [x_array, y_array, timestamps_array]
        size: Output size (default 128x128)
    
    Returns:
        128x128 bitmap as numpy array (0-255, uint8)
    """
    try:
        if not drawing:
            return None

        # Step 1: Collect all coordinates to find bounding box
        all_xs = []
        all_ys = []
        stroke_points = []
        
        for stroke in drawing:
            if not stroke or len(stroke) < 2:
                continue
            
            xs = stroke[0]
            ys = stroke[1]
            
            if not xs or not ys or len(xs) == 0 or len(ys) == 0:
                continue
            
            all_xs.extend(xs)
            all_ys.extend(ys)
            
            n = min(len(xs), len(ys))
            points = [(xs[i], ys[i]) for i in range(n)]
            stroke_points.append(points)
        
        if not all_xs or not all_ys:
            return None
        
        # Step 2: Calculate bounding box
        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 and height == 0:
            width = height = 1
        elif width == 0:
            width = 1
        elif height == 0:
            height = 1
        
        # Step 3: Set up canvas and scaling
        canvas_size = 256
        padding = 20
        scale = (canvas_size - 2 * padding) / max(width, height)
        
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Step 4: Create black canvas (0 = black background)
        img = Image.new('L', (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Step 5: Draw each stroke centered on canvas
        for points in stroke_points:
            if len(points) < 2:
                continue
            
            normalized_points = []
            for x, y in points:
                nx = (x - min_x) * scale
                ny = (y - min_y) * scale
                
                center_offset_x = padding + (canvas_size - 2 * padding - scaled_width) / 2
                center_offset_y = padding + (canvas_size - 2 * padding - scaled_height) / 2
                
                nx += center_offset_x
                ny += center_offset_y
                
                normalized_points.append((int(nx), int(ny)))
            
            # Draw with consistent width - 12px for 128x128 output (scales from 3px for 28x28)
            if len(normalized_points) > 1:
                draw.line(normalized_points, fill=255, width=12)
        
        # Step 6: Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Step 7: Convert to numpy array
        bitmap = np.array(img, dtype=np.uint8)
        
        return bitmap

    except Exception as e:
        return None


def process_ndjson_to_128(filepath, max_samples=None):
    """Process NDJSON file to 128x128 bitmaps."""
    images = []
    errors = 0
    processed = 0
    
    print(f"Processing {filepath.name}...")
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                data = json.loads(line)
                drawing = data.get('drawing', [])
                image = raw_strokes_to_bitmap_128(drawing, size=128)
                
                if image is not None:
                    images.append(image)
                    processed += 1
                else:
                    errors += 1
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1} lines ({processed} success, {errors} errors)...")
                    
            except Exception as e:
                errors += 1
                continue
    
    print(f"  ✓ Successfully processed {processed} samples ({errors} errors skipped)")
    return np.array(images, dtype=np.uint8)


def process_quickdraw_ndjson_to_128(filepath, max_samples=1200):
    """
    Process QuickDraw NDJSON file to 128x128 by rendering from strokes.
    This produces crisp images, not blurry upscaled ones.
    """
    return process_ndjson_to_128(filepath, max_samples=max_samples)


def main():
    """Process all data to 128x128."""
    print("="*70)
    print("PROCESSING ALL DATA TO 128x128")
    print("="*70)
    
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    appendix_dir = project_root / 'quickdraw_appendix'
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process penis data from NDJSON
    print("\n" + "="*70)
    print("STEP 1: Processing Penis Data (NDJSON → 128x128)")
    print("="*70)
    
    ndjson_file = appendix_dir / 'penis-raw.ndjson'
    if ndjson_file.exists():
        penis_data = process_ndjson_to_128(ndjson_file)
        output_file = processed_dir / 'penis_raw_X_128.npy'
        np.save(output_file, penis_data)
        print(f"\n✓ Saved {len(penis_data):,} penis samples to {output_file}")
        print(f"  Shape: {penis_data.shape}")
        print(f"  Range: {penis_data.min()}-{penis_data.max()}, mean: {penis_data.mean():.2f}")
    else:
        print(f"ERROR: {ndjson_file} not found!")
        return
    
    # Step 2: Process QuickDraw negative classes from NDJSON (raw strokes → 128x128)
    print("\n" + "="*70)
    print("STEP 2: Processing QuickDraw Negative Classes (NDJSON → 128x128)")
    print("="*70)
    
    negative_classes = [
        'airplane', 'apple', 'arm', 'banana', 'bird', 'boomerang',
        'cat', 'circle', 'cloud', 'dog', 'drill', 'fish', 'flower',
        'house', 'moon', 'pencil', 'square', 'star', 'sun', 'tree', 'triangle'
    ]
    
    ndjson_dir = project_root / 'data' / 'raw_ndjson'
    
    for cls in negative_classes:
        # Try to load from NDJSON first (for crisp 128x128 rendering)
        ndjson_file = ndjson_dir / f'{cls}-raw.ndjson'
        
        if ndjson_file.exists():
            data = process_quickdraw_ndjson_to_128(ndjson_file, max_samples=1200)
            
            if len(data) > 0:
                output_file = processed_dir / f'{cls}_128.npy'
                np.save(output_file, data)
                print(f"  ✓ {cls}: {len(data):,} samples → {output_file.name} (rendered from strokes)")
                print(f"     Range: {data.min()}-{data.max()}, mean: {data.mean():.2f}")
            else:
                print(f"  ⚠️  {cls}: No valid samples found!")
        else:
            print(f"  ⚠️  {cls}: NDJSON file not found at {ndjson_file}")
            print(f"      Run: python scripts/data_processing/download_quickdraw_ndjson.py")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ All data processed to 128x128 with consistent formatting:")
    print("  • Black background (pixel value 0)")
    print("  • White strokes (pixel value 255)")
    print("  • Consistent stroke width (12px on 256x256 canvas)")
    print("  • Proper bounding box fitting with 20px padding")
    print("  • Rendered directly from vector strokes (NO BLURRINESS!)")
    print("  • High-quality LANCZOS resampling")
    print(f"\nProcessed files saved to: {processed_dir}")
    print("\nNext step: Run training data regeneration")
    print("  python scripts/data_processing/regenerate_training_data.py")


if __name__ == "__main__":
    main()
