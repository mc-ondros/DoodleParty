"""
Reprocess penis NDJSON data to match QuickDraw negative samples exactly.
Ensures:
1. Same stroke width as negative samples
2. Proper bounding box fitting (centered with padding)
3. Same 28x28 output format
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def raw_strokes_to_bitmap_consistent(drawing, size=128):
    """
    Convert raw NDJSON strokes to bitmap with consistent processing.
    
    This ensures the penis drawings match QuickDraw negative samples by:
    - Using same canvas size and padding
    - Consistent stroke width (12 pixels on 256x256 canvas for 128x128 output)
    - Proper centering and scaling to fit bounding box
    - Same resize algorithm (LANCZOS)
    
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
            
            # stroke = [x_array, y_array, timestamps_array]
            xs = stroke[0]
            ys = stroke[1]
            
            if not xs or not ys or len(xs) == 0 or len(ys) == 0:
                continue
            
            # Collect all coordinates
            all_xs.extend(xs)
            all_ys.extend(ys)
            
            # Store stroke points for drawing
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
            # Single point drawing - give it minimal size
            width = height = 1
        elif width == 0:
            width = 1
        elif height == 0:
            height = 1
        
        # Step 3: Set up canvas and scaling
        # Use larger canvas for higher resolution output
        canvas_size = 512  # Increased from 256 for better quality at 128x128
        padding = 40  # Scaled padding (was 20 for 256 canvas)
        
        # Scale to fit in canvas with padding
        # Use the maximum dimension to maintain aspect ratio
        scale = (canvas_size - 2 * padding) / max(width, height)
        
        # Calculate scaled dimensions
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Step 4: Create black canvas (0 = black background) to match QuickDraw format
        # Most QuickDraw classes have black background with white strokes
        img = Image.new('L', (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Step 5: Draw each stroke centered on canvas
        for points in stroke_points:
            if len(points) < 2:
                continue
            
            # Normalize and center coordinates
            normalized_points = []
            for x, y in points:
                # Translate to origin (0, 0)
                nx = (x - min_x) * scale
                ny = (y - min_y) * scale
                
                # Center on canvas
                # Add padding, then center the scaled drawing
                center_offset_x = padding + (canvas_size - 2 * padding - scaled_width) / 2
                center_offset_y = padding + (canvas_size - 2 * padding - scaled_height) / 2
                
                nx += center_offset_x
                ny += center_offset_y
                
                normalized_points.append((int(nx), int(ny)))
            
                        # Draw the stroke with consistent width
            # Width=12 on 256x256 canvas for 128x128 output (scales proportionally from 3px for 28x28)
            # Use WHITE strokes (255) to match QuickDraw format
            if len(normalized_points) > 1:
                draw.line(normalized_points, fill=255, width=12)  # White lines on black background

        # Step 6: Resize to target size (28x28)
        # Using LANCZOS for high-quality downsampling
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Step 7: Convert to numpy array
        bitmap = np.array(img, dtype=np.uint8)
        
        return bitmap

    except Exception as e:
        print(f"Error processing drawing: {e}")
        return None


def parse_ndjson_file_consistent(filepath, max_samples=None, size=128):
    """
    Parse NDJSON file with consistent processing matching QuickDraw standards.
    
    Args:
        filepath: Path to NDJSON file
        max_samples: Maximum samples to load (None for all)
        size: Output image size (default 128x128)
    
    Returns:
        images: Array of 128x128 bitmap images (0-255, uint8)
    """
    images = []
    errors = 0
    processed = 0
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                data = json.loads(line)
                drawing = data.get('drawing', [])
                
                # Convert stroke data to bitmap
                image = raw_strokes_to_bitmap_consistent(drawing, size=size)
                
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


def main():
    """Main processing function."""
    print("="*70)
    print('REPROCESSING PENIS NDJSON DATA (128x128)')
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    appendix_dir = project_root / 'quickdraw_appendix'
    ndjson_file = appendix_dir / 'penis-raw.ndjson'
    output_dir = project_root / 'data' / 'processed'
    output_file = output_dir / 'penis_raw_X.npy'
    
    # Check input file
    if not ndjson_file.exists():
        print(f"ERROR: Input file not found: {ndjson_file}")
        return
    
    print(f"\nInput:  {ndjson_file}")
    print(f"Output: {output_file}")
    
    # Process the data
    print("\n" + "="*70)
    print('STEP 1: Converting NDJSON to 128x128 bitmaps')
    print("="*70)
    images = parse_ndjson_file_consistent(ndjson_file, max_samples=None, size=128)
    
    if len(images) == 0:
        print('ERROR: No images processed successfully!')
        return
    
    print(f"\n✓ Processed {len(images):,} images")
    print(f"  Shape: {images.shape}")
    print(f"  Dtype: {images.dtype}")
    print(f"  Value range: {images.min()} - {images.max()}")
    print(f"  Mean: {images.mean():.2f}")
    
    # Verify consistency with negative samples
    print("\n" + "="*70)
    print('STEP 2: Verifying consistency with negative samples')
    print("="*70)
    
    # Load a sample negative class for comparison
    neg_file = project_root / 'data' / 'raw' / 'airplane.npy'
    if neg_file.exists():
        neg_samples = np.load(neg_file)[:100]
        print(f"\nNegative samples (airplane):")
        print(f"  Shape: {neg_samples.shape}")
        print(f"  Dtype: {neg_samples.dtype}")
        print(f"  Value range: {neg_samples.min()} - {neg_samples.max()}")
        print(f"  Mean: {neg_samples.mean():.2f}")
        
        print(f"\nProcessed penis samples:")
        print(f"  Shape: {images.shape}")
        print(f"  Dtype: {images.dtype}")
        print(f"  Value range: {images.min()} - {images.max()}")
        print(f"  Mean: {images.mean():.2f}")
        
        # Check if formats match
        if images.shape[1:] == neg_samples.shape[1:]:
            print('\n✓ Dimensions match!')
        else:
            print(f"\n⚠️  Dimension mismatch: {images.shape[1:]} vs {neg_samples.shape[1:]}")
        
        if images.dtype == neg_samples.dtype:
            print('✓ Data types match!')
        else:
            print(f"⚠️  Data type mismatch: {images.dtype} vs {neg_samples.dtype}")
    
    # Save the processed data
    print("\n" + "="*70)
    print('STEP 3: Saving processed data')
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_file, images)
    
    print(f"\n✓ Saved {len(images):,} images to {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Summary
    print("\n" + "="*70)
    print('SUMMARY')
    print("="*70)
    print(f"✓ Successfully processed {len(images):,} penis drawings")
    print(f"✓ Output format: {images.shape} ({images.dtype})")
    print(f"✓ Value range: {images.min()}-{images.max()} (mean: {images.mean():.2f})")
    print(f"✓ Saved to: {output_file}")
    print('\nThe data is now ready for training!')
    print('Run: python scripts/data_processing/regenerate_training_data.py')


if __name__ == '__main__':
    main()
