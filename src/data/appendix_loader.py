"""
Load and process QuickDraw Appendix dataset (NDJSON format).

Converts stroke data to 28x28 bitmap images with proper normalization
and centering. Supports both raw and simplified QuickDraw formats.

Related:
- scripts/data_processing/download_quickdraw_ndjson.py (data download)
- scripts/data_processing/process_all_data_128x128.py (128x128 processing)
- src/data/loaders.py (general data loading)

Exports:
- parse_ndjson_file, strokes_to_bitmap, raw_strokes_to_bitmap
- simplified_strokes_to_bitmap, discover_appendix_files
- load_appendix_category, prepare_multi_class_dataset, prepare_appendix_dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from PIL import Image, ImageDraw


def parse_ndjson_file(filepath, max_samples=None, format_type='raw'):
    """
    Parse NDJSON file and convert drawing strokes to 28x28 bitmaps.
    
    Args:
        filepath: Path to NDJSON file
        max_samples: Maximum samples to load (None for all)
        format_type: 'raw' or 'simplified' format
    
    Returns:
        images: Array of 28x28 bitmap images (0-255)
    """
    images = []
    errors = 0
    processed = 0
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                data = json.loads(line)
                drawing = data.get('drawing', [])
                
                # Convert stroke data to bitmap based on format
                if format_type == 'simplified':
                    # Simplified format: drawing is [x_coords, y_coords, ...]
                    image = simplified_strokes_to_bitmap(drawing)
                else:
                    # Raw format: drawing is [[x, y], [x, y], ...]
                    image = raw_strokes_to_bitmap(drawing)
                
                if image is not None:
                    images.append(image)
                    processed += 1
                else:
                    errors += 1
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1} samples ({processed} success, {errors} errors)...")
                    
            except Exception as e:
                errors += 1
                continue
    
    print(f"  Successfully loaded {processed} samples ({errors} errors skipped)")
    return np.array(images, dtype=np.uint8)

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from PIL import Image, ImageDraw
from glob import glob
import os


def strokes_to_bitmap(strokes, size=28):
    """
    Convert stroke coordinates to 28x28 bitmap image.
    
    Args:
        strokes: List of strokes, each stroke is list of [x, y] coordinates
        size: Output image size (28x28)
    
    Returns:
        Bitmap image as numpy array (28x28)
    """
    try:
        if not strokes or len(strokes) == 0:
            return None
        
        # Create image
        img = Image.new('L', (256, 256), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Draw each stroke
        for stroke in strokes:
            if not stroke or len(stroke) == 0:
                continue
            
            # Handle both formats: [[x1, y1], [x2, y2], ...] or separate x and y lists
            if isinstance(stroke[0], (list, tuple)):
                # Format: [[x1, y1], [x2, y2], ...]
                points = [(int(stroke[i][0]), int(stroke[i][1])) for i in range(len(stroke))]
            else:
                # This shouldn't happen in standard QuickDraw format
                continue
            
            if len(points) > 1:
                draw.line(points, fill=0, width=2)  # Black lines
        
        # Resize to 28x28
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        bitmap = np.array(img)
        
        return bitmap
        
    except Exception as e:
        # Silently skip problematic drawings
        return None


def raw_strokes_to_bitmap(drawing, size=28):
    """
    Converter for QuickDraw official/raw format.
    
    The QuickDraw Appendix uses the official QuickDraw format where:
    - drawing is a list of strokes
    - each stroke is [x_coords_array, y_coords_array, timestamps_array]
    
    This function normalizes, centers, and scales the drawing properly.
    """
    try:
        if not drawing:
            return None

        # First pass: collect all coordinates to find bounding box
        all_xs = []
        all_ys = []
        stroke_points = []  # Store processed strokes
        
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
            
            # Store stroke for later
            n = min(len(xs), len(ys))
            points = [(xs[i], ys[i]) for i in range(n)]
            stroke_points.append(points)
        
        if not all_xs or not all_ys:
            return None
        
        # Calculate bounding box
        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        
        # Calculate dimensions and scaling
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return None
        
        # Create canvas with padding
        canvas_size = 256
        padding = 20  # Add padding to avoid edge clipping
        scale = (canvas_size - 2 * padding) / max(width, height)
        
        # Create image canvas
        img = Image.new('L', (canvas_size, canvas_size), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Draw each stroke with normalization
        for points in stroke_points:
            if len(points) < 2:
                continue
            
            # Normalize and center coordinates
            normalized_points = []
            for x, y in points:
                # Translate to origin
                nx = (x - min_x) * scale
                ny = (y - min_y) * scale
                
                # Center on canvas
                nx += padding + (canvas_size - 2 * padding - width * scale) / 2
                ny += padding + (canvas_size - 2 * padding - height * scale) / 2
                
                normalized_points.append((int(nx), int(ny)))
            
            # Draw the stroke
            if len(normalized_points) > 1:
                draw.line(normalized_points, fill=0, width=3)  # Black lines

        # Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        bitmap = np.array(img)
        
        return bitmap

    except Exception as e:
        return None


def simplified_strokes_to_bitmap(drawing, size=28):
    """
    Converter for 'simplified' drawing format.
    
    Same as raw format in this codebase since QuickDraw official format
    is already in the simplified form [x_array, y_array, timestamps].
    Uses the same normalization and centering as raw format.
    """
    # Use the same logic as raw format with proper normalization
    return raw_strokes_to_bitmap(drawing, size)


# NOTE: The format-aware `parse_ndjson_file(filepath, max_samples=None, format_type='raw')`
# is defined earlier in this file. We intentionally do not redefine it here to avoid
# accidentally shadowing the format-aware implementation.


def discover_appendix_files(appendix_dir):
    """
    Discover all NDJSON files in the QuickDraw Appendix directory.
    
    Args:
        appendix_dir: Path to quickdraw_appendix directory
    
    Returns:
        Dictionary with category -> list of file paths
    """
    appendix_dir = Path(appendix_dir)
    categories = {}
    
    # Look for NDJSON files
    ndjson_files = list(appendix_dir.glob("*.ndjson"))
    
    print(f"Found {len(ndjson_files)} NDJSON files in {appendix_dir}:")
    
    for filepath in sorted(ndjson_files):
        filename = filepath.stem
        # Extract category from filename (e.g., "penis-raw" -> "penis")
        if '-' in filename:
            category, variant = filename.rsplit('-', 1)
        else:
            category = filename
            variant = 'default'
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            'path': filepath,
            'variant': variant,
            'filename': filepath.name
        })
        print(f"  ✓ {filepath.name} ({category} - {variant})")
    
    return categories


def load_appendix_category(filepath, max_samples=None, prefer_variant = 'raw'):
    """
    Load a single category from NDJSON file.
    
    Args:
        filepath: Path to NDJSON file
        max_samples: Maximum samples to load
        prefer_variant: Prefer 'raw' or 'simplified' (raw has more detail)
    
    Returns:
        images: Array of 28x28 bitmap images
        category: Category name extracted from filename
    """
    filepath = Path(filepath)
    category = filepath.stem.split('-')[0]
    
    print(f"Loading {category} from {filepath.name}...")
    # Infer file format from filename (e.g., 'penis-simplified.ndjson' vs 'penis-raw.ndjson')
    stem = filepath.stem.lower()
    if stem.endswith('-simplified') or '-simplified' in stem:
        format_type = 'simplified'
    else:
        # default to raw for '-raw' or unknown variants
        format_type = 'raw'

    images = parse_ndjson_file(filepath, max_samples=max_samples, format_type=format_type)
    
    return images, category


def prepare_multi_class_dataset(appendix_dir, output_dir = 'data/processed', 
                                max_samples_per_class=None, test_split=0.2,
                                negative_class_type = 'noise'):
    """
    Prepare multi-class dataset from entire QuickDraw Appendix library.
    
    Creates a dataset where:
    - Positive class: All real drawings from appendix
    - Negative class: Either random noise or other QuickDraw data
    
    Args:
        appendix_dir: Path to quickdraw_appendix directory
        output_dir: Directory to save processed data
        max_samples_per_class: Max samples per category (None for all)
        test_split: Test set fraction
        negative_class_type: "noise" (random) or "other" (use similar data)
    """
    appendix_dir = Path(appendix_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover available files
    categories = discover_appendix_files(appendix_dir)
    
    if not categories:
        print('No NDJSON files found in appendix directory!')
        return None
    
    print(f"\nLoading {len(categories)} categories...")
    
    all_images = []
    all_labels = []
    category_to_idx = {}
    idx = 0
    
    # Load all positive examples (actual drawings)
    print('\nLoading Positive Examples (Appendix Drawings)')
    total_positive = 0
    
    for category, files in sorted(categories.items()):
        # Prefer raw over simplified
        best_file = None
        for f in files:
            if f['variant'] == 'raw':
                best_file = f
                break
        if best_file is None and files:
            best_file = files[0]
        
        if best_file is None:
            continue
        
        try:
            images, cat_name = load_appendix_category(
                best_file['path'],
                max_samples=max_samples_per_class,
                prefer_variant = 'raw'
            )
            
            if len(images) > 0:
                labels = np.ones(len(images), dtype=np.int32)
                all_images.append(images.astype(np.float32) / 255.0)
                all_labels.append(labels)
                category_to_idx[cat_name] = idx
                idx += 1
                total_positive += len(images)
                print(f"  ✓ {cat_name}: {len(images)} positive samples")
            
        except Exception as e:
            print(f"  ✗ Error loading {category}: {e}")
            continue
    
    print(f"\nTotal positive samples: {total_positive}")
    
    # Generate negative examples
    print('\nGenerating Negative Examples')
    if negative_class_type == "noise":
        print(f"Generating {total_positive} random noise samples...")
        negative_images = np.random.randint(0, 256, (total_positive, 28, 28), dtype=np.uint8)
        negative_labels = np.zeros(total_positive, dtype=np.int32)
        all_images.append(negative_images.astype(np.float32) / 255.0)
        all_labels.append(negative_labels)
    
    print(f"✓ Generated {total_positive} negative samples")
    
    # Combine and prepare
    print('\nPreparing Dataset')
    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Reshape and normalize
    X = X.reshape(-1, 28, 28, 1).astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    total_samples = len(X)
    test_size = int(total_samples * test_split)
    train_size = total_samples - test_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Save
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    # Save class mapping
    class_mapping = {
        'negative': 0,
        'positive': 1,
        'categories': category_to_idx,
        'description': f'Binary classification: positive={len(category_to_idx)} appendix categories, negative=random noise'
    }
    with open(output_dir / "class_mapping.pkl", 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"\n✓ Multi-class dataset prepared successfully!")
    print(f"  Categories: {len(category_to_idx)} ({', '.join(sorted(category_to_idx.keys()))})")
    print(f"  Training samples: {len(X_train)} (positive: {(y_train==1).sum()}, negative: {(y_train==0).sum()})")
    print(f"  Test samples: {len(X_test)} (positive: {(y_test==1).sum()}, negative: {(y_test==0).sum()})")
    print(f"  Saved to: {output_dir}")
    
    return (X_train, y_train), (X_test, y_test), class_mapping


def prepare_appendix_dataset(ndjson_file, output_dir = 'data/processed', 
                             max_samples_per_class=2000, test_split=0.2):
    """
    Prepare binary classification dataset from QuickDraw Appendix.
    
    Positive class: Drawings from the NDJSON file
    Negative class: Random noise (synthetic)
    
    Args:
        ndjson_file: Path to NDJSON file
        output_dir: Directory to save processed data
        max_samples_per_class: Max samples for positive class
        test_split: Test set fraction
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading QuickDraw Appendix dataset from {ndjson_file}...")
    
    # Load positive examples (actual drawings)
    positive_images = parse_ndjson_file(ndjson_file, max_samples=max_samples_per_class)
    positive_labels = np.ones(len(positive_images), dtype=np.int32)
    
    print(f"✓ Loaded {len(positive_images)} positive samples (actual drawings)")
    
    # Generate negative examples (random noise)
    negative_images = np.random.randint(0, 256, (len(positive_images), 28, 28), dtype=np.uint8)
    negative_labels = np.zeros(len(positive_images), dtype=np.int32)
    
    print(f"✓ Generated {len(negative_images)} negative samples (random noise)")
    
    # Combine and normalize
    X = np.concatenate([positive_images, negative_images], axis=0)
    y = np.concatenate([positive_labels, negative_labels], axis=0)
    
    # Reshape and normalize
    X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split train/test
    total_samples = len(X)
    test_size = int(total_samples * test_split)
    train_size = total_samples - test_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Save
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    # Save class mapping
    class_mapping = {
        'negative': 0,
        'positive': 1,
        'description': 'Binary classification: positive=appendix drawings, negative=random noise'
    }
    with open(output_dir / "class_mapping.pkl", 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"\n✓ Binary dataset prepared successfully!")
    print(f"  Training samples: {len(X_train)} (positive: {(y_train==1).sum()}, negative: {(y_train==0).sum()})")
    print(f"  Test samples: {len(X_test)} (positive: {(y_test==1).sum()}, negative: {(y_test==0).sum()})")
    print(f"  Saved to: {output_dir}")
    
    return (X_train, y_train), (X_test, y_test), class_mapping


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Process QuickDraw Appendix dataset')
    parser.add_argument("--input", help = 'Path to single NDJSON file (for single-file mode)')
    parser.add_argument("--appendix-dir", help = 'Path to QuickDraw Appendix directory (for multi-file mode)')
    parser.add_argument("--output-dir", default = 'data/processed', help = 'Output directory')
    parser.add_argument("--max-samples", type=int, default=None, help = 'Max samples per class')
    parser.add_argument("--test-split", type=float, default=0.2, help = 'Test set fraction')
    
    args = parser.parse_args()
    
    if args.input:
        # Single file mode
        prepare_appendix_dataset(
            args.input,
            output_dir=args.output_dir,
            max_samples_per_class=args.max_samples or 2000,
            test_split=args.test_split
        )
    elif args.appendix_dir:
        # Multi-file mode (entire library)
        prepare_multi_class_dataset(
            args.appendix_dir,
            output_dir=args.output_dir,
            max_samples_per_class=args.max_samples,
            test_split=args.test_split
        )
    else:
        parser.print_help()
        print('\nExample usage:')
        print('  Single file:  python appendix_loader.py --input penis-raw.ndjson')
        print('  All files:    python appendix_loader.py --appendix-dir /path/to/quickdraw_appendix')
