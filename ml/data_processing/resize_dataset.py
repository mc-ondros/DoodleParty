"""
Resize dataset to different resolutions for edge deployment optimization.

Creates downsampled versions of existing 128x128 dataset for testing
different resolution/accuracy trade-offs on resource-constrained devices
like Raspberry Pi 4B.

Usage:
    python scripts/data_processing/resize_dataset.py --target-size 64
    python scripts/data_processing/resize_dataset.py --target-size 96

Related:
- scripts/train.py (training with variable resolutions)
- docs/rpi_deployment_strategy.md (deployment strategy)
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import shutil
from tqdm import tqdm


def resize_images(images, target_size, resampling=Image.Resampling.LANCZOS):
    """
    Resize batch of images to target resolution.
    
    Args:
        images: numpy array of shape (N, H, W, 1) or (N, H, W)
        target_size: target dimension (e.g., 64 for 64x64)
        resampling: PIL resampling method (LANCZOS for best quality)
    
    Returns:
        Resized numpy array of shape (N, target_size, target_size, 1)
    """
    n_samples = len(images)
    resized = np.zeros((n_samples, target_size, target_size, 1), dtype=np.float32)
    
    for i in tqdm(range(n_samples), desc="Resizing images"):
        # Extract single image
        img = images[i]
        
        # Handle different input shapes
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Remove channel dimension if present
        
        # Convert to PIL Image
        # Assume data is already normalized [0, 1]
        img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        
        # Resize with high-quality resampling
        img_resized = img_pil.resize((target_size, target_size), resampling)
        
        # Convert back to normalized numpy array
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Add channel dimension
        resized[i] = img_array.reshape(target_size, target_size, 1)
    
    return resized


def main():
    parser = argparse.ArgumentParser(
        description='Resize dataset to different resolutions for edge deployment'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        required=True,
        choices=[32, 64, 96, 112],
        help='Target image size (will create NxN images)'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='data/processed',
        help='Source directory with 128x128 data'
    )
    parser.add_argument(
        '--resampling',
        type=str,
        default='LANCZOS',
        choices=['LANCZOS', 'BICUBIC', 'BILINEAR'],
        help='Resampling method for resizing'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    target_dir = Path(f'data/processed_{args.target_size}x{args.target_size}')
    
    # Map resampling method
    resampling_map = {
        'LANCZOS': Image.Resampling.LANCZOS,
        'BICUBIC': Image.Resampling.BICUBIC,
        'BILINEAR': Image.Resampling.BILINEAR,
    }
    resampling = resampling_map[args.resampling]
    
    print("="*70)
    print(f"DATASET RESIZING: 128x128 → {args.target_size}x{args.target_size}")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Resampling: {args.resampling}")
    print()
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to process
    data_files = {
        'X_train.npy': 'Training images',
        'y_train.npy': 'Training labels',
        'X_test.npy': 'Test images',
        'y_test.npy': 'Test labels',
    }
    
    # Process each file
    for filename, description in data_files.items():
        source_path = source_dir / filename
        target_path = target_dir / filename
        
        if not source_path.exists():
            print(f"⚠️  Skipping {filename} (not found)")
            continue
        
        print(f"\nProcessing {description} ({filename})...")
        data = np.load(source_path)
        
        print(f"  Original shape: {data.shape}")
        
        # Only resize image data (X), not labels (y)
        if filename.startswith('X_'):
            # Resize images
            resized_data = resize_images(data, args.target_size, resampling)
            print(f"  Resized shape: {resized_data.shape}")
            
            # Verify data range
            print(f"  Data range: [{resized_data.min():.3f}, {resized_data.max():.3f}]")
            print(f"  Mean: {resized_data.mean():.3f}")
        else:
            # Copy labels unchanged
            resized_data = data
            print(f"  Copied unchanged (labels)")
        
        # Save resized data
        np.save(target_path, resized_data)
        print(f"  ✓ Saved to {target_path}")
    
    # Copy class mapping file
    mapping_file = 'class_mapping.pkl'
    source_mapping = source_dir / mapping_file
    target_mapping = target_dir / mapping_file
    
    if source_mapping.exists():
        print(f"\nCopying {mapping_file}...")
        shutil.copy(source_mapping, target_mapping)
        print(f"  ✓ Saved to {target_mapping}")
    
    # Generate statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    X_train = np.load(target_dir / 'X_train.npy')
    y_train = np.load(target_dir / 'y_train.npy')
    X_test = np.load(target_dir / 'X_test.npy')
    y_test = np.load(target_dir / 'y_test.npy')
    
    print(f"\nTraining set:")
    print(f"  Samples: {len(X_train)}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Positive class: {(y_train == 1).sum()} ({100*(y_train==1).mean():.1f}%)")
    print(f"  Negative class: {(y_train == 0).sum()} ({100*(y_train==0).mean():.1f}%)")
    
    print(f"\nTest set:")
    print(f"  Samples: {len(X_test)}")
    print(f"  Shape: {X_test.shape}")
    print(f"  Positive class: {(y_test == 1).sum()} ({100*(y_test==1).mean():.1f}%)")
    print(f"  Negative class: {(y_test == 0).sum()} ({100*(y_test==0).mean():.1f}%)")
    
    # Calculate expected performance impact
    original_pixels = 128 * 128
    new_pixels = args.target_size * args.target_size
    pixel_ratio = new_pixels / original_pixels
    speedup = original_pixels / new_pixels
    
    print(f"\nPerformance implications:")
    print(f"  Pixels: {original_pixels} → {new_pixels} ({pixel_ratio*100:.1f}% of original)")
    print(f"  Expected speedup: ~{speedup:.1f}x")
    print(f"  Expected accuracy loss: ~{(1-pixel_ratio)*5:.1f}% (rough estimate)")
    
    # Size comparison
    original_size = sum(
        (source_dir / f).stat().st_size 
        for f in data_files.keys() 
        if (source_dir / f).exists()
    )
    new_size = sum(
        (target_dir / f).stat().st_size 
        for f in data_files.keys() 
        if (target_dir / f).exists()
    )
    
    print(f"\nDisk space:")
    print(f"  Original: {original_size / 1024**2:.1f} MB")
    print(f"  Resized: {new_size / 1024**2:.1f} MB")
    print(f"  Savings: {(original_size - new_size) / 1024**2:.1f} MB ({(1-new_size/original_size)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ DATASET RESIZING COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Train model with:")
    print(f"     python scripts/train.py \\")
    print(f"       --data-dir {target_dir} \\")
    print(f"       --model-output models/quickdraw_model_{args.target_size}x{args.target_size}.h5 \\")
    print(f"       --enhanced \\")
    print(f"       --aggressive-aug \\")
    print(f"       --epochs 100")
    print(f"\n  2. Convert to TFLite INT8:")
    print(f"     python scripts/convert/quantize_int8.py \\")
    print(f"       --model models/quickdraw_model_{args.target_size}x{args.target_size}.h5 \\")
    print(f"       --output models/quickdraw_model_{args.target_size}x{args.target_size}_int8.tflite")
    print()


if __name__ == '__main__':
    main()
