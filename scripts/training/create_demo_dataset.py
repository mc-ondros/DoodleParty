#!/usr/bin/env python3
"""
Create a synthetic demo dataset for testing the ML pipeline.

IMPORTANT: This is NOT for production use!
For production content moderation, you need REAL inappropriate drawings.

This script creates synthetic "inappropriate" content by:
1. Taking QuickDraw drawings
2. Applying transformations to simulate inappropriate shapes
3. Creating a balanced binary classification dataset

For PRODUCTION, you must:
- Collect actual flagged inappropriate drawings from your system
- Manually review and label them
- Build a proper dataset with real-world examples
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


def create_synthetic_inappropriate(num_samples=5000, size=128):
    """
    Create synthetic inappropriate shapes for DEMO purposes only.
    
    This creates simple geometric patterns that simulate inappropriate content
    but are NOT real examples. Use only for testing the pipeline.
    """
    images = []
    
    print(f"Generating {num_samples} synthetic inappropriate samples...")
    
    for i in range(num_samples):
        # Create blank image
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        
        # Draw random geometric patterns that could be flagged
        # (This is just for demo - NOT real inappropriate content)
        x = np.random.randint(20, size - 20)
        y = np.random.randint(20, size - 60)
        
        # Draw elongated oval (simulating inappropriate shape)
        width = np.random.randint(15, 30)
        height = np.random.randint(40, 80)
        draw.ellipse([x - width//2, y, x + width//2, y + height], fill=0)
        
        # Add two circles at bottom (completing the pattern)
        if np.random.random() > 0.3:
            circle_y = y + height
            circle_r = np.random.randint(8, 15)
            draw.ellipse([x - width//2 - circle_r, circle_y, 
                         x - width//2 + circle_r, circle_y + circle_r*2], fill=0)
            draw.ellipse([x + width//2 - circle_r, circle_y, 
                         x + width//2 + circle_r, circle_y + circle_r*2], fill=0)
        
        # Convert to numpy
        img_array = np.array(img)
        images.append(img_array)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples}...")
    
    return np.array(images, dtype=np.uint8)


def load_appropriate_content(data_dir, categories, max_per_category=5000):
    """Load appropriate content from QuickDraw categories."""
    images = []
    
    print(f"\nLoading appropriate content from QuickDraw...")
    
    for category in categories:
        file_path = os.path.join(data_dir, f'{category}.npy')
        if os.path.exists(file_path):
            print(f"  Loading {category}...", end=" ")
            data = np.load(file_path)
            
            # Take subset
            if len(data) > max_per_category:
                indices = np.random.choice(len(data), max_per_category, replace=False)
                data = data[indices]
            
            images.append(data)
            print(f"✓ ({len(data)} samples)")
        else:
            print(f"  ✗ {category}.npy not found, skipping...")
    
    if not images:
        raise ValueError("No appropriate content loaded! Download QuickDraw data first.")
    
    return np.concatenate(images, axis=0)


def create_balanced_dataset(
    inappropriate_data,
    appropriate_data,
    output_dir='data/processed',
    train_split=0.8
):
    """Create balanced binary classification dataset."""
    
    # Balance classes
    min_samples = min(len(inappropriate_data), len(appropriate_data))
    print(f"\nBalancing dataset to {min_samples} samples per class...")
    
    inappropriate_data = inappropriate_data[:min_samples]
    appropriate_data = appropriate_data[:min_samples]
    
    # Create labels (0 = appropriate, 1 = inappropriate)
    inappropriate_labels = np.ones(len(inappropriate_data), dtype=np.uint8)
    appropriate_labels = np.zeros(len(appropriate_data), dtype=np.uint8)
    
    # Combine and shuffle
    all_images = np.concatenate([appropriate_data, inappropriate_data], axis=0)
    all_labels = np.concatenate([appropriate_labels, inappropriate_labels], axis=0)
    
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    # Split train/val
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    train_labels = all_labels[:split_idx]
    val_images = all_images[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving dataset to {output_dir}...")
    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'val_images.npy'), val_images)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)
    
    print(f"  ✓ Training set: {len(train_images)} samples")
    print(f"  ✓ Validation set: {len(val_images)} samples")
    print(f"  ✓ Class distribution: {np.sum(train_labels)} inappropriate, {len(train_labels) - np.sum(train_labels)} appropriate")
    
    return (train_images, train_labels), (val_images, val_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Create synthetic demo dataset for ML pipeline testing'
    )
    parser.add_argument('--quickdraw-dir', type=str, default='data/raw',
                        help='Directory with QuickDraw .npy files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed dataset')
    parser.add_argument('--num-synthetic', type=int, default=5000,
                        help='Number of synthetic inappropriate samples to generate')
    parser.add_argument('--appropriate-categories', type=str, nargs='+',
                        default=['dog', 'cat', 'house', 'tree'],
                        help='QuickDraw categories to use as appropriate content')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Demo Dataset Creator")
    print("=" * 70)
    print("\n⚠️  WARNING: This creates SYNTHETIC data for DEMO purposes only!")
    print("For production, you MUST use real inappropriate drawings collected")
    print("from your moderation system.")
    print("=" * 70)
    
    # Create synthetic inappropriate content
    print("\n[1/3] Creating synthetic inappropriate content...")
    inappropriate_data = create_synthetic_inappropriate(args.num_synthetic)
    
    # Load appropriate content
    print("\n[2/3] Loading appropriate content from QuickDraw...")
    appropriate_data = load_appropriate_content(
        args.quickdraw_dir,
        args.appropriate_categories
    )
    
    # Create balanced dataset
    print("\n[3/3] Creating balanced dataset...")
    create_balanced_dataset(
        inappropriate_data,
        appropriate_data,
        args.output_dir
    )
    
    print("\n" + "=" * 70)
    print("Dataset created successfully!")
    print("=" * 70)
    print(f"\nYou can now train with:")
    print(f"  python scripts/training/train_binary.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
