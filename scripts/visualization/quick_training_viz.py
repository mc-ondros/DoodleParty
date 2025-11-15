#!/usr/bin/env python3
"""
Quick Training Data Visualizer

Simple script to visualize what the training data looks like.
Shows you the exact format: black background, white strokes, 128x128 grayscale.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

project_root = Path(__file__).parent.parent.parent


def find_and_visualize():
    """Find and visualize training data"""
    
    # Try different possible data locations
    data_paths = [
        project_root / 'data' / 'processed_128x128',
        project_root / 'data' / 'processed_96x96',
        project_root / 'data' / 'processed_64x64',
        project_root / 'data' / 'processed',
    ]
    
    print("🔍 Searching for training data...\n")
    
    for data_path in data_paths:
        if data_path.exists():
            print(f"✓ Found: {data_path}")
            categories = [d.name for d in data_path.iterdir() if d.is_dir()]
            if categories:
                print(f"  Categories: {', '.join(categories[:10])}")
                visualize_from_directory(data_path, categories)
                return
    
    print("❌ No training data found in standard locations.")
    print("\nSearched:")
    for p in data_paths:
        print(f"  - {p}")
    print("\nTo generate training data, run:")
    print("  python scripts/data_processing/process_all_data_128x128.py")


def visualize_from_directory(data_path: Path, categories: list):
    """Visualize samples from directory structure"""
    
    # Pick some interesting categories
    sample_categories = []
    
    # Try to find positive class (inappropriate)
    pos_candidates = ['penis', 'inappropriate', 'nsfw']
    for cat in pos_candidates:
        if cat in categories:
            sample_categories.append(('POSITIVE (Inappropriate)', cat, 'red'))
            break
    
    # Add some negative classes (safe)
    neg_candidates = ['airplane', 'apple', 'banana', 'cat', 'circle', 'cloud', 'face']
    for cat in neg_candidates:
        if cat in categories:
            sample_categories.append(('NEGATIVE (Safe)', cat, 'green'))
            if len(sample_categories) >= 4:
                break
    
    # If we didn't find specific ones, just use first available
    if len(sample_categories) < 2:
        for cat in categories[:4]:
            label = 'POSITIVE' if cat == categories[0] else 'NEGATIVE'
            color = 'red' if cat == categories[0] else 'green'
            sample_categories.append((label, cat, color))
    
    # Create figure
    fig = plt.figure(figsize=(20, len(sample_categories) * 3))
    fig.suptitle('Training Data Format Visualization - QuickDraw Style', 
                 fontsize=16, weight='bold')
    
    rows = len(sample_categories)
    cols = 8
    
    for row_idx, (label, category, color) in enumerate(sample_categories):
        cat_path = data_path / category
        
        # Load samples
        samples = load_category_samples(cat_path, max_samples=cols)
        
        if not samples:
            print(f"⚠ No samples found for {category}")
            continue
        
        print(f"\n{label}: {category}")
        print(f"  Samples found: {len(samples)}")
        
        # Analyze first sample
        if samples:
            s = samples[0]
            print(f"  Shape: {s.shape}")
            print(f"  Dtype: {s.dtype}")
            print(f"  Range: [{s.min()}, {s.max()}]")
            print(f"  Mean: {s.mean():.1f}")
            
            # Check format
            corners = [s[0,0], s[0,-1], s[-1,0], s[-1,-1]]
            bg_mean = np.mean(corners)
            
            if bg_mean < 50:
                print(f"  Format: ✓ Black background ({bg_mean:.0f}) - CORRECT QuickDraw format")
            elif bg_mean > 200:
                print(f"  Format: ✗ White background ({bg_mean:.0f}) - WRONG! Needs inversion")
            else:
                print(f"  Format: ? Unclear ({bg_mean:.0f})")
        
        # Plot samples
        for col_idx, sample in enumerate(samples):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1)
            
            # Handle different shapes
            if len(sample.shape) == 3 and sample.shape[-1] == 1:
                sample = sample[:, :, 0]
            
            ax.imshow(sample, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            
            # Label first image
            if col_idx == 0:
                ax.set_title(f'{category}\n{label}', 
                           fontsize=9, color=color, weight='bold', pad=3)
    
    # Add format explanation
    info_text = (
        "✓ CORRECT FORMAT: Black background (0-50), White strokes (200-255)\n"
        "✗ WRONG FORMAT: White background, Dark strokes (needs color inversion)\n"
        "Model expects: Grayscale, 128×128, Black BG + White strokes"
    )
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    
    # Save
    output_path = project_root / 'logs' / 'training_data_visualization.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Also show
    print("\n📊 Showing plot (close window to exit)...")
    plt.show()


def load_category_samples(cat_path: Path, max_samples: int = 8):
    """Load samples from a category directory"""
    samples = []
    
    # Try .npy files first
    npy_files = list(cat_path.glob('*.npy'))
    if npy_files:
        for npy_file in npy_files[:max_samples]:
            try:
                data = np.load(npy_file)
                if len(data.shape) == 1:
                    # Array of images
                    for img in data[:max_samples - len(samples)]:
                        samples.append(img)
                        if len(samples) >= max_samples:
                            break
                else:
                    samples.append(data)
                
                if len(samples) >= max_samples:
                    break
            except Exception as e:
                print(f"  Error loading {npy_file.name}: {e}")
    
    # Try image files
    if not samples:
        img_files = (list(cat_path.glob('*.png')) + 
                    list(cat_path.glob('*.jpg')) + 
                    list(cat_path.glob('*.jpeg')))
        
        for img_file in img_files[:max_samples]:
            try:
                img = Image.open(img_file).convert('L')
                samples.append(np.array(img))
            except Exception as e:
                print(f"  Error loading {img_file.name}: {e}")
    
    return samples[:max_samples]


if __name__ == '__main__':
    print("="*60)
    print("  TRAINING DATA FORMAT VISUALIZER")
    print("="*60)
    print()
    
    find_and_visualize()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
