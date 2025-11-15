#!/usr/bin/env python3
"""
Simple Training Data Grid Viewer

Creates a visual grid of training data samples without matplotlib dependency.
Saves to PNG and prints format statistics.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).parent.parent.parent


def create_sample_grid(samples, title, text_color=(0, 255, 0)):
    """Create a grid of sample images"""
    if not samples:
        return None
    
    # Grid dimensions
    cols = min(8, len(samples))
    rows = (len(samples) + cols - 1) // cols
    
    # Get sample size
    sample_size = samples[0].shape[0]  # Assuming square images
    
    # Create grid image
    padding = 10
    text_height = 30
    grid_width = cols * sample_size + (cols + 1) * padding
    grid_height = rows * sample_size + (rows + 1) * padding + text_height
    
    grid = Image.new('RGB', (grid_width, grid_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)
    
    # Add title
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((padding, 5), title, fill=text_color, font=font)
    
    # Place samples
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        
        x = col * sample_size + (col + 1) * padding
        y = row * sample_size + (row + 1) * padding + text_height
        
        # Convert sample to PIL Image
        if len(sample.shape) == 3 and sample.shape[-1] == 1:
            sample = sample[:, :, 0]
        
        sample_img = Image.fromarray(sample.astype(np.uint8), mode='L')
        grid.paste(sample_img, (x, y))
    
    return grid


def analyze_samples(samples, category_name):
    """Analyze and print statistics about samples"""
    if not samples:
        return None
    
    sample = samples[0]
    
    # Calculate statistics
    all_pixels = np.concatenate([s.flatten() for s in samples[:min(10, len(samples))]])
    
    # Check corners for background color
    corners = []
    for s in samples[:min(5, len(samples))]:
        if len(s.shape) == 3:
            s = s[:, :, 0]
        corners.extend([s[0,0], s[0,-1], s[-1,0], s[-1,-1]])
    bg_mean = np.mean(corners)
    
    stats = {
        'category': category_name,
        'count': len(samples),
        'shape': sample.shape,
        'dtype': sample.dtype,
        'min': all_pixels.min(),
        'max': all_pixels.max(),
        'mean': all_pixels.mean(),
        'std': all_pixels.std(),
        'bg_mean': bg_mean
    }
    
    # Format assessment
    if bg_mean < 50:
        stats['format'] = 'CORRECT'
        stats['format_desc'] = f'Black background ({bg_mean:.0f}), white strokes'
    elif bg_mean > 200:
        stats['format'] = 'WRONG'
        stats['format_desc'] = f'White background ({bg_mean:.0f}), needs inversion!'
    else:
        stats['format'] = 'UNCLEAR'
        stats['format_desc'] = f'Ambiguous background ({bg_mean:.0f})'
    
    return stats


def load_samples_from_category(cat_path: Path, max_samples: int = 16):
    """Load samples from category directory"""
    samples = []
    
    # Try .npy files
    npy_files = sorted(cat_path.glob('*.npy'))
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            if len(data.shape) == 1:
                # Array of images
                for img in data:
                    samples.append(img)
                    if len(samples) >= max_samples:
                        break
            elif len(data.shape) in [2, 3]:
                samples.append(data)
            
            if len(samples) >= max_samples:
                break
        except Exception as e:
            print(f"  ⚠ Error loading {npy_file.name}: {e}")
    
    # Try image files if no .npy found
    if not samples:
        img_files = sorted(list(cat_path.glob('*.png')) + list(cat_path.glob('*.jpg')))
        for img_file in img_files[:max_samples]:
            try:
                img = Image.open(img_file).convert('L')
                samples.append(np.array(img))
            except Exception as e:
                print(f"  ⚠ Error loading {img_file.name}: {e}")
    
    return samples[:max_samples]


def main():
    print("="*70)
    print("  TRAINING DATA FORMAT CHECKER")
    print("="*70)
    print()
    
    # Find data directory
    data_paths = [
        project_root / 'data' / 'processed_128x128',
        project_root / 'data' / 'processed_96x96',
        project_root / 'data' / 'processed_64x64',
        project_root / 'data' / 'processed',
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            print(f"✓ Found data: {path}")
            break
    
    if not data_path:
        print("❌ No training data found!")
        print("\nSearched locations:")
        for p in data_paths:
            print(f"  - {p}")
        return
    
    # Get categories
    categories = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"  Categories: {len(categories)}")
    print()
    
    # Select categories to visualize
    sample_cats = []
    
    # Positive class
    for cat in ['penis', 'inappropriate', 'nsfw']:
        if cat in categories:
            sample_cats.append(('POSITIVE', cat, (255, 100, 100)))
            break
    
    # Negative classes
    for cat in ['airplane', 'apple', 'banana', 'cat', 'circle', 'cloud']:
        if cat in categories and len(sample_cats) < 3:
            sample_cats.append(('NEGATIVE', cat, (100, 255, 100)))
    
    # Fallback to first available
    if not sample_cats:
        for cat in categories[:3]:
            label = 'SAMPLE'
            color = (200, 200, 200)
            sample_cats.append((label, cat, color))
    
    # Process each category
    all_stats = []
    grids = []
    
    for label, cat, color in sample_cats:
        print(f"{'='*70}")
        print(f"{label}: {cat}")
        print(f"{'='*70}")
        
        cat_path = data_path / cat
        samples = load_samples_from_category(cat_path, max_samples=16)
        
        if not samples:
            print(f"⚠ No samples found in {cat_path}")
            print()
            continue
        
        # Analyze
        stats = analyze_samples(samples, cat)
        all_stats.append(stats)
        
        # Print stats
        print(f"  Samples: {stats['count']}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Value range: [{stats['min']}, {stats['max']}]")
        print(f"  Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        print(f"  Background: {stats['bg_mean']:.0f}")
        print(f"  Format: {stats['format']} - {stats['format_desc']}")
        print()
        
        # Create grid
        title = f"{label}: {cat} ({stats['count']} samples) - {stats['format']}"
        grid = create_sample_grid(samples, title, color)
        if grid:
            grids.append(grid)
    
    # Combine grids vertically
    if grids:
        total_height = sum(g.height for g in grids) + 100
        combined_width = max(g.width for g in grids)
        
        combined = Image.new('RGB', (combined_width, total_height), color=(20, 20, 20))
        
        # Add main title
        draw = ImageDraw.Draw(combined)
        try:
            font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
        except:
            font_title = ImageFont.load_default()
        
        draw.text((20, 20), "Training Data Format Visualization", fill=(255, 255, 255), font=font_title)
        draw.text((20, 55), f"Expected: Black BG (0-50), White strokes (200-255), Grayscale", 
                 fill=(200, 200, 200))
        
        # Paste grids
        y_offset = 100
        for grid in grids:
            combined.paste(grid, (0, y_offset))
            y_offset += grid.height
        
        # Save
        output_path = project_root / 'logs' / 'training_data_visualization.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path)
        
        print(f"{'='*70}")
        print(f"✅ Visualization saved to:")
        print(f"   {output_path}")
        print(f"{'='*70}")
        
        # Summary
        print("\nFORMAT SUMMARY:")
        print("-" * 70)
        correct = sum(1 for s in all_stats if s['format'] == 'CORRECT')
        wrong = sum(1 for s in all_stats if s['format'] == 'WRONG')
        
        if correct == len(all_stats):
            print("✅ ALL categories have CORRECT format (black BG, white strokes)")
        elif wrong > 0:
            print(f"⚠️ {wrong}/{len(all_stats)} categories have WRONG format!")
            print("   Images need color inversion before ML processing")
        
        print()
        
    else:
        print("❌ No visualizations created")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
