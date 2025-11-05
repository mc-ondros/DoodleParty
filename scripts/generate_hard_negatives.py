"""
Generate hard negative samples from QuickDraw dataset.

Hard negatives are challenging examples that fool the model:
1. Doodles from similar categories (potential false positives)
2. Partial/incomplete doodles
3. Rotated and scaled versions
4. Low-quality/noisy sketches

This creates a more robust training set by forcing the model
to learn subtle discriminative features.
"""

import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw


def load_quickdraw_category(category, max_samples=5000):
    """Load QuickDraw category data."""
    data_dir = Path("/home/mcvaj/ML/data/raw")
    filepath = data_dir / f"{category}.npy"
    
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None
    
    data = np.load(filepath)
    if max_samples and len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
    
    return data


def create_partial_doodles(images, num_samples=None):
    """Create partial doodles by randomly cropping."""
    if num_samples is None:
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=True)
    partial = []
    
    for idx in indices:
        img = Image.fromarray(images[idx])
        width, height = img.size
        
        # Random crop that removes part of the drawing
        crop_size = np.random.randint(int(height * 0.5), height)
        left = np.random.randint(0, width - crop_size + 1)
        top = np.random.randint(0, height - crop_size + 1)
        
        cropped = img.crop((left, top, left + crop_size, top + crop_size))
        # Resize back to 28x28
        cropped = cropped.resize((28, 28), Image.Resampling.LANCZOS)
        
        partial.append(np.array(cropped))
    
    return np.array(partial, dtype=np.uint8)


def create_rotated_doodles(images, num_samples=None, max_rotation=45):
    """Create rotated doodles with random angles."""
    if num_samples is None:
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=True)
    rotated = []
    
    for idx in indices:
        img = Image.fromarray(images[idx])
        angle = np.random.uniform(-max_rotation, max_rotation)
        rotated_img = img.rotate(angle, expand=False, fillcolor=255)
        rotated.append(np.array(rotated_img))
    
    return np.array(rotated, dtype=np.uint8)


def create_noisy_doodles(images, num_samples=None, noise_level=0.2):
    """Add Gaussian noise to create degraded doodles."""
    if num_samples is None:
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=True)
    noisy = []
    
    for idx in indices:
        img = images[idx].astype(float)
        noise = np.random.normal(0, 255 * noise_level, img.shape)
        noisy_img = np.clip(img + noise, 0, 255)
        noisy.append(noisy_img.astype(np.uint8))
    
    return np.array(noisy, dtype=np.uint8)


def create_faded_doodles(images, num_samples=None):
    """Create faded/light doodles by reducing contrast."""
    if num_samples is None:
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=True)
    faded = []
    
    for idx in indices:
        img = images[idx].astype(float)
        # Fade towards white (255)
        fade_factor = np.random.uniform(0.3, 0.7)
        faded_img = img * fade_factor + 255 * (1 - fade_factor)
        faded.append(np.clip(faded_img, 0, 255).astype(np.uint8))
    
    return np.array(faded, dtype=np.uint8)


def create_scaled_doodles(images, num_samples=None):
    """Create scaled/zoomed versions of doodles."""
    if num_samples is None:
        num_samples = len(images)
    
    indices = np.random.choice(len(images), num_samples, replace=True)
    scaled = []
    
    for idx in indices:
        img = Image.fromarray(images[idx])
        
        # Random zoom factor
        zoom = np.random.uniform(0.6, 1.4)
        new_size = int(28 * zoom)
        
        # Resize
        resized = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
        
        # Pad or crop back to 28x28
        if new_size > 28:
            # Crop from center
            left = (new_size - 28) // 2
            top = (new_size - 28) // 2
            resized = resized.crop((left, top, left + 28, top + 28))
        else:
            # Pad with white
            padded = Image.new('L', (28, 28), color=255)
            pad_left = (28 - new_size) // 2
            pad_top = (28 - new_size) // 2
            padded.paste(resized, (pad_left, pad_top))
            resized = padded
        
        scaled.append(np.array(resized))
    
    return np.array(scaled, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate hard negative samples")
    parser.add_argument("--similar-categories", nargs="+", 
                       default=["flower", "tree", "house", "bird"],
                       help="Similar categories to use as hard negatives")
    parser.add_argument("--samples-per-type", type=int, default=2000,
                       help="Samples per hard negative type")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HARD NEGATIVE SAMPLES GENERATOR")
    print("=" * 70)
    
    all_hard_negatives = []
    
    # 1. Similar category doodles
    print("\n1. Loading similar category doodles...")
    for category in args.similar_categories:
        print(f"   Loading {category}...", end=" ", flush=True)
        data = load_quickdraw_category(category, max_samples=args.samples_per_type)
        if data is not None:
            all_hard_negatives.append(data)
            print(f"✓ ({len(data)} samples)")
        else:
            print("✗")
    
    # 2. Partial doodles (challenging - incomplete drawings)
    print("\n2. Generating partial doodles...")
    if all_hard_negatives:
        source_data = all_hard_negatives[0]
        partial = create_partial_doodles(source_data, num_samples=args.samples_per_type)
        all_hard_negatives.append(partial)
        print(f"   ✓ Generated {len(partial)} partial doodles")
    
    # 3. Rotated doodles
    print("\n3. Generating rotated doodles...")
    if all_hard_negatives:
        source_data = all_hard_negatives[0]
        rotated = create_rotated_doodles(source_data, num_samples=args.samples_per_type, max_rotation=30)
        all_hard_negatives.append(rotated)
        print(f"   ✓ Generated {len(rotated)} rotated doodles")
    
    # 4. Noisy doodles (degraded quality)
    print("\n4. Generating noisy doodles...")
    if all_hard_negatives:
        source_data = all_hard_negatives[0]
        noisy = create_noisy_doodles(source_data, num_samples=args.samples_per_type, noise_level=0.15)
        all_hard_negatives.append(noisy)
        print(f"   ✓ Generated {len(noisy)} noisy doodles")
    
    # 5. Faded doodles
    print("\n5. Generating faded doodles...")
    if all_hard_negatives:
        source_data = all_hard_negatives[0]
        faded = create_faded_doodles(source_data, num_samples=args.samples_per_type)
        all_hard_negatives.append(faded)
        print(f"   ✓ Generated {len(faded)} faded doodles")
    
    # 6. Scaled doodles
    print("\n6. Generating scaled doodles...")
    if all_hard_negatives:
        source_data = all_hard_negatives[0]
        scaled = create_scaled_doodles(source_data, num_samples=args.samples_per_type)
        all_hard_negatives.append(scaled)
        print(f"   ✓ Generated {len(scaled)} scaled doodles")
    
    # Combine all
    if all_hard_negatives:
        X_hard_negatives = np.concatenate(all_hard_negatives, axis=0)
        
        # Save
        output_file = output_dir / "hard_negatives.npy"
        np.save(output_file, X_hard_negatives)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total hard negative samples: {len(X_hard_negatives)}")
        print(f"Breakdown:")
        print(f"  - Similar categories: {sum(len(d) for d in all_hard_negatives[:len(args.similar_categories)])}")
        print(f"  - Partial doodles: {len(partial) if 'partial' in locals() else 0}")
        print(f"  - Rotated doodles: {len(rotated) if 'rotated' in locals() else 0}")
        print(f"  - Noisy doodles: {len(noisy) if 'noisy' in locals() else 0}")
        print(f"  - Faded doodles: {len(faded) if 'faded' in locals() else 0}")
        print(f"  - Scaled doodles: {len(scaled) if 'scaled' in locals() else 0}")
        print(f"\nSaved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f}MB")
    else:
        print("No data loaded!")


if __name__ == "__main__":
    main()
