"""
Training Dataset Regeneration (Per-Image Normalization)

Creates balanced binary classification dataset with 128x128 resolution and
per-image normalization. Applies per-image standardization to remove brightness
bias and ensure consistent pixel value distributions across all samples.

Why per-image normalization: Prevents model from learning brightness shortcuts by
forcing focus on shape patterns. Each image is independently normalized to have
similar statistical properties regardless of original brightness.

Why 128x128: Higher resolution captures more detailed shape features compared to
original 28x28, improving model ability to distinguish complex anatomical features.

Related:
- data/processed/penis_raw_X_128.npy (source positive class, already 128x128)
- data/processed/{category}_128.npy (source negative class data, preprocessed)
- scripts/train_max_accuracy.sh (training orchestration)

Exports:
- Balanced dataset: X_train.npy, X_test.npy, y_train.npy, y_test.npy
- Class mapping: class_mapping.pkl
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

print("="*70)
print("REGENERATING TRAINING DATA (128x128)")
print("="*70)

# Paths
data_dir = Path('data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'

# Load positive class samples with high resolution
print("\n1. Loading fixed penis data (positive class, 128x128)...")
penis_data = np.load(processed_dir / 'penis_raw_X_128.npy')
print(f"   ✓ Loaded {len(penis_data):,} penis samples")
print(f"   Shape: {penis_data.shape}")
print(f"   Range: {penis_data.min()} - {penis_data.max()}, mean: {penis_data.mean():.2f}")

# Use data in native format (black background, white strokes)
# Why no inversion: penis_raw_X_128.npy already matches QuickDraw convention
print("   ✓ Penis data already in QuickDraw format (black background, white strokes)")

# Load negative class samples from preprocessed QuickDraw data
print("\n2. Loading QuickDraw negative classes (128x128)...")
negative_classes = [
    'airplane', 'apple', 'arm', 'banana', 'bird', 'boomerang',
    'cat', 'circle', 'cloud', 'dog', 'drill', 'fish', 'flower',
    'house', 'moon', 'pencil', 'square', 'star', 'sun', 'tree', 'triangle'
]

negative_data_list = []
for cls in negative_classes:
    # Use preprocessed 128x128 files that are already filtered and inverted
    # Why use processed: Avoids redundant preprocessing; these files are ready for training
    filepath = processed_dir / f'{cls}_128.npy'
    if filepath.exists():
        data = np.load(filepath)
        print(f"   ✓ {cls}: {len(data):,} samples (128x128)")
        negative_data_list.append(data)
    else:
        print(f"   ⚠️  {cls}: file not found, skipping")

# Combine all negative class samples
negative_data = np.concatenate(negative_data_list, axis=0)
print(f"\n   Total negative samples: {len(negative_data):,}")

# Balance dataset to prevent class imbalance during training
print("\n3. Balancing dataset...")
n_positive = len(penis_data)
n_negative = len(negative_data)

if n_negative > n_positive:
    # Subsample majority class to match minority class size
    # Why balanced: Prevents model bias toward majority class
    np.random.seed(42)
    indices = np.random.choice(n_negative, n_positive, replace=False)
    negative_data = negative_data[indices]
    print(f"   ✓ Sampled {len(negative_data):,} negative samples to match positives")
elif n_positive > n_negative:
    # Symmetric case when positive class is larger
    np.random.seed(42)
    indices = np.random.choice(n_positive, n_negative, replace=False)
    penis_data = penis_data[indices]
    print(f"   ✓ Sampled {len(penis_data):,} positive samples to match negatives")

# Create binary labels for positive (1) and negative (0) classes
y_positive = np.ones(len(penis_data), dtype=np.float32)
y_negative = np.zeros(len(negative_data), dtype=np.float32)

# Combine data and shuffle to prevent class ordering bias
X = np.concatenate([penis_data, negative_data], axis=0)
y = np.concatenate([y_positive, y_negative], axis=0)

print(f"\n   Combined dataset: {len(X):,} samples")
print(f"   - Positive (penis): {(y == 1).sum():,} ({100*(y==1).sum()/len(y):.1f}%)")
print(f"   - Negative (QuickDraw): {(y == 0).sum():,} ({100*(y==0).sum()/len(y):.1f}%)")

# Shuffle immediately to ensure random class distribution
# Why shuffle: Prevents sequential batches containing only one class
print("\n   ✓ Shuffling to prevent class clustering...")
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]
print(f"   First 10 labels after shuffle: {y[:10]}")

# Normalize pixel values to [0, 1] range and add channel dimension
print("\n4. Normalizing and reshaping...")
X = X.reshape(-1, 128, 128, 1).astype(np.float32) / 255.0
print(f"   ✓ Shape: {X.shape}, dtype: {X.dtype}")
print(f"   Value range: {X.min():.3f} - {X.max():.3f}")

# Apply per-image standardization to remove brightness-based shortcuts
# Why per-image: Normalizes each image independently to have similar statistical
# properties, forcing model to learn shape features rather than brightness patterns
print("\n4b. Applying per-image normalization to remove brightness shortcuts...")
for i in range(len(X)):
    img = X[i]
    img_flat = img.flatten()
    # Standardize to zero mean and unit variance, then rescale to [0, 1]
    if img_flat.std() > 0.01:  # Skip normalization for blank images
        img = (img - img_flat.mean()) / (img_flat.std() + 1e-7)
        img = (img + 3) / 6  # Rescale from [-3, 3] to approximately [0, 1]
        X[i] = np.clip(img, 0, 1)

print(f"   ✓ Applied per-image normalization to {len(X):,} samples")
print(f"   New value range: {X.min():.3f} - {X.max():.3f}")
print(f"   Mean pixel value: {X.mean():.3f}")

# Create stratified train/test split with shuffling
print("\n5. Splitting into train/test...")
# Why stratification: Ensures both sets have same class distribution
# Why shuffling: Randomizes order to prevent bias from class ordering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y, shuffle=True
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"     - Positive: {(y_train == 1).sum():,} ({100*(y_train==1).sum()/len(y_train):.1f}%)")
print(f"     - Negative: {(y_train == 0).sum():,} ({100*(y_train==0).sum()/len(y_train):.1f}%)")
print(f"   ✓ Test set: {len(X_test):,} samples")
print(f"     - Positive: {(y_test == 1).sum():,} ({100*(y_test==1).sum()/len(y_test):.1f}%)")
print(f"     - Negative: {(y_test == 0).sum():,} ({100*(y_test==0).sum()/len(y_test):.1f}%)")

# Persist processed datasets for training
print("\n6. Saving datasets...")
np.save(processed_dir / 'X_train.npy', X_train)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'X_test.npy', X_test)
np.save(processed_dir / 'y_test.npy', y_test)

# Report file sizes for storage planning
train_size = X_train.nbytes / 1024 / 1024
test_size = X_test.nbytes / 1024 / 1024
print(f"   ✓ X_train.npy: {train_size:.2f} MB")
print(f"   ✓ X_test.npy: {test_size:.2f} MB")

# Save metadata including normalization strategy
class_mapping = {
    'positive': 1,
    'negative': 0,
    'categories': {
        'penis': 1,
        'hard_negatives': negative_classes
    },
    'description': 'Binary classification: positive=penis, negative=21 QuickDraw (per-image normalized, 128x128)',
    'image_size': (128, 128),
    'normalization': 'per_image',
}

with open(processed_dir / 'class_mapping.pkl', 'wb') as f:
    pickle.dump(class_mapping, f)

print(f"   ✓ Saved to {processed_dir}/")
print(f"     - X_train.npy: {X_train.nbytes / 1024 / 1024:.2f} MB")
print(f"     - X_test.npy: {X_test.nbytes / 1024 / 1024:.2f} MB")
print(f"     - y_train.npy: {y_train.nbytes / 1024:.2f} KB")
print(f"     - y_test.npy: {y_test.nbytes / 1024:.2f} KB")
print(f"     - class_mapping.pkl")

print("\n" + "="*70)
print("✅ TRAINING DATA REGENERATED SUCCESSFULLY")
print("="*70)
print("\nDataset ready for training with:")
print("  • 128x128 high-resolution images")
print("  • Fixed, properly centered penis drawings")
print("  • Balanced positive/negative classes")
print("  • 80/20 train/test split")
print("  • Per-image normalization (removes brightness bias)")
print("  • Filtered blank samples")
print("\nWhy per-image normalization: Forces model to learn shape patterns")
print("rather than brightness shortcuts, improving generalization")
print("\nNext step: Run training")
print("  bash train_max_accuracy.sh")
