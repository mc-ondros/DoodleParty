"""
Training Dataset Regeneration (Simple Normalization)

Creates balanced binary classification dataset WITHOUT per-image normalization.
Why skip per-image normalization: Preserves natural brightness differences between
penis drawings and QuickDraw shapes, potentially providing exploitable signals for
the model while simplifying the preprocessing pipeline.

This approach differs from regenerate_data_simple_norm.py which applies per-image
normalization to remove brightness bias. Comparing both approaches helps determine
whether brightness is a useful signal or a shortcut that hurts generalization.

Related:
- data/processed/penis_raw_X.npy (source positive class)
- data/raw/{category}.npy (source negative class data)
- scripts/test_extended_training.sh (training orchestration)

Exports:
- Balanced dataset: X_train.npy, X_test.npy, y_train.npy, y_test.npy
- Class mapping: class_mapping.pkl
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

print("="*70)
print("REGENERATING DATA *WITHOUT* PER-IMAGE NORMALIZATION")
print("="*70)

# Paths
data_dir = Path('data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'

# Load positive class samples from preprocessed data
print("\n1. Loading penis data...")
penis_data = np.load(processed_dir / 'penis_raw_X.npy')
print(f"   ✓ Loaded {len(penis_data):,} penis samples")

# Convert from white-on-black to black-on-white to match QuickDraw convention
# Why invert: QuickDraw dataset uses black background (0) with white strokes (255)
# Penis data comes as white-on-black, requiring inversion for consistency
print("   ✓ Inverting to black background")
penis_data = 255 - penis_data

# Load negative class samples from raw QuickDraw data
print("\n2. Loading QuickDraw negatives...")
negative_classes = [
    'airplane', 'apple', 'arm', 'banana', 'bird', 'boomerang',
    'cat', 'circle', 'cloud', 'dog', 'drill', 'fish', 'flower',
    'house', 'moon', 'pencil', 'square', 'star', 'sun', 'tree', 'triangle'
]

negative_data_list = []
for cls in negative_classes:
    filepath = raw_dir / f'{cls}.npy'
    if filepath.exists():
        data = np.load(filepath)
        # Limit samples per class to prevent dataset imbalance
        # Why 1200: Balances having enough data while keeping training time reasonable
        samples = min(1200, len(data))
        data = data[:samples]

        # Ensure consistent 28x28 shape regardless of input dimensions
        if len(data.shape) == 2:
            data = data.reshape(-1, 28, 28)
        elif len(data.shape) in [3, 4]:
            data = data.reshape(-1, 28, 28)

        negative_data_list.append(data)
        print(f"   ✓ {cls}: {samples:,} samples")

negative_data = np.concatenate(negative_data_list, axis=0)
print(f"\n   Total negatives: {len(negative_data):,}")

# Balance positive and negative classes to prevent bias
print("\n3. Balancing...")
n_positive = len(penis_data)
n_negative = len(negative_data)

if n_negative > n_positive:
    # Subsample majority class to match minority class size
    # Why this approach: Maintains equal representation while reducing computational cost
    np.random.seed(42)
    indices = np.random.choice(n_negative, n_positive, replace=False)
    negative_data = negative_data[indices]
elif n_positive > n_negative:
    # Symmetric case for when positive class is larger
    np.random.seed(42)
    indices = np.random.choice(n_positive, n_negative, replace=False)
    penis_data = penis_data[indices]

# Create labels
y_positive = np.ones(len(penis_data), dtype=np.float32)
y_negative = np.zeros(len(negative_data), dtype=np.float32)

# Combine samples and create corresponding labels
X = np.concatenate([penis_data, negative_data], axis=0)
y = np.concatenate([y_positive, y_negative], axis=0)

print(f"\n   Combined: {len(X):,} samples (50/50 split)")

# Shuffle to randomize order before train/test split
# Why shuffle: Prevents ordered batches that could bias training
print("   ✓ Shuffling...")
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Normalize pixel values to [0, 1] range using simple division
# Why simple normalization: Preserves relative brightness differences between classes
# This differs from per-image normalization which centers each image independently
print("\n4. Simple normalization (divide by 255)...")
X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
print(f"   ✓ Shape: {X.shape}")
print(f"   Value range: {X.min():.3f} - {X.max():.3f}")
print(f"   Mean: {X.mean():.3f}")

# Verify brightness differences are preserved between classes
# Why this matters: This normalization approach should maintain class-discriminative
# brightness patterns that could aid classification
pos_mean = X[y == 1].mean()
neg_mean = X[y == 0].mean()
print(f"\n   Class brightness difference: {abs(pos_mean - neg_mean):.4f}")
if abs(pos_mean - neg_mean) > 0.01:
    print(f"   ✓ GOOD: Brightness signal preserved!")
else:
    print(f"   ⚠️  Classes have similar brightness")

# Create stratified train/test split to maintain class balance
print("\n5. Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y, shuffle=True
)

print(f"   ✓ Training: {len(X_train):,}")
print(f"   ✓ Test: {len(X_test):,}")

# Persist processed datasets for training
print("\n6. Saving...")
np.save(processed_dir / 'X_train.npy', X_train)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'X_test.npy', X_test)
np.save(processed_dir / 'y_test.npy', y_test)

# Save metadata for training and evaluation
class_mapping = {
    'positive': 1,
    'negative': 0,
    'categories': {
        'penis': 1,
        'hard_negatives': negative_classes
    },
    'description': 'Binary classification: positive=penis, negative=21 QuickDraw (simple /255 normalization)'
}

with open(processed_dir / 'class_mapping.pkl', 'wb') as f:
    pickle.dump(class_mapping, f)

print(f"   ✓ Saved to {processed_dir}/")

print("\n" + "="*70)
print("✅ DATA READY - Simple normalization (NO per-image)")
print("="*70)
print("\nKey difference: Brightness signal PRESERVED")
print("Why this matters: Preserves natural class differences while simplifying preprocessing")
print("Expected benefit: More stable training compared to per-image normalization")
print("\nNext: bash test_extended_training.sh")
