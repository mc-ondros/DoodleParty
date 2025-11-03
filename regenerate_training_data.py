"""
Regenerate training dataset with the fixed penis data.
Uses penis as positive class and QuickDraw classes as hard negatives.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

print("="*70)
print("REGENERATING TRAINING DATA WITH FIXED PENIS PROCESSING")
print("="*70)

# Paths
data_dir = Path('data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'

# Load fixed penis data (positive class)
print("\n1. Loading fixed penis data (positive class)...")
penis_data = np.load(processed_dir / 'penis_raw_X.npy')
print(f"   ✓ Loaded {len(penis_data):,} penis samples")
print(f"   Shape: {penis_data.shape}")

# Load negative classes (QuickDraw standard classes)
print("\n2. Loading QuickDraw negative classes...")
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
        # Take up to 1200 samples per class to match penis data size
        samples = min(1200, len(data))
        data = data[:samples]
        
        # Ensure consistent shape (N, 28, 28)
        if len(data.shape) == 2:
            # Flattened (N, 784), reshape to (N, 28, 28)
            data = data.reshape(-1, 28, 28)
        elif len(data.shape) == 3 and data.shape[-1] == 1:
            # Has channel dimension (N, 28, 28, 1), squeeze it
            data = data.reshape(-1, 28, 28)
        elif len(data.shape) == 4:
            # (N, 28, 28, 1), squeeze channel
            data = data.squeeze(-1)
        
        negative_data_list.append(data)
        print(f"   ✓ {cls}: {samples:,} samples")
    else:
        print(f"   ⚠️  {cls}: file not found, skipping")

# Combine negative samples
negative_data = np.concatenate(negative_data_list, axis=0)
print(f"\n   Total negative samples: {len(negative_data):,}")

# Balance the dataset
print("\n3. Balancing dataset...")
n_positive = len(penis_data)
n_negative = len(negative_data)

if n_negative > n_positive:
    # Randomly sample negatives to match positives
    np.random.seed(42)
    indices = np.random.choice(n_negative, n_positive, replace=False)
    negative_data = negative_data[indices]
    print(f"   ✓ Sampled {len(negative_data):,} negative samples to match positives")
elif n_positive > n_negative:
    # Randomly sample positives to match negatives
    np.random.seed(42)
    indices = np.random.choice(n_positive, n_negative, replace=False)
    penis_data = penis_data[indices]
    print(f"   ✓ Sampled {len(penis_data):,} positive samples to match negatives")

# Create labels
y_positive = np.ones(len(penis_data), dtype=np.float32)
y_negative = np.zeros(len(negative_data), dtype=np.float32)

# Combine data
X = np.concatenate([penis_data, negative_data], axis=0)
y = np.concatenate([y_positive, y_negative], axis=0)

print(f"\n   Combined dataset: {len(X):,} samples")
print(f"   - Positive (penis): {(y == 1).sum():,} ({100*(y==1).sum()/len(y):.1f}%)")
print(f"   - Negative (QuickDraw): {(y == 0).sum():,} ({100*(y==0).sum()/len(y):.1f}%)")

# Normalize to 0-1 range and add channel dimension
print("\n4. Normalizing and reshaping...")
X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
print(f"   ✓ Shape: {X.shape}, dtype: {X.dtype}")
print(f"   Value range: {X.min():.3f} - {X.max():.3f}")

# Split into train and test
print("\n5. Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"     - Positive: {(y_train == 1).sum():,} ({100*(y_train==1).sum()/len(y_train):.1f}%)")
print(f"     - Negative: {(y_train == 0).sum():,} ({100*(y_train==0).sum()/len(y_train):.1f}%)")
print(f"   ✓ Test set: {len(X_test):,} samples")
print(f"     - Positive: {(y_test == 1).sum():,} ({100*(y_test==1).sum()/len(y_test):.1f}%)")
print(f"     - Negative: {(y_test == 0).sum():,} ({100*(y_test==0).sum()/len(y_test):.1f}%)")

# Save datasets
print("\n6. Saving datasets...")
np.save(processed_dir / 'X_train.npy', X_train)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'X_test.npy', X_test)
np.save(processed_dir / 'y_test.npy', y_test)

# Save class mapping
class_mapping = {
    'positive': 1,
    'negative': 0,
    'categories': {
        'penis': 1,
        'hard_negatives': negative_classes
    },
    'description': 'Binary classification: positive=penis, negative=21 QuickDraw (per-image normalized)'
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
print("  • Fixed, properly centered penis drawings")
print("  • Balanced positive/negative classes")
print("  • 80/20 train/test split")
print("  • Normalized to 0-1 range")
print("\nNext step: Run training")
print("  bash train_max_accuracy.sh")
