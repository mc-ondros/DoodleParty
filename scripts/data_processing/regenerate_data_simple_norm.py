"""
Regenerate training dataset WITHOUT per-image normalization.
This should preserve brightness differences between classes.
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

# Load fixed penis data
print("\n1. Loading penis data...")
penis_data = np.load(processed_dir / 'penis_raw_X.npy')
print(f"   ✓ Loaded {len(penis_data):,} penis samples")

# Invert to match QuickDraw format
print("   ✓ Inverting to black background")
penis_data = 255 - penis_data

# Load negative classes
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
        samples = min(1200, len(data))
        data = data[:samples]
        
        if len(data.shape) == 2:
            data = data.reshape(-1, 28, 28)
        elif len(data.shape) in [3, 4]:
            data = data.reshape(-1, 28, 28)
        
        negative_data_list.append(data)
        print(f"   ✓ {cls}: {samples:,} samples")

negative_data = np.concatenate(negative_data_list, axis=0)
print(f"\n   Total negatives: {len(negative_data):,}")

# Balance dataset
print("\n3. Balancing...")
n_positive = len(penis_data)
n_negative = len(negative_data)

if n_negative > n_positive:
    np.random.seed(42)
    indices = np.random.choice(n_negative, n_positive, replace=False)
    negative_data = negative_data[indices]
elif n_positive > n_negative:
    np.random.seed(42)
    indices = np.random.choice(n_positive, n_negative, replace=False)
    penis_data = penis_data[indices]

# Create labels
y_positive = np.ones(len(penis_data), dtype=np.float32)
y_negative = np.zeros(len(negative_data), dtype=np.float32)

# Combine and SHUFFLE
X = np.concatenate([penis_data, negative_data], axis=0)
y = np.concatenate([y_positive, y_negative], axis=0)

print(f"\n   Combined: {len(X):,} samples (50/50 split)")

# Shuffle
print("   ✓ Shuffling...")
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Normalize to 0-1 (SIMPLE normalization, no per-image)
print("\n4. Simple normalization (divide by 255)...")
X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
print(f"   ✓ Shape: {X.shape}")
print(f"   Value range: {X.min():.3f} - {X.max():.3f}")
print(f"   Mean: {X.mean():.3f}")

# Check class differences ARE PRESERVED
pos_mean = X[y == 1].mean()
neg_mean = X[y == 0].mean()
print(f"\n   Class brightness difference: {abs(pos_mean - neg_mean):.4f}")
if abs(pos_mean - neg_mean) > 0.01:
    print(f"   ✓ GOOD: Brightness signal preserved!")
else:
    print(f"   ⚠️  Classes have similar brightness")

# Split
print("\n5. Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y, shuffle=True
)

print(f"   ✓ Training: {len(X_train):,}")
print(f"   ✓ Test: {len(X_test):,}")

# Save
print("\n6. Saving...")
np.save(processed_dir / 'X_train.npy', X_train)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'X_test.npy', X_test)
np.save(processed_dir / 'y_test.npy', y_test)

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
print("This should make training much more stable!")
print("\nNext: bash test_extended_training.sh")
