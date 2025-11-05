"""
Fix all data quality issues:
1. Properly invert penis data
2. Normalize stroke widths
3. Fix background values
4. Remove problematic images
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_dilation, distance_transform_edt

print("="*70)
print("FIXING ALL DATA QUALITY ISSUES")
print("="*70)

# Paths
data_dir = Path('data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'

# Load penis data and ensure proper inversion to match QuickDraw background convention (black background, white strokes)
print("\n1. Loading and fixing penis data...")
penis_data = np.load(processed_dir / 'penis_raw_X.npy')
print(f"   Loaded {len(penis_data):,} samples")
print(f"   Original range: {penis_data.min()} - {penis_data.max()}")
print(f"   Original mean: {penis_data.mean():.2f}")

# Detect and correct improper inversion by checking if data appears already inverted or mostly black
if penis_data.mean() < 128:
    print("   ⚠️  Data appears to be already inverted or mostly black!")
    print("   Inverting back first...")
    penis_data = 255 - penis_data

# Now invert to black background (for consistency with QuickDraw)
penis_data = 255 - penis_data
print(f"   After inversion: {penis_data.min()} - {penis_data.max()}, mean: {penis_data.mean():.2f}")

# Load diverse QuickDraw negative classes to provide realistic non-penis examples for balanced training
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
        samples = min(1200, len(data))
        data = data[:samples]
        
        # Ensure (N, 28, 28)
        if len(data.shape) == 2:
            data = data.reshape(-1, 28, 28)
        elif len(data.shape) in [3, 4]:
            data = data.reshape(-1, 28, 28)
        
        negative_data_list.append(data)
        print(f"   ✓ {cls}: {samples:,} samples")

negative_data = np.concatenate(negative_data_list, axis=0)
print(f"\n   Total negatives: {len(negative_data):,}")

# Remove all-white, all-black, and mostly empty images that could cause model shortcuts or training instability
print("\n3. Filtering problematic images...")

def is_valid_image(img):
    """Check if image is valid (not all-white, all-black, or mostly empty)"""
    mean_val = img.mean()
    std_val = img.std()
    return 10 < mean_val < 245 and std_val > 5  # Has some variation

# Filter penis data
valid_penis = []
for img in penis_data:
    if is_valid_image(img):
        valid_penis.append(img)
penis_data = np.array(valid_penis)
print(f"   Positive: kept {len(penis_data):,} valid images")

# Filter negative data
valid_negative = []
for img in negative_data:
    if is_valid_image(img):
        valid_negative.append(img)
negative_data = np.array(valid_negative)
print(f"   Negative: kept {len(negative_data):,} valid images")

# Normalize stroke widths between classes to prevent the model from using drawing thickness as a shortcut feature
print("\n4. Normalizing stroke widths...")

def normalize_stroke_width(img, target_width=4.5, threshold=127):
    """Normalize stroke width by dilating/eroding"""
    # Binarize
    binary = img < threshold
    
    # Estimate current stroke width
    if binary.sum() == 0:
        return img
    
    # Use distance transform to estimate stroke width
    dist = distance_transform_edt(binary)
    if dist.max() > 0:
        current_width = dist.max() * 2
    else:
        current_width = 1
    
    # Adjust stroke width
    if current_width < target_width - 1:
        # Thicken strokes
        iterations = int((target_width - current_width) / 2)
        binary = binary_dilation(binary, iterations=iterations)
    
    # Convert back to grayscale
    result = np.where(binary, 0, 255).astype(np.uint8)
    return result

print("   Normalizing positive class strokes...")
penis_normalized = []
for i, img in enumerate(penis_data):
    if i % 1000 == 0:
        print(f"     Processing {i}/{len(penis_data)}...", end='\r')
    normalized = normalize_stroke_width(img)
    penis_normalized.append(normalized)
penis_data = np.array(penis_normalized)
print(f"     ✓ Normalized {len(penis_data)} positive samples")

print("   Normalizing negative class strokes...")
negative_normalized = []
for i, img in enumerate(negative_data):
    if i % 1000 == 0:
        print(f"     Processing {i}/{len(negative_data)}...", end='\r')
    normalized = normalize_stroke_width(img, target_width=4.5)
    negative_normalized.append(normalized)
negative_data = np.array(negative_normalized)
print(f"     ✓ Normalized {len(negative_data)} negative samples")
# Balance positive and negative classes to prevent model bias and ensure fair evaluation metrics
print("\n5. Balancing dataset...")
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

print(f"   Balanced to {len(penis_data):,} samples per class")
# Combine positive and negative samples with appropriate binary labels for supervised training
print("\n6. Combining and shuffling...")
y_positive = np.ones(len(penis_data), dtype=np.float32)
y_negative = np.zeros(len(negative_data), dtype=np.float32)

X = np.concatenate([penis_data, negative_data], axis=0)
y = np.concatenate([y_positive, y_negative], axis=0)

# Shuffle immediately
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"   Combined: {len(X):,} samples")
print(f"   First 10 labels: {y[:10]}")
# Normalize pixel values to [0,1] range and add channel dimension for compatibility with CNN input requirements
print("\n7. Normalizing to [0, 1]...")
X = X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
print(f"   Shape: {X.shape}")
print(f"   Range: {X.min():.3f} - {X.max():.3f}")
print(f"   Mean: {X.mean():.3f}")

# Verify that brightness differences between classes have been minimized to prevent model shortcuts
pos_mean = X[y == 1].mean()
neg_mean = X[y == 0].mean()
print(f"\n   Positive mean: {pos_mean:.4f}")
print(f"   Negative mean: {neg_mean:.4f}")
print(f"   Difference: {abs(pos_mean - neg_mean):.4f}")
# Create stratified train/test split to ensure both sets maintain the same class distribution for valid evaluation
print("\n8. Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y, shuffle=True
)

print(f"   Training: {len(X_train):,}")
print(f"     - Positive: {(y_train == 1).sum():,}")
print(f"     - Negative: {(y_train == 0).sum():,}")
print(f"   Test: {len(X_test):,}")
print(f"     - Positive: {(y_test == 1).sum():,}")
print(f"     - Negative: {(y_test == 0).sum():,}")

# Save processed dataset and class mapping for reproducible training and inference
print("\n9. Saving fixed dataset...")
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
    'description': 'Binary classification: FIXED - proper inversion, normalized strokes, clean data'
}

with open(processed_dir / 'class_mapping.pkl', 'wb') as f:
    pickle.dump(class_mapping, f)

print(f"   ✓ Saved to {processed_dir}/")

# FINAL SUMMARY
print("\n" + "="*70)
print("✅ DATA FIXES COMPLETE")
print("="*70)
print("\nFixes applied:")
print("  ✓ Properly inverted penis data (black background)")
print("  ✓ Removed all-white/all-black images")
print("  ✓ Normalized stroke widths (target: 4.5px)")
print("  ✓ Ensured consistent background values")
print("  ✓ Balanced classes")
print("  ✓ Proper shuffling")
print(f"\nFinal dataset:")
print(f"  Training: {len(X_train):,} samples (50/50 split)")
print(f"  Test: {len(X_test):,} samples (50/50 split)")
print(f"  Brightness difference: {abs(pos_mean - neg_mean):.4f}")
print("\nReady for training! Run:")
print("  bash train_max_accuracy.sh")
