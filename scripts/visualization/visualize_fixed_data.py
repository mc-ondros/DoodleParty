"""
Visualize the fixes: show before/after comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("BEFORE/AFTER FIX COMPARISON")
print("="*70)

# Load new fixed data
X_train_new = np.load('data/processed/X_train.npy')
y_train_new = np.load('data/processed/y_train.npy')

print(f"\nNew dataset:")
print(f"  Training samples: {len(X_train_new):,}")
print(f"  Shape: {X_train_new.shape}")
print(f"  Range: {X_train_new.min():.3f} - {X_train_new.max():.3f}")
print(f"  Mean: {X_train_new.mean():.3f}")

pos_idx = np.where(y_train_new == 1)[0]
neg_idx = np.where(y_train_new == 0)[0]

# ============================================================================
# VISUALIZATION: Show Fixed Samples
# ============================================================================
print("\n1. Creating visualization of fixed samples...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('FIXED Training Data - Quality Improvements', 
             fontsize=16, fontweight='bold')

# Row 1-2: Fixed positive samples
for i in range(20):
    ax = plt.subplot(6, 10, i + 1)
    idx = pos_idx[np.random.randint(len(pos_idx))]
    img = X_train_new[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(f'{img.mean():.2f}', fontsize=7)
    if i == 0:
        ax.set_ylabel('POSITIVE\n(Fixed)', fontsize=10, fontweight='bold',
                     rotation=0, ha='right', va='center')

# Row 3-4: Fixed negative samples
for i in range(20):
    ax = plt.subplot(6, 10, 20 + i + 1)
    idx = neg_idx[np.random.randint(len(neg_idx))]
    img = X_train_new[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(f'{img.mean():.2f}', fontsize=7)
    if i == 0:
        ax.set_ylabel('NEGATIVE\n(Fixed)', fontsize=10, fontweight='bold',
                     rotation=0, ha='right', va='center')

# Row 5: Augmentation test with correct background
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Calculate background value
sample_corners = []
for i in np.random.choice(len(X_train_new), 100, replace=False):
    img = X_train_new[i].squeeze()
    corners = [
        img[0:2, 0:2].mean(),
        img[0:2, -2:].mean(),
        img[-2:, 0:2].mean(),
        img[-2:, -2:].mean()
    ]
    sample_corners.extend(corners)
background_value = np.median(sample_corners)

print(f"   Background value for augmentation: {background_value:.3f}")

# Create augmentation with CORRECT background
augmentation = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    fill_mode='constant',
    cval=background_value
)

# Show augmentation examples
idx = pos_idx[5]
original = X_train_new[idx:idx+1]

ax = plt.subplot(6, 10, 41)
ax.imshow(original.squeeze(), cmap='gray', vmin=0, vmax=1)
ax.set_title('Original', fontweight='bold', fontsize=8)
ax.axis('off')
if True:
    ax.set_ylabel('AUGMENTED\n(Fixed BG)', fontsize=10, fontweight='bold',
                 rotation=0, ha='right', va='center')

gen = augmentation.flow(original, batch_size=1, shuffle=False)
for i in range(9):
    ax = plt.subplot(6, 10, 41 + i + 1)
    aug_img = next(gen)[0]
    ax.imshow(aug_img.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Aug {i+1}', fontsize=7)
    ax.axis('off')

# Row 6: Statistics comparison
ax = plt.subplot(6, 3, 16)
pos_means = X_train_new[y_train_new == 1].mean(axis=(1,2,3))
neg_means = X_train_new[y_train_new == 0].mean(axis=(1,2,3))
ax.hist(pos_means, bins=50, alpha=0.6, label='Positive', color='red', density=True)
ax.hist(neg_means, bins=50, alpha=0.6, label='Negative', color='blue', density=True)
ax.set_xlabel('Mean Brightness')
ax.set_ylabel('Density')
ax.set_title(f'Fixed Data\nΔ={abs(pos_means.mean()-neg_means.mean()):.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(6, 3, 17)
ax.axis('off')
summary = f"""
FIXES APPLIED:

✓ Properly inverted penis data
✓ Removed all-white images
✓ Removed all-black images  
✓ Normalized stroke widths
✓ Consistent backgrounds

RESULTS:
• Training samples: {len(X_train_new):,}
• Brightness diff: {abs(pos_means.mean()-neg_means.mean()):.4f}
• Background fill: {background_value:.3f}
• Stroke widths: normalized

Ready for stable training!
"""
ax.text(0.1, 0.5, summary, fontsize=9, family='monospace',
       verticalalignment='center')

ax = plt.subplot(6, 3, 18)
# Show mean images
pos_mean_img = X_train_new[y_train_new == 1].mean(axis=0).squeeze()
neg_mean_img = X_train_new[y_train_new == 0].mean(axis=0).squeeze()
diff_img = pos_mean_img - neg_mean_img
im = ax.imshow(diff_img, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
ax.set_title('Difference Map\n(Positive - Negative)')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('viz_10_fixed_data.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_10_fixed_data.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE")
print("="*70)
print("\nKey improvements:")
print(f"  • Dataset size: {len(X_train_new):,} samples")
print(f"  • Brightness difference: {abs(pos_means.mean()-neg_means.mean()):.4f}")
print(f"  • Background value: {background_value:.3f} (used for augmentation)")
print(f"  • All problematic images removed")
print(f"  • Stroke widths normalized")
print("\nVisualization saved: viz_10_fixed_data.png")
print("\nThe data is now clean and ready for training!")
print("Run: bash train_max_accuracy.sh")
