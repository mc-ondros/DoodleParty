"""
Comprehensive visualization of training data to diagnose issues.
Shows original samples, augmented samples, and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

print("="*70)
print("COMPREHENSIVE TRAINING DATA VISUALIZATION")
print("="*70)

# Load processed training data to visualize the actual samples the model will see during training
X_train_full = np.load('data/processed/X_train.npy')
y_train_full = np.load('data/processed/y_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Replicate the exact train/validation split used during model training to ensure visualization matches actual training conditions
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=456,
    stratify=y_train_full,
    shuffle=True
)

print(f"\nDataset sizes:")
print(f"  Training: {len(X_train_split):,}")
print(f"  Validation: {len(X_val_split):,}")
print(f"  Test: {len(X_test):,}")

# VISUALIZATION 1: Original Samples Grid
print("\n1. Creating original samples grid...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Training Data: Original Samples (Positive=Penis, Negative=QuickDraw)', 
             fontsize=16, fontweight='bold')

# Separate indices for positive and negative classes to enable class-specific sampling and analysis
pos_idx = np.where(y_train_split == 1)[0]
neg_idx = np.where(y_train_split == 0)[0]

# Positive samples (4 rows)
for i in range(40):
    ax = plt.subplot(8, 10, i + 1)
    idx = pos_idx[np.random.randint(len(pos_idx))]
    img = X_train_split[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if i % 10 == 0:
        ax.set_ylabel('POSITIVE', fontsize=10, fontweight='bold', rotation=0, 
                     ha='right', va='center')

# Negative samples (4 rows)
for i in range(40):
    ax = plt.subplot(8, 10, 40 + i + 1)
    idx = neg_idx[np.random.randint(len(neg_idx))]
    img = X_train_split[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if i % 10 == 0:
        ax.set_ylabel('NEGATIVE', fontsize=10, fontweight='bold', rotation=0,
                     ha='right', va='center')

plt.tight_layout()
plt.savefig('viz_01_original_samples.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_01_original_samples.png")
plt.close()

# VISUALIZATION 2: Augmented Samples (Aggressive)
print("\n2. Creating augmented samples visualization...")

# Configure augmentation generators that match the exact parameters used during model training to visualize realistic augmented samples
augmentation_aggressive = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    fill_mode='constant',
    cval=0.5,
)

augmentation_standard = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    fill_mode='constant',
    cval=0.5,
)

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Data Augmentation Examples (Same Sample, Different Augmentations)', 
             fontsize=16, fontweight='bold')

# Display positive class samples with aggressive augmentation to show how penis drawings are transformed during training
for sample_num in range(2):
    # Pick a positive sample
    idx = pos_idx[sample_num]
    original = X_train_split[idx:idx+1]
    
    # Original
    row = sample_num
    ax = plt.subplot(4, 10, row * 10 + 1)
    ax.imshow(original.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Original', fontsize=8, fontweight='bold')
    ax.axis('off')
    
    # Generate 9 augmented versions (aggressive)
    aug_gen = augmentation_aggressive.flow(original, batch_size=1, shuffle=False)
    for i in range(9):
        ax = plt.subplot(4, 10, row * 10 + i + 2)
        aug_img = next(aug_gen)[0]
        ax.imshow(aug_img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Aug {i+1}', fontsize=8)
        ax.axis('off')

# Display negative class samples with aggressive augmentation to show how QuickDraw drawings are transformed during training for fair comparison
for sample_num in range(2):
    idx = neg_idx[sample_num]
    original = X_train_split[idx:idx+1]
    
    # Original
    row = sample_num + 2
    ax = plt.subplot(4, 10, row * 10 + 1)
    ax.imshow(original.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Original', fontsize=8, fontweight='bold')
    ax.axis('off')
    
    # Generate 9 augmented versions
    aug_gen = augmentation_aggressive.flow(original, batch_size=1, shuffle=False)
    for i in range(9):
        ax = plt.subplot(4, 10, row * 10 + i + 2)
        aug_img = next(aug_gen)[0]
        ax.imshow(aug_img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Aug {i+1}', fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig('viz_02_augmented_samples.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_02_augmented_samples.png")
plt.close()

# VISUALIZATION 3: Statistical Analysis
print("\n3. Creating statistical analysis...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Statistical Analysis of Training Data', fontsize=16, fontweight='bold')

# 3.1: Pixel intensity distributions
ax1 = plt.subplot(3, 4, 1)
pos_samples = X_train_split[y_train_split == 1].flatten()
neg_samples = X_train_split[y_train_split == 0].flatten()
ax1.hist(pos_samples, bins=50, alpha=0.6, label='Positive', color='red', density=True)
ax1.hist(neg_samples, bins=50, alpha=0.6, label='Negative', color='blue', density=True)
ax1.set_xlabel('Pixel Intensity')
ax1.set_ylabel('Density')
ax1.set_title('Pixel Intensity Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2: Mean brightness per image
ax2 = plt.subplot(3, 4, 2)
pos_means = X_train_split[y_train_split == 1].mean(axis=(1,2,3))
neg_means = X_train_split[y_train_split == 0].mean(axis=(1,2,3))
ax2.hist(pos_means, bins=50, alpha=0.6, label='Positive', color='red', density=True)
ax2.hist(neg_means, bins=50, alpha=0.6, label='Negative', color='blue', density=True)
ax2.set_xlabel('Mean Brightness')
ax2.set_ylabel('Density')
ax2.set_title(f'Per-Image Mean Brightness\nΔ={abs(pos_means.mean()-neg_means.mean()):.4f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3: Std deviation per image
ax3 = plt.subplot(3, 4, 3)
pos_stds = X_train_split[y_train_split == 1].std(axis=(1,2,3))
neg_stds = X_train_split[y_train_split == 0].std(axis=(1,2,3))
ax3.hist(pos_stds, bins=50, alpha=0.6, label='Positive', color='red', density=True)
ax3.hist(neg_stds, bins=50, alpha=0.6, label='Negative', color='blue', density=True)
ax3.set_xlabel('Std Deviation')
ax3.set_ylabel('Density')
ax3.set_title('Per-Image Std Deviation')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4: Ink density (proportion of dark pixels)
ax4 = plt.subplot(3, 4, 4)
threshold = 0.3
pos_ink = (X_train_split[y_train_split == 1] < threshold).mean(axis=(1,2,3))
neg_ink = (X_train_split[y_train_split == 0] < threshold).mean(axis=(1,2,3))
ax4.hist(pos_ink, bins=50, alpha=0.6, label='Positive', color='red', density=True)
ax4.hist(neg_ink, bins=50, alpha=0.6, label='Negative', color='blue', density=True)
ax4.set_xlabel('Ink Density')
ax4.set_ylabel('Density')
ax4.set_title(f'Ink Density (pixels < {threshold})\nΔ={abs(pos_ink.mean()-neg_ink.mean()):.4f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 3.5-3.8: Mean images
ax5 = plt.subplot(3, 4, 5)
pos_mean_img = X_train_split[y_train_split == 1].mean(axis=0).squeeze()
im5 = ax5.imshow(pos_mean_img, cmap='gray', vmin=0, vmax=1)
ax5.set_title(f'Positive Mean Image\n(avg={pos_mean_img.mean():.3f})')
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = plt.subplot(3, 4, 6)
neg_mean_img = X_train_split[y_train_split == 0].mean(axis=0).squeeze()
im6 = ax6.imshow(neg_mean_img, cmap='gray', vmin=0, vmax=1)
ax6.set_title(f'Negative Mean Image\n(avg={neg_mean_img.mean():.3f})')
ax6.axis('off')
plt.colorbar(im6, ax=ax6, fraction=0.046)

ax7 = plt.subplot(3, 4, 7)
diff_img = pos_mean_img - neg_mean_img
im7 = ax7.imshow(diff_img, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax7.set_title('Difference\n(Positive - Negative)')
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046)

ax8 = plt.subplot(3, 4, 8)
pos_std_img = X_train_split[y_train_split == 1].std(axis=0).squeeze()
im8 = ax8.imshow(pos_std_img, cmap='hot')
ax8.set_title('Positive Std Dev Map')
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046)

# 3.9: Class balance across splits
ax9 = plt.subplot(3, 4, 9)
splits = ['Train', 'Val', 'Test']
pos_counts = [
    (y_train_split == 1).sum(),
    (y_val_split == 1).sum(),
    (y_test == 1).sum()
]
neg_counts = [
    (y_train_split == 0).sum(),
    (y_val_split == 0).sum(),
    (y_test == 0).sum()
]
x = np.arange(len(splits))
width = 0.35
ax9.bar(x - width/2, pos_counts, width, label='Positive', color='red', alpha=0.7)
ax9.bar(x + width/2, neg_counts, width, label='Negative', color='blue', alpha=0.7)
ax9.set_ylabel('Count')
ax9.set_title('Class Balance Across Splits')
ax9.set_xticks(x)
ax9.set_xticklabels(splits)
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

# 3.10: Edge detection comparison
ax10 = plt.subplot(3, 4, 10)
from scipy.ndimage import sobel
pos_sample = X_train_split[pos_idx[0]].squeeze()
pos_edges = np.hypot(sobel(pos_sample, axis=0), sobel(pos_sample, axis=1))
ax10.imshow(pos_edges, cmap='gray')
ax10.set_title('Positive: Edge Density')
ax10.axis('off')

ax11 = plt.subplot(3, 4, 11)
neg_sample = X_train_split[neg_idx[0]].squeeze()
neg_edges = np.hypot(sobel(neg_sample, axis=0), sobel(neg_sample, axis=1))
ax11.imshow(neg_edges, cmap='gray')
ax11.set_title('Negative: Edge Density')
ax11.axis('off')

# 3.11: Summary statistics
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Positive Class:
  Mean brightness: {pos_means.mean():.4f} ± {pos_means.std():.4f}
  Mean std dev: {pos_stds.mean():.4f} ± {pos_stds.std():.4f}
  Mean ink density: {pos_ink.mean():.4f} ± {pos_ink.std():.4f}
  
Negative Class:
  Mean brightness: {neg_means.mean():.4f} ± {neg_means.std():.4f}
  Mean std dev: {neg_stds.mean():.4f} ± {neg_stds.std():.4f}
  Mean ink density: {neg_ink.mean():.4f} ± {neg_ink.std():.4f}

Separability:
  Brightness diff: {abs(pos_means.mean()-neg_means.mean()):.4f}
  Std dev diff: {abs(pos_stds.mean()-neg_stds.mean()):.4f}
  Ink density diff: {abs(pos_ink.mean()-neg_ink.mean()):.4f}

⚠️ Large differences indicate potential shortcuts!
"""
ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('viz_03_statistical_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_03_statistical_analysis.png")
plt.close()

# VISUALIZATION 4: Augmentation Comparison
print("\n4. Creating augmentation strength comparison...")

fig = plt.figure(figsize=(20, 8))
fig.suptitle('Augmentation Strength Comparison: Standard vs Aggressive', 
             fontsize=16, fontweight='bold')

# Select a representative positive sample to demonstrate the difference between standard and aggressive augmentation strategies
idx = pos_idx[5]
original = X_train_split[idx:idx+1]

# Show the original un-augmented sample as baseline for comparison
ax = plt.subplot(3, 10, 1)
ax.imshow(original.squeeze(), cmap='gray', vmin=0, vmax=1)
ax.set_title('Original', fontweight='bold')
ax.axis('off')

# Show standard augmentation results that represent typical training-time transformations
gen_std = augmentation_standard.flow(original, batch_size=1, shuffle=False)
for i in range(9):
    ax = plt.subplot(3, 10, i + 2)
    aug_img = next(gen_std)[0]
    ax.imshow(aug_img.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Std {i+1}', fontsize=8)
    ax.axis('off')

ax = plt.subplot(3, 10, 11)
ax.imshow(original.squeeze(), cmap='gray', vmin=0, vmax=1)
ax.set_title('Original', fontweight='bold')
ax.axis('off')

# Show aggressive augmentation results that represent the full range of possible training-time transformations
gen_agg = augmentation_aggressive.flow(original, batch_size=1, shuffle=False)
for i in range(9):
    ax = plt.subplot(3, 10, 11 + i + 2)
    aug_img = next(gen_agg)[0]
    ax.imshow(aug_img.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Agg {i+1}', fontsize=8)
    ax.axis('off')

# Add text labels
fig.text(0.02, 0.67, 'STANDARD\nAugmentation', fontsize=12, fontweight='bold',
         rotation=90, va='center', ha='center')
fig.text(0.02, 0.33, 'AGGRESSIVE\nAugmentation', fontsize=12, fontweight='bold',
         rotation=90, va='center', ha='center')

plt.tight_layout()
plt.savefig('viz_04_augmentation_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_04_augmentation_comparison.png")
plt.close()

# VISUALIZATION 5: Difficult Cases
print("\n5. Creating difficult cases visualization...")

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Potentially Difficult Cases for the Model', fontsize=16, fontweight='bold')

# Find edge cases
# Simple/minimal positives
pos_ink_density = (X_train_split[y_train_split == 1] < 0.3).mean(axis=(1,2,3))
simple_pos_idx = pos_idx[np.argsort(pos_ink_density)[:20]]

# Complex/dense negatives
neg_ink_density = (X_train_split[y_train_split == 0] < 0.3).mean(axis=(1,2,3))
complex_neg_idx = neg_idx[np.argsort(-neg_ink_density)[:20]]

# Show simple positives
for i in range(20):
    ax = plt.subplot(4, 10, i + 1)
    idx = simple_pos_idx[i]
    img = X_train_split[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('Simple\nPositives', fontsize=10, fontweight='bold',
                     rotation=0, ha='right', va='center')

# Show complex negatives
for i in range(20):
    ax = plt.subplot(4, 10, 20 + i + 1)
    idx = complex_neg_idx[i]
    img = X_train_split[idx].squeeze()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('Complex\nNegatives', fontsize=10, fontweight='bold',
                     rotation=0, ha='right', va='center')

plt.tight_layout()
plt.savefig('viz_05_difficult_cases.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_05_difficult_cases.png")
plt.close()

# FINAL REPORT
print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated 5 visualizations:")
print("  1. viz_01_original_samples.png - Grid of original training samples")
print("  2. viz_02_augmented_samples.png - Augmentation examples")
print("  3. viz_03_statistical_analysis.png - Detailed statistics")
print("  4. viz_04_augmentation_comparison.png - Standard vs Aggressive aug")
print("  5. viz_05_difficult_cases.png - Edge cases and difficult samples")
print("\nKey findings:")
print(f"  • Brightness difference: {abs(pos_means.mean()-neg_means.mean()):.4f}")
print(f"  • Ink density difference: {abs(pos_ink.mean()-neg_ink.mean()):.4f}")
if abs(pos_means.mean()-neg_means.mean()) > 0.05:
    print("  ⚠️  WARNING: Large brightness difference - model may use shortcuts!")
if abs(pos_ink.mean()-neg_ink.mean()) > 0.1:
    print("  ⚠️  WARNING: Large ink density difference - model may use shortcuts!")
print("\nReview the images to identify potential issues with the data!")
