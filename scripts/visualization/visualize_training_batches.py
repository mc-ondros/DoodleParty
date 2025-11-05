"""
Visualize exactly what the model sees during training - 
actual batches with augmentation applied in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*70)
print("VISUALIZING ACTUAL TRAINING BATCHES")
print("="*70)

# Load and split data (same as training)
X_train_full = np.load('data/processed/X_train.npy')
y_train_full = np.load('data/processed/y_train.npy')

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=456,
    stratify=y_train_full,
    shuffle=True
)

# Create augmentation generator (same as training)
augmentation = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    fill_mode='constant',
    cval=0.5,
)

batch_size = 32

# Create training generator
train_generator = augmentation.flow(
    X_train_split, y_train_split,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

print(f"\nGenerating visualization of actual training batches...")
print(f"Batch size: {batch_size}")

# VISUALIZATION 6: Actual Training Batches
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Actual Training Batches (What the Model Sees During Training)', 
             fontsize=16, fontweight='bold')

# Get 4 real batches from the generator
for batch_num in range(4):
    X_batch, y_batch = next(train_generator)
    
    print(f"  Batch {batch_num+1}: {len(X_batch)} samples, "
          f"{(y_batch == 1).sum()} positive, {(y_batch == 0).sum()} negative")
    
    # Show first 10 samples from this batch
    for i in range(10):
        row = batch_num
        col = i
        ax = plt.subplot(4, 10, row * 10 + col + 1)
        
        img = X_batch[i].squeeze()
        label = 'POS' if y_batch[i] > 0.5 else 'NEG'
        color = 'red' if y_batch[i] > 0.5 else 'blue'
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label, fontsize=8, color=color, fontweight='bold')
        ax.axis('off')
        
        # Add batch number on first column
        if col == 0:
            ax.text(-0.1, 0.5, f'Batch {batch_num+1}', 
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   rotation=90, va='center', ha='right')

plt.tight_layout()
plt.savefig('viz_06_actual_training_batches.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_06_actual_training_batches.png")
plt.close()

# VISUALIZATION 7: Batch-level Statistics
print("\n  Analyzing batch-level statistics over 100 batches...")

batch_stats = {
    'pos_ratio': [],
    'mean_brightness': [],
    'mean_brightness_pos': [],
    'mean_brightness_neg': [],
    'aug_intensity': []  # How much pixels changed
}

# Reset generator
train_generator = augmentation.flow(
    X_train_split, y_train_split,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

for _ in range(100):
    X_batch, y_batch = next(train_generator)
    
    batch_stats['pos_ratio'].append((y_batch > 0.5).mean())
    batch_stats['mean_brightness'].append(X_batch.mean())
    
    if (y_batch > 0.5).any():
        batch_stats['mean_brightness_pos'].append(X_batch[y_batch > 0.5].mean())
    if (y_batch <= 0.5).any():
        batch_stats['mean_brightness_neg'].append(X_batch[y_batch <= 0.5].mean())

fig = plt.figure(figsize=(20, 8))
fig.suptitle('Training Batch Statistics Over 100 Batches', 
             fontsize=16, fontweight='bold')

# Positive ratio per batch
ax1 = plt.subplot(2, 3, 1)
ax1.plot(batch_stats['pos_ratio'], 'o-', alpha=0.6)
ax1.axhline(y=0.5, color='red', linestyle='--', label='Ideal (50%)')
ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Positive Ratio')
ax1.set_title('Class Balance Per Batch')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mean brightness per batch
ax2 = plt.subplot(2, 3, 2)
ax2.plot(batch_stats['mean_brightness'], 'o-', alpha=0.6)
ax2.set_xlabel('Batch Number')
ax2.set_ylabel('Mean Brightness')
ax2.set_title('Batch Brightness Variation')
ax2.grid(True, alpha=0.3)

# Brightness by class
ax3 = plt.subplot(2, 3, 3)
ax3.plot(batch_stats['mean_brightness_pos'], 'o-', alpha=0.6, 
         color='red', label='Positive')
ax3.plot(batch_stats['mean_brightness_neg'], 'o-', alpha=0.6,
         color='blue', label='Negative')
ax3.set_xlabel('Batch Number')
ax3.set_ylabel('Mean Brightness')
ax3.set_title('Brightness by Class Across Batches')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Histograms
ax4 = plt.subplot(2, 3, 4)
ax4.hist(batch_stats['pos_ratio'], bins=20, alpha=0.7, edgecolor='black')
ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Positive Ratio')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Class Balance')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
ax5.hist(batch_stats['mean_brightness'], bins=20, alpha=0.7, 
         edgecolor='black', color='green')
ax5.set_xlabel('Mean Brightness')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Batch Brightness')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
ax6.hist(batch_stats['mean_brightness_pos'], bins=20, alpha=0.6,
         label='Positive', color='red', edgecolor='black')
ax6.hist(batch_stats['mean_brightness_neg'], bins=20, alpha=0.6,
         label='Negative', color='blue', edgecolor='black')
ax6.set_xlabel('Mean Brightness')
ax6.set_ylabel('Frequency')
ax6.set_title('Brightness Distribution by Class')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz_07_batch_statistics.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_07_batch_statistics.png")
plt.close()

# ============================================================================
# VISUALIZATION 8: Augmentation Effect on Separability
# ============================================================================
print("\n  Analyzing how augmentation affects class separability...")

# Get samples before and after augmentation
pos_idx = np.where(y_train_split == 1)[0][:100]
neg_idx = np.where(y_train_split == 0)[0][:100]

# Original samples
pos_original = X_train_split[pos_idx]
neg_original = X_train_split[neg_idx]

# Generate augmented samples
pos_aug_list = []
neg_aug_list = []

pos_gen = augmentation.flow(pos_original, batch_size=1, shuffle=False)
neg_gen = augmentation.flow(neg_original, batch_size=1, shuffle=False)

for _ in range(100):
    pos_aug_list.append(next(pos_gen)[0])
    neg_aug_list.append(next(neg_gen)[0])

pos_augmented = np.array(pos_aug_list)
neg_augmented = np.array(neg_aug_list)

fig = plt.figure(figsize=(20, 10))
fig.suptitle('How Augmentation Affects Class Separability', 
             fontsize=16, fontweight='bold')

# Brightness distributions
ax1 = plt.subplot(2, 4, 1)
pos_orig_bright = pos_original.mean(axis=(1,2,3))
neg_orig_bright = neg_original.mean(axis=(1,2,3))
ax1.hist(pos_orig_bright, bins=30, alpha=0.6, label='Pos', color='red', density=True)
ax1.hist(neg_orig_bright, bins=30, alpha=0.6, label='Neg', color='blue', density=True)
ax1.set_xlabel('Mean Brightness')
ax1.set_ylabel('Density')
ax1.set_title(f'Original Data\nΔ={abs(pos_orig_bright.mean()-neg_orig_bright.mean()):.4f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 4, 2)
pos_aug_bright = pos_augmented.mean(axis=(1,2,3))
neg_aug_bright = neg_augmented.mean(axis=(1,2,3))
ax2.hist(pos_aug_bright, bins=30, alpha=0.6, label='Pos', color='red', density=True)
ax2.hist(neg_aug_bright, bins=30, alpha=0.6, label='Neg', color='blue', density=True)
ax2.set_xlabel('Mean Brightness')
ax2.set_ylabel('Density')
ax2.set_title(f'After Augmentation\nΔ={abs(pos_aug_bright.mean()-neg_aug_bright.mean()):.4f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Std dev distributions
ax3 = plt.subplot(2, 4, 3)
pos_orig_std = pos_original.std(axis=(1,2,3))
neg_orig_std = neg_original.std(axis=(1,2,3))
ax3.hist(pos_orig_std, bins=30, alpha=0.6, label='Pos', color='red', density=True)
ax3.hist(neg_orig_std, bins=30, alpha=0.6, label='Neg', color='blue', density=True)
ax3.set_xlabel('Std Deviation')
ax3.set_ylabel('Density')
ax3.set_title('Original: Std Deviation')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 4, 4)
pos_aug_std = pos_augmented.std(axis=(1,2,3))
neg_aug_std = neg_augmented.std(axis=(1,2,3))
ax4.hist(pos_aug_std, bins=30, alpha=0.6, label='Pos', color='red', density=True)
ax4.hist(neg_aug_std, bins=30, alpha=0.6, label='Neg', color='blue', density=True)
ax4.set_xlabel('Std Deviation')
ax4.set_ylabel('Density')
ax4.set_title('Augmented: Std Deviation')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Sample comparison
ax5 = plt.subplot(2, 4, 5)
ax5.imshow(pos_original[0].squeeze(), cmap='gray', vmin=0, vmax=1)
ax5.set_title('Positive: Original')
ax5.axis('off')

ax6 = plt.subplot(2, 4, 6)
ax6.imshow(pos_augmented[0].squeeze(), cmap='gray', vmin=0, vmax=1)
ax6.set_title('Positive: Augmented')
ax6.axis('off')

ax7 = plt.subplot(2, 4, 7)
ax7.imshow(neg_original[0].squeeze(), cmap='gray', vmin=0, vmax=1)
ax7.set_title('Negative: Original')
ax7.axis('off')

ax8 = plt.subplot(2, 4, 8)
ax8.imshow(neg_augmented[0].squeeze(), cmap='gray', vmin=0, vmax=1)
ax8.set_title('Negative: Augmented')
ax8.axis('off')

plt.tight_layout()
plt.savefig('viz_08_augmentation_separability.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_08_augmentation_separability.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ BATCH VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated 3 additional visualizations:")
print("  6. viz_06_actual_training_batches.png - Real batches during training")
print("  7. viz_07_batch_statistics.png - Batch-level statistics")
print("  8. viz_08_augmentation_separability.png - Augmentation effects")
print("\nKey insights:")
print(f"  • Mean pos ratio per batch: {np.mean(batch_stats['pos_ratio']):.3f}")
print(f"  • Batch brightness std: {np.std(batch_stats['mean_brightness']):.4f}")
print(f"  • Original brightness Δ: {abs(pos_orig_bright.mean()-neg_orig_bright.mean()):.4f}")
print(f"  • Augmented brightness Δ: {abs(pos_aug_bright.mean()-neg_aug_bright.mean()):.4f}")

if abs(pos_aug_bright.mean()-neg_aug_bright.mean()) > 0.15:
    print("\n  ⚠️  Even after augmentation, brightness difference remains large!")
    print("     Model will likely use brightness as primary feature.")
else:
    print("\n  ✓ Augmentation helps reduce brightness bias")

print("\nAll 8 visualizations are ready for analysis!")
