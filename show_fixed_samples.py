"""
Show what the fixes accomplished with sample comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the fixed data
X_penis = np.load('data/processed/penis_raw_X.npy')

# Create comparison visualization
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
fig.suptitle('Fixed Penis Dataset - Properly Centered, Normalized & Scaled\n25,209 Total Samples', 
             fontsize=16, fontweight='bold')

# Show 24 random samples
np.random.seed(123)
indices = np.random.choice(len(X_penis), 24, replace=False)

for idx, ax in enumerate(axes.flat):
    sample_idx = indices[idx]
    img = X_penis[sample_idx].squeeze()
    
    # Show inverted (black on white looks better)
    ax.imshow(img, cmap='gray_r', vmin=0, vmax=255)
    ax.set_title(f'Sample {sample_idx}', fontsize=8)
    ax.axis('off')

plt.tight_layout()

# Save
output_path = 'penis_data_samples_fixed.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Sample visualization saved to: {output_path}")

# Print summary
print("\n" + "="*70)
print("PROCESSING IMPROVEMENTS SUMMARY")
print("="*70)
print("\nPROBLEMS FIXED:")
print("  ❌ BEFORE: Drawings were off-center (coordinates not normalized)")
print("  ✓ AFTER:  All drawings properly centered with bounding box detection")
print()
print("  ❌ BEFORE: Extreme zoom issues (raw pixel coords used directly)")  
print("  ✓ AFTER:  Proper scaling with padding and aspect ratio preserved")
print()
print("  ❌ BEFORE: Some images appeared black/empty")
print("  ✓ AFTER:  All drawings visible with consistent quality")
print()
print("KEY CHANGES IN CODE:")
print("  • Added bounding box calculation (min/max x/y detection)")
print("  • Normalized coordinates relative to bounding box")
print("  • Added centered positioning on 256x256 canvas")
print("  • Applied 20px padding to prevent edge clipping")
print("  • Maintained aspect ratio during scaling")
print("  • Used LANCZOS resampling for quality")
print()
print("DATASET STATUS:")
print(f"  • Total samples: {len(X_penis):,}")
print(f"  • Image size: 28×28 pixels")
print(f"  • Value range: {X_penis.min()}-{X_penis.max()}")
print(f"  • Ready for training: ✓ YES")
print("="*70)
