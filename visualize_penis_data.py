"""
Visualize the imported QuickDraw penis data.
Checks data integrity and generates a visualization PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the penis data
data_path = Path('data/processed/penis_raw_X.npy')

print("Loading penis data from:", data_path)
X_penis = np.load(data_path)

# Check data integrity
print("\n=== Data Integrity Check ===")
print(f"Shape: {X_penis.shape}")
print(f"Data type: {X_penis.dtype}")
print(f"Min value: {X_penis.min()}")
print(f"Max value: {X_penis.max()}")
print(f"Mean value: {X_penis.mean():.2f}")
print(f"Number of samples: {len(X_penis)}")

# Check if data is properly normalized
if X_penis.max() > 1.5:
    print("⚠️  Data appears to be in 0-255 range (needs normalization for model)")
else:
    print("✓ Data appears to be normalized (0-1 range)")

# Create visualization
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle(f'QuickDraw Penis Dataset Samples (Total: {len(X_penis)} images)', 
             fontsize=16, fontweight='bold')

# Display 32 random samples
np.random.seed(42)
indices = np.random.choice(len(X_penis), min(32, len(X_penis)), replace=False)

for idx, ax in enumerate(axes.flat):
    if idx < len(indices):
        sample_idx = indices[idx]
        img = X_penis[sample_idx]
        
        # Handle different data formats
        if len(img.shape) == 3:
            # If shape is (28, 28, 1), squeeze it
            img = img.squeeze()
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Sample {sample_idx}', fontsize=8)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()

# Save the visualization
output_path = 'penis_data_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# Additional statistics
print("\n=== Additional Statistics ===")
print(f"Memory size: {X_penis.nbytes / 1024 / 1024:.2f} MB")

# Check for empty/blank images
blank_threshold = 0.01 if X_penis.max() <= 1.5 else 2.5
blank_images = np.sum(X_penis.mean(axis=(1, 2)) < blank_threshold)
print(f"Potentially blank images: {blank_images} ({100*blank_images/len(X_penis):.2f}%)")

# Show pixel intensity distribution
plt.figure(figsize=(10, 6))
plt.hist(X_penis.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.title('Pixel Intensity Distribution - Penis Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Save histogram
hist_path = 'penis_data_histogram.png'
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
print(f"✓ Histogram saved to: {hist_path}")

print("\n=== Summary ===")
print("✓ Data successfully loaded and verified")
print("✓ Visualizations generated successfully")
