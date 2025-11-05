"""
Diagnose stroke width, background fill, and inversion issues in training data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("DIAGNOSING DATA ISSUES")
print("="*70)

# Load data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

# ============================================================================
# ISSUE 1: Check for non-inverted or all-white images
# ============================================================================
print("\n1. CHECKING FOR PROBLEMATIC IMAGES:")
print("-" * 70)

def is_mostly_white(img, threshold=0.9):
    """Check if image is mostly white (mean > threshold)"""
    return img.mean() > threshold

def is_mostly_black(img, threshold=0.1):
    """Check if image is mostly black (mean < threshold)"""
    return img.mean() < threshold

# Check positive class
pos_white = [i for i in pos_idx if is_mostly_white(X_train[i])]
pos_black = [i for i in pos_idx if is_mostly_black(X_train[i])]

print(f"Positive class:")
print(f"  All-white images: {len(pos_white)} ({100*len(pos_white)/len(pos_idx):.2f}%)")
print(f"  All-black images: {len(pos_black)} ({100*len(pos_black)/len(pos_idx):.2f}%)")

# Check negative class
neg_white = [i for i in neg_idx if is_mostly_white(X_train[i])]
neg_black = [i for i in neg_idx if is_mostly_black(X_train[i])]

print(f"Negative class:")
print(f"  All-white images: {len(neg_white)} ({100*len(neg_white)/len(neg_idx):.2f}%)")
print(f"  All-black images: {len(neg_black)} ({100*len(neg_black)/len(neg_idx):.2f}%)")

# ============================================================================
# ISSUE 2: Analyze stroke width
# ============================================================================
print("\n2. ANALYZING STROKE WIDTH:")
print("-" * 70)

from scipy.ndimage import binary_erosion, binary_dilation

def estimate_stroke_width(img, threshold=0.5):
    """Estimate average stroke width using morphological operations"""
    # Binarize
    binary = img < threshold
    
    # Erode until most strokes disappear
    eroded = binary.copy()
    erosions = 0
    while eroded.sum() > binary.sum() * 0.1 and erosions < 10:
        eroded = binary_erosion(eroded)
        erosions += 1
    
    return erosions

# Sample and estimate stroke widths
n_samples = 100
pos_strokes = [estimate_stroke_width(X_train[i].squeeze()) 
               for i in pos_idx[:n_samples]]
neg_strokes = [estimate_stroke_width(X_train[i].squeeze()) 
               for i in neg_idx[:n_samples]]

print(f"Positive class stroke width: {np.mean(pos_strokes):.2f} ± {np.std(pos_strokes):.2f}")
print(f"Negative class stroke width: {np.mean(neg_strokes):.2f} ± {np.std(neg_strokes):.2f}")
print(f"Difference: {abs(np.mean(pos_strokes) - np.mean(neg_strokes)):.2f}")

# ============================================================================
# ISSUE 3: Background value distribution
# ============================================================================
print("\n3. CHECKING BACKGROUND VALUES:")
print("-" * 70)

# Sample corner pixels (likely background)
def get_background_value(img):
    """Estimate background by sampling corners"""
    corners = [
        img[0:3, 0:3].mean(),
        img[0:3, -3:].mean(),
        img[-3:, 0:3].mean(),
        img[-3:, -3:].mean()
    ]
    return np.mean(corners)

pos_bg = [get_background_value(X_train[i].squeeze()) for i in pos_idx[:100]]
neg_bg = [get_background_value(X_train[i].squeeze()) for i in neg_idx[:100]]

print(f"Positive class background: {np.mean(pos_bg):.3f} ± {np.std(pos_bg):.3f}")
print(f"Negative class background: {np.mean(neg_bg):.3f} ± {np.std(neg_bg):.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n4. CREATING DIAGNOSTIC VISUALIZATION...")

fig = plt.figure(figsize=(20, 14))
fig.suptitle('Data Quality Issues Diagnosis', fontsize=16, fontweight='bold')

# Row 1: Problematic positive samples
if len(pos_white) > 0 or len(pos_black) > 0:
    for i in range(min(10, max(len(pos_white), len(pos_black)))):
        ax = plt.subplot(5, 10, i + 1)
        if i < len(pos_white):
            idx = pos_white[i]
            img = X_train[idx].squeeze()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'WHITE\n{img.mean():.2f}', fontsize=7, color='red')
        elif i - len(pos_white) < len(pos_black):
            idx = pos_black[i - len(pos_white)]
            img = X_train[idx].squeeze()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'BLACK\n{img.mean():.2f}', fontsize=7, color='red')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('POS\nIssues', fontsize=9, fontweight='bold',
                         rotation=0, ha='right', va='center')

# Row 2: Problematic negative samples  
if len(neg_white) > 0 or len(neg_black) > 0:
    for i in range(min(10, max(len(neg_white), len(neg_black)))):
        ax = plt.subplot(5, 10, 10 + i + 1)
        if i < len(neg_white):
            idx = neg_white[i]
            img = X_train[idx].squeeze()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'WHITE\n{img.mean():.2f}', fontsize=7, color='red')
        elif i - len(neg_white) < len(neg_black):
            idx = neg_black[i - len(neg_white)]
            img = X_train[idx].squeeze()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'BLACK\n{img.mean():.2f}', fontsize=7, color='red')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('NEG\nIssues', fontsize=9, fontweight='bold',
                         rotation=0, ha='right', va='center')

# Row 3: Stroke width comparison
for i in range(10):
    ax = plt.subplot(5, 10, 20 + i + 1)
    idx = pos_idx[i]
    img = X_train[idx].squeeze()
    stroke = estimate_stroke_width(img)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'POS\nStroke:{stroke}', fontsize=7)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('POS\nStroke', fontsize=9, fontweight='bold',
                     rotation=0, ha='right', va='center')

# Row 4: Negative stroke widths
for i in range(10):
    ax = plt.subplot(5, 10, 30 + i + 1)
    idx = neg_idx[i]
    img = X_train[idx].squeeze()
    stroke = estimate_stroke_width(img)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'NEG\nStroke:{stroke}', fontsize=7)
    ax.axis('off')
    if i == 0:
        ax.set_ylabel('NEG\nStroke', fontsize=9, fontweight='bold',
                     rotation=0, ha='right', va='center')

# Row 5: Statistics
ax = plt.subplot(5, 2, 9)
ax.hist(pos_strokes, bins=15, alpha=0.6, label='Positive', color='red', edgecolor='black')
ax.hist(neg_strokes, bins=15, alpha=0.6, label='Negative', color='blue', edgecolor='black')
ax.set_xlabel('Stroke Width (pixels)')
ax.set_ylabel('Count')
ax.set_title('Stroke Width Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = plt.subplot(5, 2, 10)
ax.hist(pos_bg, bins=20, alpha=0.6, label='Positive', color='red', edgecolor='black')
ax.hist(neg_bg, bins=20, alpha=0.6, label='Negative', color='blue', edgecolor='black')
ax.set_xlabel('Background Value')
ax.set_ylabel('Count')
ax.set_title('Background Value Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz_09_data_quality_issues.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: viz_09_data_quality_issues.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF ISSUES")
print("="*70)

issues_found = []

if len(pos_white) > 10 or len(neg_white) > 10:
    issues_found.append(f"⚠️  All-white images: {len(pos_white)} pos, {len(neg_white)} neg")
if len(pos_black) > 10 or len(neg_black) > 10:
    issues_found.append(f"⚠️  All-black images: {len(pos_black)} pos, {len(neg_black)} neg")
if abs(np.mean(pos_strokes) - np.mean(neg_strokes)) > 1.0:
    issues_found.append(f"⚠️  Stroke width difference: {abs(np.mean(pos_strokes) - np.mean(neg_strokes)):.2f} pixels")
if abs(np.mean(pos_bg) - np.mean(neg_bg)) > 0.1:
    issues_found.append(f"⚠️  Background value difference: {abs(np.mean(pos_bg) - np.mean(neg_bg)):.3f}")

if issues_found:
    print("\nISSUES FOUND:")
    for issue in issues_found:
        print(f"  {issue}")
    print("\nRECOMMENDED FIXES:")
    print("  1. Re-process penis data with proper inversion")
    print("  2. Normalize stroke widths (dilate thinner strokes)")
    print("  3. Ensure consistent background values")
    print("  4. Remove all-white/all-black images")
else:
    print("\n✓ No major issues detected!")

print("\nVisualization saved: viz_09_data_quality_issues.png")
