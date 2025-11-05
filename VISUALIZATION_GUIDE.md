# Training Data Visualization Guide

Generated on November 4, 2025

## 📊 Generated Visualizations (8 files)

### 1. **viz_01_original_samples.png** (134 KB)
**Grid of Original Training Samples**

Shows 80 random samples from the training set:
- Top 4 rows: Positive class (penis drawings)
- Bottom 4 rows: Negative class (QuickDraw objects)

**What to look for:**
- Visual similarity within each class
- Clear differences between classes
- Quality and consistency of drawings
- Any mislabeled samples

---

### 2. **viz_02_augmented_samples.png** (95 KB)
**Data Augmentation Examples**

Shows the same original sample transformed 9 different ways using aggressive augmentation:
- Rows 1-2: Positive samples with augmentation
- Rows 3-4: Negative samples with augmentation

**Augmentation parameters:**
- Rotation: ±25°
- Shift: ±15% horizontal/vertical
- Zoom: ±20%

**What to look for:**
- Whether augmented samples still look realistic
- If shapes are preserved after transformation
- Potential issues with over-aggressive augmentation

---

### 3. **viz_03_statistical_analysis.png** (288 KB) ⭐ MOST IMPORTANT
**Detailed Statistical Analysis**

12 subplots showing comprehensive statistics:

1. **Pixel Intensity Distribution** - Overall pixel value distribution
2. **Mean Brightness** - Per-image brightness (⚠️ Δ=0.1969 - LARGE!)
3. **Std Deviation** - Per-image variation
4. **Ink Density** - Proportion of dark pixels (⚠️ Δ=0.2213 - LARGE!)
5. **Positive Mean Image** - Average of all positive samples
6. **Negative Mean Image** - Average of all negative samples
7. **Difference Map** - Shows where classes differ spatially
8. **Positive Std Dev Map** - Variation across positive samples
9. **Class Balance** - Distribution across train/val/test splits
10. **Edge Density (Positive)** - Edge detection on positive sample
11. **Edge Density (Negative)** - Edge detection on negative sample
12. **Summary Statistics** - Numerical summary

**⚠️ KEY FINDINGS:**
- Brightness difference: **0.1969** (very high!)
- Ink density difference: **0.2213** (very high!)
- Model will likely use these shortcuts instead of learning shapes

---

### 4. **viz_04_augmentation_comparison.png** (77 KB)
**Standard vs Aggressive Augmentation**

Compares two augmentation strategies:
- Row 2: Standard augmentation (±15° rotation, ±10% shift, ±15% zoom)
- Row 3: Aggressive augmentation (±25° rotation, ±15% shift, ±20% zoom)

**What to look for:**
- Which augmentation level preserves shape better
- Whether aggressive augmentation creates unrealistic samples

---

### 5. **viz_05_difficult_cases.png** (73 KB)
**Edge Cases and Difficult Samples**

Shows potentially problematic samples:
- Top 2 rows: Simple/minimal positive samples (low ink density)
- Bottom 2 rows: Complex/dense negative samples (high ink density)

**What to look for:**
- Ambiguous samples that could confuse the model
- Simple positives that might be hard to recognize
- Complex negatives that might look similar to positives

---

### 6. **viz_06_actual_training_batches.png** (119 KB)
**Real Training Batches**

Shows exactly what the model sees during training - 4 actual batches with augmentation applied:
- Each row is one batch of 10 samples (out of 32)
- Labels show POS (positive) or NEG (negative)

**What to look for:**
- Class balance within batches (should be ~50/50)
- Quality of augmented samples
- Whether samples are recognizable after augmentation

---

### 7. **viz_07_batch_statistics.png** (422 KB)
**Batch-Level Statistics Over 100 Batches**

Analyzes stability and consistency across batches:

1. **Class Balance Per Batch** - Should hover around 50%
2. **Batch Brightness Variation** - How much brightness varies
3. **Brightness by Class** - Separate tracking of positive/negative
4. **Distribution of Class Balance** - Histogram of pos/neg ratios
5. **Distribution of Batch Brightness** - Overall brightness variation
6. **Brightness Distribution by Class** - Comparison histogram

**Key metrics:**
- Mean positive ratio: **0.514** (good - close to 50%)
- Batch brightness std: **0.0294** (low variation = consistent)

---

### 8. **viz_08_augmentation_separability.png** (113 KB) ⭐ IMPORTANT
**How Augmentation Affects Class Separability**

Shows whether augmentation reduces the "shortcut" signals:

Top row:
- Original data brightness distributions
- Augmented data brightness distributions
- Original std deviation
- Augmented std deviation

Bottom row:
- Example positive samples (original vs augmented)
- Example negative samples (original vs augmented)

**⚠️ CRITICAL FINDING:**
- Original brightness Δ: **0.1997**
- Augmented brightness Δ: **0.1809**
- **Augmentation only reduces difference by 9%!**
- Model will STILL use brightness as primary feature

---

## 🎯 Key Insights from Analysis

### Problem Identified
Your training data has **strong shortcuts** that make the task artificially easy:

1. **Brightness Bias** (Δ=0.1969)
   - Positive class is darker (mean=0.0439)
   - Negative class is lighter (mean=0.2408)
   - Model can achieve high accuracy just by checking brightness!

2. **Ink Density Bias** (Δ=0.2213)
   - Positive class has more ink (83.9% coverage)
   - Negative class has less ink (61.8% coverage)
   - Another easy shortcut for the model

3. **Augmentation Doesn't Help Enough**
   - Even with aggressive augmentation, brightness difference remains large
   - Model learns shortcuts instead of actual shape features

### Why This Causes Validation Bouncing

When the model relies on brightness/density shortcuts:
- It achieves high accuracy quickly (96-99% in 1-2 epochs)
- But these features are fragile and don't generalize well
- Small weight changes → predictions flip entirely
- Validation accuracy crashes to 50% (random guessing)
- Model eventually recovers but keeps bouncing

### Recommended Solutions

1. **Use Simple Normalization** ✅ (Already implemented)
   - Run: `python regenerate_data_simple_norm.py`
   - This preserves brightness differences but makes them less extreme

2. **Remove Per-Image Normalization** ✅ (Already implemented)
   - The aggressive normalization removed too much signal

3. **Use Smaller Model** ⚠️ TODO
   - Use standard model (423K params) instead of enhanced (1.2M params)
   - Remove `--enhanced` flag from training

4. **Accept the Shortcuts** 💡
   - If your production use case ALSO has these differences, it's actually good!
   - The model will work well in practice
   - High accuracy is real, not a bug

## 📝 Next Steps

1. Review all 8 visualizations to understand your data
2. Decide: Are these shortcuts acceptable for your use case?
3. If yes: Train with current data (should work well)
4. If no: Need to add more diverse training data to remove shortcuts

## 🔍 How to Use These Visualizations

Open the images in order and look for:
- ✅ Good: Balanced classes, diverse samples, realistic augmentation
- ⚠️ Warning: Obvious visual patterns, unrealistic augmentation, mislabeled data
- ❌ Bad: Severe class imbalance, broken augmentation, corrupted samples

The visualizations will help you understand why your model behaves the way it does!
