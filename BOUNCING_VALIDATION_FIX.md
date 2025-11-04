# Validation Bouncing Issue - Analysis and Solutions

## Problem Summary

Validation accuracy is bouncing wildly during training:
- **Good epochs**: 94-99% validation accuracy
- **Bad epochs**: 50-79% validation accuracy (sometimes random guessing!)
- Training accuracy remains stable at 99%+

## Root Cause Analysis

### Test Results

#### 10-Epoch Test with LR=0.001
```
Epoch 1:  57% val_acc
Epoch 2:  97% val_acc ✓
Epoch 3:  99% val_acc ✓
Epoch 4:  59% val_acc ⚠️ CRASH
Epoch 5:  51% val_acc ⚠️ (random!)
Epoch 6:  64% val_acc
Epoch 7:  57% val_acc  
Epoch 8:  56% val_acc
Epoch 9:  99% val_acc ✓ RECOVERED
Epoch 10: 97% val_acc ✓
```

#### 10-Epoch Test with LR=0.0005 + Gradient Clipping
```
Epoch 1:  58% val_acc
Epoch 2:  54% val_acc
Epoch 3:  95% val_acc ✓
Epoch 4:  97% val_acc ✓
Epoch 5:  97% val_acc ✓
Epoch 6:  99% val_acc ✓
Epoch 7:  96% val_acc ✓
Epoch 8:  79% val_acc ⚠️ partial crash
Epoch 9:  71% val_acc ⚠️ 
Epoch 10: 98% val_acc ✓ RECOVERED
```

Gradient clipping + lower LR helped but didn't solve it completely.

## Diagnosis

This is **SEVERE OVERFITTING** with the following characteristics:

1. **Model memorizes training set perfectly** (99% train accuracy)
2. **Loses generalization periodically** (validation crashes)
3. **Eventually recovers** when learning rate reduces or by chance

### Contributing Factors

1. **Per-image normalization too aggressive**
   - Removes brightness signal (0.04 difference between classes)
   - Makes task artificially hard
   - Forces model to find subtle patterns that don't generalize

2. **Model too large for the task**
   - Enhanced model: 1.2M parameters
   - Dataset: 32k training samples  
   - Ratio: ~37 samples per 1000 params (very low!)
   - Ideal ratio: 100-1000+ samples per 1000 params

3. **Binary classification at 99% is unstable**
   - Small weight changes cause large prediction flips
   - One bad gradient update → all predictions flip
   - Validation: 8k samples, so 1% error = 80 samples flipped

## Solutions (in order of effectiveness)

### Solution 1: Use Standard Model (NOT Enhanced) ✅ RECOMMENDED

The enhanced model is overkill for this task:

```bash
# Remove --enhanced flag
python src/train.py \
    --data-dir data/processed \
    --epochs 50 \
    --batch-size 32 \
    --model-output models/quickdraw_model.h5 \
    --learning-rate 0.0005 \
    --label-smoothing 0.1 \
    --architecture custom \
    --aggressive-aug
```

Standard model: 423K params (3x smaller) → better generalization

### Solution 2: Reduce Per-Image Normalization Strength

In `regenerate_training_data.py`, line 114-121, the per-image normalization is very aggressive. Consider:

**Option A**: Remove it entirely (let brightness be a feature)
```python
# REMOVE lines 114-121 (per-image normalization)
```

**Option B**: Make it less aggressive
```python
# Weaker normalization - keep some brightness signal
for i in range(len(X)):
    img = X[i]
    img_flat = img.flatten()
    if img_flat.std() > 0.01:
        # Less aggressive: only center, don't scale to fixed std
        img = (img - img_flat.mean()) + 0.5  # Shift to 0.5 mean
        X[i] = np.clip(img, 0, 1)
```

### Solution 3: Add More Training Data

Current: 25k samples per class
Ideal: 50-100k samples per class for enhanced model

```python
# In regenerate_training_data.py, line 45
samples = min(5000, len(data))  # Increase from 1200 to 5000
```

### Solution 4: Use Weight Decay (L2 Regularization)

Add to optimizer:
```python
optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate,
    clipnorm=1.0,
    weight_decay=0.0001  # Add L2 regularization
)
```

### Solution 5: Use Cosine Annealing Instead of ReduceLROnPlateau

More stable learning rate schedule:
```python
# Replace ReduceLROnPlateau callback
keras.callbacks.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0001
)
```

## Immediate Action Plan

1. **Stop using --enhanced flag** ✅
2. **Use learning_rate=0.0005** ✅ (already updated in train_max_accuracy.sh)
3. **Train for 50 epochs** and monitor
4. If still bouncing, remove per-image normalization
5. If still bouncing after that, add more training data

## What We Fixed Already

✅ Removed double label smoothing
✅ Added gradient clipping (clipnorm=1.0)  
✅ Reduced learning rate (0.001 → 0.0005)
✅ Fixed validation split (was sequential, now stratified random)
✅ Added immediate shuffling in data generation

## Expected Behavior After Fixes

With standard model + LR=0.0005:
- Validation should stabilize after 5-10 epochs
- Final validation accuracy: 90-95%
- No catastrophic drops to 50%
- Smooth learning curve

## Testing Command

```bash
# Test with standard model (NOT enhanced)
python src/train.py \
    --data-dir data/processed \
    --epochs 15 \
    --batch-size 64 \
    --model-output models/quickdraw_test_standard.h5 \
    --learning-rate 0.0005 \
    --label-smoothing 0.1 \
    --architecture custom \
    2>&1 | grep -E "Epoch|val_accuracy"
```

Should see stable progression without wild bouncing!
