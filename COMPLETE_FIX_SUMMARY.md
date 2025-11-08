# Complete Preprocessing Fix Summary

## Problem: 0% Confidence on All Drawings

The model was returning 0.0% confidence for all drawings due to **THREE critical preprocessing mismatches** between training and inference.

---

## Fix 1: Canvas Color Inversion ✅

**File:** `src/web/app.py` line 903

**Problem:** Canvas sends black strokes on white background, but QuickDraw model expects white strokes on black background (like the training dataset).

**Solution:**
```python
# BEFORE (incorrect):
# canvas = img_array  # No inversion

# AFTER (correct):
img_array = 255 - img_array  # Invert to match QuickDraw format
```

---

## Fix 2: Z-Score Normalization ✅

**File:** `src/core/shape_normalization.py` lines 144-155

**Problem:** Training data used z-score normalization, but inference only used simple `/255.0` scaling.

**Solution:**
```python
# Convert to [0, 1] range
img_array = image.astype(np.float32) / 255.0

# Apply z-score normalization (same as training pipeline)
img_flat = img_array.flatten()
if img_flat.std() > 0.01:
    # Standardize to zero mean, unit variance
    img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
    # Rescale from ~[-2, 2] to [0, 1] for model compatibility
    img_array = (img_array + 2) / 4
    img_array = np.clip(img_array, 0, 1)
```

---

## Fix 3: Padding Color ✅

**Files:** 
- `src/core/shape_types.py` line 26
- `src/web/app.py` line 929

**Problem:** Used gray padding (243) but model was trained on black background (0).

**Solution:**
```python
# BEFORE:
PADDING_COLOR_BG: int = 243  # Gray
padding_color=243

# AFTER:
PADDING_COLOR_BG: int = 0  # Black (matches QuickDraw)
padding_color=0
```

---

## Fix 4: Shape Margins & Scaling ✅

**File:** `src/core/shape_normalization.py` lines 51-54, 82-84

**Problem:** Too much margin (100%) and too small scaling (75%) made drawings tiny on canvas, creating mostly empty images.

**Solution:**
```python
# BEFORE:
margin = max(int(1.0 * max(w, h)), 32)  # 100% margin
inner_size = int(target_size * 0.75)    # Use only 75% of canvas

# AFTER:
margin = max(int(0.3 * max(w, h)), 16)  # 30% margin
inner_size = int(target_size * 0.9)     # Use 90% of canvas
```

This makes drawings fill more of the 128x128 canvas, matching training data better.

---

## Visual Comparison

### Before Fixes:
- Canvas: Black strokes on white → Model receives: Black on white (WRONG)
- Normalization: Simple /255.0 → Model trained on z-score (MISMATCH)
- Padding: Gray (243) → Model trained on black (MISMATCH)
- Drawing size: ~50% of canvas → Too much empty space

### After Fixes:
- Canvas: Black strokes on white → **Inverted** → Model receives: White on black ✅
- Normalization: Z-score applied → Matches training ✅
- Padding: Black (0) → Matches training ✅
- Drawing size: ~90% of canvas → Better matches training data ✅

---

## Expected Results After Fixes

| Drawing Type | Expected Confidence |
|--------------|-------------------|
| Clear penis shape | **50-95%** |
| Penis-like (ambiguous) | **30-50%** |
| Simple shapes (circle, line) | **5-20%** |
| Empty canvas | **<5%** |

---

## Files Modified

1. **`src/web/app.py`** - Added canvas inversion, fixed padding color
2. **`src/core/shape_normalization.py`** - Added z-score normalization, reduced margins
3. **`src/core/shape_types.py`** - Changed default padding color to black

---

## Testing

All 216 tests pass ✅

To restart the app with fixes:
```bash
cd /home/diatom/Documents/DoodleHunter
nix develop --command python src/web/app.py
```

Then visit http://localhost:5000 and draw something. You should now see **meaningful confidence scores**!

---

## Technical Notes

### Why Z-Score Normalization?

The training pipeline (`src/data/augmentation.py`) applies per-image z-score normalization to remove brightness bias:

1. **Without z-score:** Model could memorize "dense drawings = positive"
2. **With z-score:** Model learns actual shape patterns, not just density

### Why Black Padding?

The QuickDraw dataset and your training data (`scripts/data_processing/process_all_data_128x128.py` line 83) use:
- Black background (0)
- White strokes (255)

Using gray padding (243) confuses the model because it never saw that during training.

### Why Reduce Margins?

Training data (`process_all_data_128x128.py` line 76) uses:
- 20px padding on 256x256 canvas = ~8% margin
- Centered with minimal padding

The old code used 100% margin + 75% scaling, making drawings occupy only ~30% of the canvas. The new code (30% margin + 90% scaling) makes drawings occupy ~70% of the canvas, much closer to training data.

---

## Verification

Check debug images at `/tmp/doodlehunter_debug/` - they should now show:
- **Black background** (not gray)
- **Larger white drawings** (filling most of the 128x128 canvas)
- **Varied confidence scores** (not all 0.0)

If you still see issues, check that:
1. Model file is not corrupted
2. Flask app has been restarted (old code cached)
3. Browser cache is cleared
