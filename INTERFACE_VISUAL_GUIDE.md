# 🎨 DoodleHunter ML Testing Interface - Visual Guide

## Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    DoodleHunter                                  │
│            Test the ML Model with Your Drawings                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┬──────────────────────────────┐
│                                  │                              │
│  📐 DRAWING CANVAS               │  📊 RESULTS SECTION         │
│  ┌────────────────────────────┐  │  ┌──────────────────────────┐
│  │                            │  │  │                          │
│  │                            │  │  │  ✓ VALID QUICKDRAW      │
│  │  [Your Drawing Here]       │  │  │                          │
│  │                            │  │  │  Looks like a QuickDraw  │
│  │                            │  │  │  doodle! ✓               │
│  └────────────────────────────┘  │  │                          │
│                                  │  │  Confidence: 95.5%       │
│  [Clear] [Get Verdict]           │  │  ████████░ (bar)        │
│                                  │  │                          │
│  Brush Size: ●●●●●● 5px         │  │  Raw: 0.9550             │
│                                  │  │  Threshold: 0.5          │
│                                  │  └──────────────────────────┘
└──────────────────────────────────┴──────────────────────────────┘

Draw a doodle and click "Get Verdict" to test the ML model
Model: QuickDraw Binary Classifier | Input: 28x28 Grayscale
```

## Workflow

```
START
  ↓
[Draw on Canvas] ← User draws a doodle
  ↓
[Adjust Brush Size] ← Optional: change pen thickness
  ↓
[Click "Get Verdict"] ← Submit drawing for prediction
  ↓
[Loading...] ← Server processes image
  ↓
[Preprocess] ← Convert to 28×28 grayscale
  ↓
[ML Model] ← CNN predicts: 0.0 to 1.0 probability
  ↓
[Display Result] ← Show verdict + confidence
  ↓
[Clear or Draw Again] ← Start over or refine
  ↓
END
```

## Model Decision Logic

```
Canvas Drawing
      ↓
Resize to 28×28
      ↓
Convert to Grayscale
      ↓
Normalize (0-1 range)
      ↓
CNN Prediction: 0.0 ────────────── 1.0
                ↓                   ↓
            OUT-OF-DISTRIBUTION  IN-DISTRIBUTION
            (Not QuickDraw)     (Is QuickDraw)
            
            ← THRESHOLD (0.5) →
```

## Feature Descriptions

### 🎯 Drawing Canvas
- **Size**: 400×400 pixels (input resized to 28×28 for model)
- **Colors**: Black pen on white background
- **Touch Support**: Works with mouse and touchscreen
- **Quality**: Anti-aliased lines with smooth curves

### 🎚️ Brush Size Control
- **Range**: 2-20 pixels
- **Default**: 5 pixels
- **Real-time**: Size preview updates instantly

### 🔘 Clear Button
- Clears the entire canvas
- Hides previous results
- Resets for new drawing

### ✨ Get Verdict Button
- Sends current drawing to ML model
- Shows loading spinner while processing
- Displays results when complete
- Disabled during prediction

### 📊 Results Display

#### Three Possible States:

1. **IN-DISTRIBUTION** (Green)
   - Drawing matches QuickDraw style
   - Confidence: High probability (≥ 0.5)
   - Message: "Looks like a QuickDraw doodle! ✓"

2. **OUT-OF-DISTRIBUTION** (Red)
   - Drawing doesn't match QuickDraw style
   - Confidence: High reverse probability (1 - prob)
   - Message: "Doesn't match QuickDraw style. ✗"

3. **ERROR**
   - Canvas empty or network issue
   - Shows error message
   - Can try again

### 📈 Confidence Display
- **Percentage**: 0-100% confidence
- **Visual Bar**: Animated progress bar
- **Raw Value**: Exact probability from model (0.0000-1.0000)
- **Threshold**: Current decision boundary

## Color Scheme

| Element | Color | Use |
|---------|-------|-----|
| Header Background | Purple Gradient | Branding |
| Buttons (Primary) | Purple Gradient | Main action |
| Buttons (Secondary) | Light Gray | Secondary action |
| Canvas | White | Drawing area |
| Pen | Black | Drawing stroke |
| Positive Result | Green | Valid doodle |
| Negative Result | Red | Invalid doodle |
| Confidence Bar | Purple Gradient | Visual feedback |

## Performance Characteristics

- **Processing Time**: ~50-200ms (varies by system)
- **Network Latency**: Depends on Flask server location
- **Canvas Rendering**: 60 FPS smooth drawing
- **File Size**: Base64 image ~5KB

## Browser Compatibility

✓ Chrome/Chromium 90+
✓ Firefox 88+
✓ Safari 14+
✓ Edge 90+
✓ Mobile browsers (iOS Safari, Chrome Mobile)

## Responsive Design Breakpoints

- **Desktop** (>768px): Side-by-side layout
- **Tablet** (481-768px): Stacked layout, adjusted sizes
- **Mobile** (<480px): Full-width, optimized touch

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + Enter | Get Verdict |
| Click Canvas | Start drawing |
| Press Clear Button | Reset canvas |

## File Sizes

- HTML: ~3 KB
- CSS: ~8 KB
- JavaScript: ~5 KB
- Model: 5 MB (loaded once on startup)

## API Response Example

```json
{
  "success": true,
  "verdict": "IN-DISTRIBUTION",
  "verdict_text": "Looks like a QuickDraw doodle! ✓",
  "confidence": 0.9550,
  "raw_probability": 0.9550,
  "threshold": 0.5
}
```

---

**Created**: November 3, 2025
**For**: DoodleHunter ML Project
**Status**: ✓ Production Ready
