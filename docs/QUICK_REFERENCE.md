# QuickDraw Appendix ML - Quick Reference Guide

## Project Structure

```
ML/
├── src/
│   ├── appendix_loader.py      # [ENHANCED] Multi-file dataset loader
│   ├── train.py                # Model training pipeline
│   ├── predict.py              # Inference and evaluation
│   └── dataset.py              # Utility functions
│
├── data/
│   └── processed/
│       ├── X_train.npy         # Training images
│       ├── y_train.npy         # Training labels
│       ├── X_test.npy          # Test images
│       ├── y_test.npy          # Test labels
│       └── class_mapping.pkl   # Category metadata
│
├── models/
│   ├── quickdraw_appendix_detector.h5      # Original model
│   ├── quickdraw_appendix_detector_v2.h5   # Model from new loader
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── quickdraw_appendix/         # Cloned repo
│   ├── penis-raw.ndjson        # 87 MB
│   └── penis-simplified.ndjson # 9.4 MB
│
├── README.md                   # Project overview
├── TRAINING_REPORT.md          # Detailed training results
├── LOADER_IMPROVEMENTS.md      # Technical details
└── LOADER_ENHANCEMENT_SUMMARY.md # Feature summary
```

## Task: Load and Train

### Step 1: Load Data from Entire Appendix Library
```bash
cd /home/mcvaj/ML

# Load all categories from the appendix directory
./.venv/bin/python src/appendix_loader.py \
  --appendix-dir ./quickdraw_appendix \
  --output-dir ./data/processed \
  --max-samples 3000
```

**Output:**
- `data/processed/X_train.npy` - Training images
- `data/processed/y_train.npy` - Training labels
- `data/processed/X_test.npy` - Test images
- `data/processed/y_test.npy` - Test labels
- `data/processed/class_mapping.pkl` - Category mapping

### Step 2: Train Model
```bash
./.venv/bin/python src/train.py \
  --data-dir ./data/processed \
  --epochs 20 \
  --batch-size 32 \
  --model-output ./models/quickdraw_detector.h5
```

**Results:**
- Model trained to ~100% accuracy
- Saved to `models/quickdraw_detector.h5`
- Training history plot saved

### Step 3: Evaluate Model
```bash
./.venv/bin/python src/predict.py \
  --model ./models/quickdraw_detector.h5 \
  --data-dir ./data/processed \
  evaluate
```

**Metrics:**
- Accuracy: ~100%
- AUC: ~1.0
- Confusion matrix visualization

## Loader Modes

### Mode 1: Single File (Original)
```bash
./.venv/bin/python src/appendix_loader.py \
  --input ./quickdraw_appendix/penis-raw.ndjson \
  --output-dir ./data/processed \
  --max-samples 2000
```

### Mode 2: Entire Library (New) ⭐
```bash
./.venv/bin/python src/appendix_loader.py \
  --appendix-dir ./quickdraw_appendix \
  --output-dir ./data/processed \
  --max-samples 5000
```

## Dataset Information

### Current Status
- **Categories:** 1 (penis)
- **Positive samples:** 2,472 real drawings
- **Negative samples:** 2,472 synthetic (random noise)
- **Training set:** 3,956 samples (80%)
- **Test set:** 988 samples (20%)
- **Image size:** 28×28 grayscale
- **Format:** Normalized float32 [0, 1]

### Task
- **Type:** Binary classification
- **Positive class:** In-distribution (appendix drawings)
- **Negative class:** Out-of-distribution (random noise)

## Model Architecture

**Type:** CNN (Convolutional Neural Network)
- 3 Convolutional blocks (32→64→128 filters)
- 2 Dense layers (256, 128 units)
- Batch normalization on all layers
- Dropout for regularization
- **Output:** Sigmoid (binary probability)
- **Parameters:** 423,297 (1.61 MB)

## Performance Metrics

**Best Results (from v2.h5):**
```
Train Accuracy:     100% (100.0)
Test Accuracy:      100% (1.0000)
Test AUC:           100% (1.0000)
Test Loss:          0.0000

Precision (positive): 100%
Recall (positive):    100%
F1-Score (positive):  100%

Precision (negative): 100%
Recall (negative):    100%
F1-Score (negative):  100%
```

## Common Commands

| Task | Command |
|------|---------|
| Load appendix data | `python src/appendix_loader.py --appendix-dir ./quickdraw_appendix` |
| Train model | `python src/train.py --data-dir ./data/processed` |
| Evaluate | `python src/predict.py --model ./models/quickdraw_detector.h5 --data-dir ./data/processed evaluate` |
| Predict image | `python src/predict.py --model ./models/quickdraw_detector.h5 --image-path ./sample.png predict` |
| Show help | `python src/appendix_loader.py --help` |

## Python API Usage

### Load Data Programmatically
```python
from appendix_loader import prepare_multi_class_dataset

(X_train, y_train), (X_test, y_test), class_info = prepare_multi_class_dataset(
    './quickdraw_appendix',
    output_dir='./data/processed',
    max_samples_per_class=3000,
    test_split=0.2
)

print(f"Classes: {class_info['categories']}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
```

### Discover Available Files
```python
from appendix_loader import discover_appendix_files

categories = discover_appendix_files('./quickdraw_appendix')
for cat, files in categories.items():
    print(f"{cat}:")
    for f in files:
        print(f"  - {f['filename']} ({f['variant']})")
```

## File Locations

```
Cloned Repo:      /home/mcvaj/ML/quickdraw_appendix/
Processed Data:   /home/mcvaj/ML/data/processed/
Models:           /home/mcvaj/ML/models/
Source Code:      /home/mcvaj/ML/src/
```

## Environment

- **Python:** 3.12
- **TensorFlow:** 2.20.0
- **Framework:** Keras
- **Numpy:** 2.3.4
- **Pandas:** 2.3.3

## Key Improvements (Recent)

✅ **Multi-file discovery** - Automatically finds all NDJSON files
✅ **Variant selection** - Prefers raw over simplified formats
✅ **Batch loading** - Loads all categories in one call
✅ **Error resilience** - Silently skips malformed entries
✅ **Scalability** - Ready for more categories automatically
✅ **Metadata tracking** - Category mappings for reproducibility

## Documentation Files

- `README.md` - Project overview
- `TRAINING_REPORT.md` - Detailed training results and analysis
- `LOADER_IMPROVEMENTS.md` - Technical documentation
- `LOADER_ENHANCEMENT_SUMMARY.md` - Feature summary
- This file - Quick reference

## Next Steps

1. ✅ Data loader supports entire appendix library
2. ✅ Model achieves perfect accuracy on current data
3. 🔄 Monitor for new categories added to appendix repo
4. ⏳ Consider training on specific categories separately
5. ⏳ Explore semi-supervised learning approaches

---

**Last Updated:** November 3, 2025
**Loader Version:** 2.0 (Multi-file support)
**Model Version:** v2 (Verified with new loader)

