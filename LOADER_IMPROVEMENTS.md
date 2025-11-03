# Loader Improvements - QuickDraw Appendix Dataset

## Overview

The `appendix_loader.py` module has been significantly enhanced to support loading the **entire QuickDraw Appendix library** with multiple classes and variants.

## Key Improvements

### 1. **Multi-File Discovery**
- New function: `discover_appendix_files(appendix_dir)`
- Automatically scans directory for all NDJSON files
- Extracts category names and variants from filenames (e.g., `penis-raw.ndjson` → category: `penis`, variant: `raw`)
- Returns organized dictionary structure for easy iteration

### 2. **Multi-Category Loading**
- New function: `load_appendix_category(filepath, max_samples, prefer_variant)`
- Loads individual category with error handling
- Supports both raw and simplified stroke formats
- Returns both images and category name

### 3. **Full Library Dataset Preparation**
- New function: `prepare_multi_class_dataset(appendix_dir, ...)`
- Loads **all categories** from the appendix directory
- Automatically chooses best variant (prefers 'raw' over 'simplified')
- Generates corresponding negative samples (random noise)
- Saves complete dataset with category mapping

### 4. **Flexible CLI Interface**
The module now supports two distinct modes:

#### Mode A: Single File (Original)
```bash
python appendix_loader.py --input penis-raw.ndjson \
  --output-dir data/processed \
  --max-samples 2000
```

#### Mode B: Entire Library (New)
```bash
python appendix_loader.py --appendix-dir /path/to/quickdraw_appendix \
  --output-dir data/processed \
  --max-samples 3000
```

## Function Reference

### `discover_appendix_files(appendix_dir)` → dict
**Purpose:** Discover all NDJSON files in the appendix directory

**Returns:**
```python
{
    'category_name': [
        {'path': Path(...), 'variant': 'raw', 'filename': 'category-raw.ndjson'},
        {'path': Path(...), 'variant': 'simplified', 'filename': 'category-simplified.ndjson'},
        ...
    ],
    ...
}
```

**Example:**
```python
categories = discover_appendix_files('/path/to/quickdraw_appendix')
# Found 2 NDJSON files:
#   ✓ penis-raw.ndjson (penis - raw)
#   ✓ penis-simplified.ndjson (penis - simplified)
```

### `load_appendix_category(filepath, max_samples, prefer_variant)` → (images, category)
**Purpose:** Load a single category with error handling

**Parameters:**
- `filepath` (Path): NDJSON file path
- `max_samples` (int, optional): Limit number of samples
- `prefer_variant` (str): 'raw' (default) or 'simplified'

**Returns:**
- `images` (ndarray): 28×28 bitmaps as uint8 array
- `category` (str): Category name extracted from filename

### `prepare_multi_class_dataset(appendix_dir, output_dir, max_samples_per_class, test_split, negative_class_type)` → ((X_train, y_train), (X_test, y_test), class_mapping)
**Purpose:** Prepare complete dataset from entire library

**Parameters:**
- `appendix_dir` (str): Path to QuickDraw Appendix directory
- `output_dir` (str): Output directory (default: "data/processed")
- `max_samples_per_class` (int, optional): Max samples per category (None = all)
- `test_split` (float): Test set fraction (default: 0.2)
- `negative_class_type` (str): "noise" or "other" (default: "noise")

**Output Files:**
- `X_train.npy` - Training images (normalized to [0, 1])
- `y_train.npy` - Training labels (1 = positive, 0 = negative)
- `X_test.npy` - Test images
- `y_test.npy` - Test labels
- `class_mapping.pkl` - Class indices and metadata

**Returns:**
```python
(
    (X_train, y_train),  # Training set
    (X_test, y_test),    # Test set
    {
        'negative': 0,
        'positive': 1,
        'categories': {'penis': 0, ...},
        'description': '...'
    }
)
```

## Dataset Schema

### Multi-Class Dataset Structure
```
Binary Classification: Positive vs Negative

Positive Class (1):
  - penis: 2472 samples from penis-raw.ndjson
  - (scalable to more categories when added to appendix)

Negative Class (0):
  - Random noise: 2472 synthetic images
```

### Data Format
- **Images:** 28×28 pixels, grayscale, float32, normalized to [0, 1]
- **Labels:** Binary (0 or 1)
- **Shape:** (num_samples, 28, 28, 1)

## Usage Examples

### Load entire appendix library with default settings:
```python
from appendix_loader import prepare_multi_class_dataset

(X_train, y_train), (X_test, y_test), class_info = prepare_multi_class_dataset(
    '/home/mcvaj/ML/quickdraw_appendix'
)
print(f"Categories: {class_info['categories']}")
```

### Load with max samples per category:
```bash
python appendix_loader.py --appendix-dir ./quickdraw_appendix \
  --output-dir ./data/processed \
  --max-samples 5000 \
  --test-split 0.15
```

### Discover available files without loading:
```python
from appendix_loader import discover_appendix_files

categories = discover_appendix_files('./quickdraw_appendix')
for category, files in categories.items():
    print(f"\n{category}:")
    for f in files:
        print(f"  - {f['filename']} ({f['variant']})")
```

## Benefits

✅ **Scalability:** Easily add new categories - just place NDJSON files in the appendix directory
✅ **Flexibility:** Choose between single-file or multi-file loading
✅ **Robustness:** Error handling for malformed drawings with silent skip
✅ **Format Agnostic:** Automatically detects and prefers raw over simplified variants
✅ **Structured Output:** Comprehensive class mappings for reproducible training
✅ **CLI Ready:** Command-line interface for quick dataset preparation

## Current Status

**Tested with:**
- Penis category (1 class, 2 variants)
- 2472 valid drawings loaded
- 528 malformed entries skipped
- Binary classification dataset: 3956 train, 988 test

**Ready for:**
- Multiple categories (when appendix repo is updated)
- Large-scale training with aggregated categories
- Binary classification pipeline against random noise

