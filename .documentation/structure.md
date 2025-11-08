# DoodleHunter Project Structure

**Purpose:** Documentation of file organization and directory structure.

**Status: Updated to match actual code** - Nov 2024

## Current Structure

The project follows Python best practices using src-layout with proper package organization:

```
DoodleHunter/
├── src/                   # Source package (importable)
│   ├── __init__.py
│   ├── core/              # Core ML functionality
│   │   ├── __init__.py
│   │   ├── models.py      # Multiple CNN architectures (4 different models)
│   │   ├── training.py    # Training logic
│   │   ├── inference.py   # Prediction logic
│   │   ├── patch_extraction.py  # Region-based detection (basic)
│   │   ├── optimization.py      # Model optimization
│   │   ├── batch_inference.py   # Batch processing
│   │   └── tile_detection.py    # Tile-based detection (experimental)
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py     # Dataset loading (NumPy bitmap format)
│   │   ├── appendix_loader.py  # Additional data loading
│   │   └── augmentation.py
│   └── web/               # Web interface
│       ├── __init__.py
│       ├── app.py         # Flask app with multiple routes
│       └── routes.py      # API route definitions
├── scripts/               # Executable scripts
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation & testing scripts
│   ├── demos/             # Demo & utility scripts
│   ├── data_processing/   # Data download & processing
│   ├── visualization/     # Visualization scripts
│   └── convert/           # Model conversion scripts
├── tests/                 # Test suite (minimal/incomplete)
│   └── (test files may be minimal)
├── data/                  # Data storage
│   └── raw/               # QuickDraw NumPy bitmap files (28x28)
├── models/                # Saved models
│   ├── quickdraw_model.h5              # Main model file
│   ├── quickdraw_model.tflite          # TensorFlow Lite (float32)
│   ├── quickdraw_model_int8.tflite     # TensorFlow Lite (INT8 quantized)
│   ├── quickdraw_model.onnx            # ONNX format
│   └── benchmarks/        # Performance benchmark results
├── logs/                  # Application logs
├── .documentation/        # Documentation
│   ├── api.md
│   ├── architecture.md
│   ├── installation.md
│   ├── nix-usage.md
│   ├── roadmap.md
│   ├── structure.md
│   ├── troubleshooting.md
│   └── model_improvements.md
├── .github/               # GitHub workflows
│   └── workflows/
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Dependencies
├── flake.nix              # Nix flake configuration
├── module.nix             # NixOS module
├── STYLE_GUIDE.md         # Code style guide
└── README.md
```

## Structure Benefits

1. ✅ **Organized `src/` directory** - Clear separation by functionality
2. ✅ **Proper package structure** - Can import as `from src.core import models`
3. ✅ **Separated scripts** - Training scripts in `scripts/`, library code in `src/`
4. ✅ **Tests directory** - Test suite structure (though may be incomplete)
5. ✅ **Proper `__init__.py`** - Valid Python package
6. ✅ **Clear separation** - Executable scripts vs importable modules
7. ✅ **Modern packaging** - Uses `pyproject.toml` for configuration

## Structure Details

### Package Layout (`src/`)

**Core ML (`src/core/`):**
- `models.py` - Multiple CNN architectures (Custom, ResNet50, MobileNetV3, EfficientNet)
- `training.py` - Training loops, callbacks, checkpointing
- `inference.py` - Single/batch prediction, region-based detection
- `patch_extraction.py` - Sliding window detection (basic implementation)
- `batch_inference.py` - Optimized batch processing for multiple patches
- `optimization.py` - Model quantization, pruning, distillation
- `tile_detection.py` - Tile-based detection (experimental/incomplete)

**Data Handling (`src/data/`):**
- `loaders.py` - QuickDraw dataset loading (NumPy bitmap format)
- `appendix_loader.py` - Additional data loading utilities
- `augmentation.py` - Data augmentation pipeline

**Web Interface (`src/web/`):**
- `app.py` - Flask application with multiple prediction modes
- `routes.py` - API route definitions
- Note: templates/ and static/ directories are missing (should be checked/created)

### Scripts Directory

Executable scripts organized by purpose:

**Training Scripts (`scripts/training/`):**
- `train.py` - CLI for training models

**Evaluation Scripts (`scripts/evaluation/`):**
- `evaluate.py` - CLI for evaluation
- `optimize_threshold.py` - Threshold optimization
- `test_time_augmentation.py` - TTA implementation

**Data Processing Scripts (`scripts/data_processing/`):**
- `download_quickdraw_npy.py` - **MAIN:** Download NumPy bitmap format (28x28)
- `download_quickdraw_gcs.py` - Download from Google Cloud Storage
- `download_quickdraw_npy.py` - Download NumPy format data
- `process_all_data_128x128.py` - Historical, no longer needed (data is already 28x28)
- `fix_data_quality.py` - Data quality improvements
- `regenerate_data_simple_norm.py` - Regenerate with simple normalization
- `regenerate_training_data.py` - Regenerate training data
- `reprocess_penis_ndjson.py` - Reprocess specific category (legacy NDJSON)

**Note:** NDJSON processing scripts are legacy and not used in current implementation.

**Visualization Scripts (`scripts/visualization/`):**
- `visualize_training_data.py` - Visualize training dataset
- `visualize_training_batches.py` - Visualize training batches
- `visualize_fixed_data.py` - Visualize fixed data
- `diagnose_data_issues.py` - Data quality diagnostics

**Conversion Scripts (`scripts/convert/`):**
- `convert_to_onnx.py` - Convert models to ONNX format
- `convert_to_tflite.py` - Convert Keras models to TensorFlow Lite
- `quantize_int8.py` - Apply INT8 quantization to TFLite models
- `benchmark_tflite.py` - Benchmark TFLite model performance
- `prune_model.py` - Apply weight pruning to reduce model size
- `distill_model.py` - Knowledge distillation for smaller models
- `optimize_graph.py` - TensorFlow graph optimization

### Tests Directory

Organized by module (NOTE: Test coverage is minimal):
- Test files may be incomplete or missing
- See `test_region_detection.py` in root for existing test example

### Documentation Directory

Currently `.documentation/` (hidden directory):
- `api.md` - API reference
- `architecture.md` - System design
- `installation.md` - Setup guide
- `nix-usage.md` - Nix/NixOS guide
- `roadmap.md` - Development roadmap
- `structure.md` - This file
- `troubleshooting.md` - Common issues
- `model_improvements.md` - Model enhancement docs (may be outdated)

**Removed:** `region_detection.md` - Marked as outdated and removed.

## Benefits of Current Structure

1. ✅ **Proper Python Package** - Can install with `pip install -e .`
2. ✅ **Clear Imports** - `from src.core import models`
3. ✅ **Separation of Concerns** - Scripts vs library code
4. ✅ **Standard Src-Layout** - Follows Python packaging best practices
5. ✅ **Testable** - Dedicated tests directory (though may be incomplete)
6. ✅ **Nix Integration** - Full Nix flake support for reproducible builds
7. ✅ **Professional** - Used by major Python projects (pytest, setuptools, etc.)

## Data Format

**QuickDraw Dataset:**
- Format: NumPy bitmap (pre-processed 28x28 grayscale images)
- Location: `data/raw/`
- Files: `{category}.npy` (e.g., `penis.npy`, `circle.npy`)
- Source: Google's QuickDraw dataset (numpy_bitmap format)
- Each file contains NumPy array of 28x28 grayscale bitmaps

**Processed Data:**
- Location: `data/processed/`
- Format: NumPy arrays and pickle files (class_mapping.pkl)
- Generated during training

**Note:** This is a change from earlier versions that used NDJSON format.
Current implementation uses pre-processed NumPy bitmaps (28x28) for simplicity.

## File Naming Conventions

**Python Modules:**
- snake_case: `data_loaders.py`, `model_training.py`
- Private modules: `_internal.py`

**Python Packages:**
- snake_case: `src/`, `core/`, `data/`

**Scripts:**
- snake_case: `train.py`, `download_data.py`
- Executable with shebang: `#!/usr/bin/env python`

**Documentation:**
- kebab-case: `api.md`, `architecture.md`

**Tests:**
- Prefix with `test_`: `test_models.py`, `test_loaders.py`

## Import Examples

**Core ML functionality:**
```python
from src.core.models import build_custom_cnn
from src.core.training import train_model
from src.core.inference import predict_image
```

**Data handling:**
```python
from src.data.loaders import QuickDrawDataset
from src.data.augmentation import create_augmentation_pipeline
```

**Web interface:**
```python
from src.web.app import app
```

## Model Optimization Structure

### Model Optimization Pipeline

```
Original Model (quickdraw_model.h5)
    ↓
TFLite Conversion (convert_to_tflite.py)
    ↓
INT8 Quantization (quantize_int8.py)
    ↓
Benchmarking (benchmark_tflite.py)
    ↓
Production Model (quickdraw_model_int8.tflite)
```

### Inference Pipeline Architecture

**Single Image (Current):**
```
Canvas → Preprocess → Model → Result
```

**Region-Based (Basic Implementation):**
```
Canvas → Extract Contours → Individual Contour Classification
    ↓
Aggregate Results → Final Decision
```

**Tile-Based (Experimental):**
```
Canvas → Tile Grid (8x8) → Batch Inference
    ↓
Aggregate Results → Final Decision
```

### Key Components

**`src/core/batch_inference.py`** - Core batch processing
- Batch patch extraction
- Parallel preprocessing
- Single forward pass for all patches
- Result aggregation strategies

**`src/core/optimization.py`** - Model optimization
- INT8 quantization
- Weight pruning
- Knowledge distillation
- Graph optimization

**`src/core/patch_extraction.py`** - Region-based detection
- Sliding window detection
- Contour-based shape extraction
- Aggregation strategies (MAX, MEAN, etc.)

## Missing Components

**NOTE:** The following directories/files are referenced in documentation but may be missing:
- `src/web/templates/` - HTML templates for web interface
- `src/web/static/` - CSS, JS, images for web interface
- Comprehensive test suite in `tests/`
- Some scripts in `scripts/training/` and `scripts/demos/`

These may need to be created or the documentation needs updating to reflect actual structure.

## Nix Integration

The project includes comprehensive Nix support:
- `flake.nix` - Nix flake with development shell and apps
- `module.nix` - NixOS service module
- `.envrc` - direnv integration

See [Nix Usage Guide](nix-usage.md) for details.

## Related Documentation

- [Architecture](architecture.md) - System design details
- [Roadmap](roadmap.md) - Development timeline
- [API Documentation](api.md) - API reference
- [Style Guide](../STYLE_GUIDE.md) - Code conventions
- [README](../README.md) - Project overview

*Project structure documentation for DoodleHunter v1.0 - Updated Nov 2024*
