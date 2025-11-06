# DoodleHunter Project Structure

**Purpose:** Documentation of file organization and directory structure.

## Current Structure

The project follows Python best practices using src-layout with proper package organization:

```
DoodleHunter/
├── src/                   # Source package (importable)
│   ├── __init__.py
│   ├── core/              # Core ML functionality
│   │   ├── __init__.py
│   │   ├── models.py      # Model architectures
│   │   ├── training.py    # Training logic
│   │   └── inference.py   # Prediction logic
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py     # Dataset loading
│   │   ├── appendix_loader.py  # Additional data loading
│   │   └── augmentation.py
│   ├── utils/             # Utilities
│   │   └── __init__.py
│   └── web/               # Web interface
│       ├── __init__.py
│       ├── app.py         # Flask app
│       ├── templates/
│       └── static/
├── scripts/               # Executable scripts
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation & testing scripts
│   ├── demos/             # Demo & utility scripts
│   ├── data_processing/   # Data download & processing
│   ├── visualization/     # Visualization scripts
│   └── convert/           # Model conversion scripts
├── tests/                 # Test suite
│   ├── test_core/
│   ├── test_data/
│   ├── test_web/
│   └── conftest.py
├── data/                  # Data storage
│   └── raw_ndjson/        # QuickDraw NDJSON files
├── models/                # Saved models
│   ├── *.keras            # TensorFlow/Keras models (float32)
│   ├── *.h5               # Legacy Keras format
│   ├── *.tflite           # TensorFlow Lite (float32)
│   ├── *_int8.tflite      # TensorFlow Lite (INT8 quantized)
│   ├── *.onnx             # ONNX format
│   └── benchmarks/        # Performance benchmark results
├── .documentation/        # Documentation
│   ├── api.md
│   ├── architecture.md
│   ├── installation.md
│   ├── nix-usage.md
│   ├── roadmap.md
│   ├── structure.md
│   ├── troubleshooting.md
│   ├── model_improvements.md
│   └── region_detection.md
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
4. ✅ **Tests directory** - Organized test suite with proper structure
5. ✅ **Proper `__init__.py`** - Valid Python package
6. ✅ **Clear separation** - Executable scripts vs importable modules
7. ✅ **Modern packaging** - Uses `pyproject.toml` for configuration

## Structure Details

### Package Layout (`src/`)

**Core ML (`src/core/`):**
- `models.py` - CNN architectures, model factory functions
- `training.py` - Training loops, callbacks, checkpointing
- `inference.py` - Single/batch prediction, TTA, region-based detection
- `optimization.py` - Model quantization, pruning, distillation
- `batch_inference.py` - Optimized batch processing for multiple patches

**Data Handling (`src/data/`):**
- `loaders.py` - QuickDraw dataset loading, class mapping
- `appendix_loader.py` - Additional data loading utilities
- `augmentation.py` - Data augmentation pipeline

**Utilities (`src/utils/`):**
- `profiling.py` - Performance profiling and benchmarking
- `caching.py` - Redis integration for result caching
- `metrics.py` - Performance metrics collection

**Web Interface (`src/web/`):**
- `app.py` - Flask application
- `routes.py` - API endpoints
- `templates/` - HTML templates
- `static/` - CSS, JS, images

### Scripts Directory

Executable scripts organized by purpose:

**Training Scripts (`scripts/training/`):**
- `train.py` - CLI for training models
- `train_ensemble.py` - Train ensemble models
- `ensemble_model.py` - Ensemble model creation
- `generate_hard_negatives.py` - Hard negative mining
- `generate_negatives.py` - Generate negative samples

**Evaluation Scripts (`scripts/evaluation/`):**
- `evaluate.py` - CLI for evaluation
- `optimize_threshold.py` - Threshold optimization
- `test_time_augmentation.py` - TTA implementation
- `test_model_improvements.py` - Test model improvements

**Demo Scripts (`scripts/demos/`):**
- `demo_region_detection.py` - Demo region-based detection
- `run_interface.sh` - Start web interface
- `test_model.sh` - Quick model testing
- `train_max_accuracy.sh` - Train for maximum accuracy

**Data Processing Scripts (`scripts/data_processing/`):**
- `download_quickdraw_ndjson.py` - Download QuickDraw NDJSON files
- `download_quickdraw.py` - Download QuickDraw data
- `download_quickdraw_gcs.py` - Download from Google Cloud Storage
- `download_quickdraw_npy.py` - Download NumPy format data
- `process_all_data_128x128.py` - Process data to 128x128 format
- `fix_data_quality.py` - Data quality improvements
- `regenerate_data_simple_norm.py` - Regenerate with simple normalization
- `regenerate_training_data.py` - Regenerate training data
- `reprocess_penis_ndjson.py` - Reprocess specific category

**Visualization Scripts (`scripts/visualization/`):**
- `visualize_training_batches.py` - Visualize training data
- `visualize_training_data.py` - Visualize training dataset
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

Organized by module:
- `test_core/` - Test core ML functionality
  - `test_batch_inference.py` - Batch processing tests
  - `test_optimization.py` - Model optimization tests
- `test_data/` - Test data loading/preprocessing
- `test_web/` - Test Flask endpoints
- `test_performance/` - Performance and load tests
  - `test_latency.py` - Latency benchmarks
  - `test_throughput.py` - Throughput tests
  - `test_memory.py` - Memory profiling
- `conftest.py` - Pytest fixtures

### Documentation Directory

Currently `.documentation/` (hidden directory):
- `api.md` - API reference
- `architecture.md` - System design
- `installation.md` - Setup guide
- `nix-usage.md` - Nix/NixOS guide
- `roadmap.md` - Development roadmap
- `structure.md` - This file
- `troubleshooting.md` - Common issues
- `model_improvements.md` - Model enhancement docs
- `region_detection.md` - Region-based detection guide

## Benefits of Current Structure

1. ✅ **Proper Python Package** - Can install with `pip install -e .`
2. ✅ **Clear Imports** - `from src.core import models`
3. ✅ **Separation of Concerns** - Scripts vs library code
4. ✅ **Standard Src-Layout** - Follows Python packaging best practices
5. ✅ **Testable** - Dedicated tests directory with proper structure
6. ✅ **Nix Integration** - Full Nix flake support for reproducible builds
7. ✅ **Professional** - Used by major Python projects (pytest, setuptools, etc.)

## Data Format

**QuickDraw Dataset:**
- Format: NDJSON (newline-delimited JSON)
- Location: `data/raw_ndjson/`
- Files: `{category}-raw.ndjson`
- Each line contains a drawing with strokes and metadata

**Processed Data:**
- Location: `data/processed/`
- Format: NumPy arrays and pickle files
- Generated during training

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
from src.core.models import create_cnn_model
from src.core.training import train_model
from src.core.inference import predict_single
```

**Data handling:**
```python
from src.data.loaders import load_data
from src.data.augmentation import create_augmentation_pipeline
```

**Web interface:**
```python
from src.web.app import app
```

## Performance Optimization Structure

### Model Optimization Pipeline

```
Original Model (*.keras)
    ↓
TFLite Conversion (convert_to_tflite.py)
    ↓
INT8 Quantization (quantize_int8.py)
    ↓
Benchmarking (benchmark_tflite.py)
    ↓
Production Model (*_int8.tflite)
```

### Inference Pipeline Architecture

**Single Image (Current):**
```
Canvas → Preprocess → Model → Result (78-85ms)
```

**Multi-Region (Planned):**
```
Canvas → Extract Patches (9-16) → Batch Preprocess
    ↓
Batch Inference (single forward pass)
    ↓
Aggregate Results → Final Decision (<200ms target)
```

### Key Performance Components

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

**`src/utils/profiling.py`** - Performance monitoring
- Latency tracking
- Memory profiling
- Throughput measurement
- Bottleneck identification

**`scripts/benchmark_performance.py`** - Comprehensive benchmarking
- Single vs batch inference comparison
- Memory usage analysis
- Latency distribution (p50, p95, p99)
- GPU utilization metrics

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

*Project structure documentation for DoodleHunter v1.0*
