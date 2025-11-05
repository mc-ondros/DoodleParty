# DoodleHunter Project Structure

**Purpose:** Documentation of file organization and directory structure.

## Proposed Better Structure

The current structure mixes concerns and doesn't follow Python best practices. Here's the recommended structure using src-layout:

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
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── utils/             # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── web/               # Web interface
│       ├── __init__.py
│       ├── app.py         # Flask app
│       ├── templates/
│       └── static/
├── scripts/               # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── download_data.py
│   └── optimize_threshold.py
├── tests/                 # Test suite
│   ├── test_core/
│   ├── test_data/
│   └── test_web/
├── data/                  # Data storage (gitignored)
├── models/                # Saved models (gitignored)
├── docs/                  # Documentation (not hidden)
│   ├── api.md
│   ├── architecture.md
│   └── ...
├── .github/               # GitHub workflows
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Dependencies
└── README.md
```

## Current Structure Issues

1. **Flat `src/` directory** - All files mixed together, hard to navigate
2. **Hidden documentation** - `.documentation/` is non-standard
3. **No package structure** - Can't import as `from src.core import models`
4. **Scripts mixed with modules** - Training scripts mixed with library code
5. **No tests directory** - Tests scattered or missing
6. **Missing `__init__.py`** - Not a proper Python package
7. **No separation** - Executable scripts vs importable modules

## Recommended Structure Details

### Package Layout (`src/`)

**Core ML (`src/core/`):**
- `models.py` - CNN architectures, model factory functions
- `training.py` - Training loops, callbacks, checkpointing
- `inference.py` - Single/batch prediction, TTA

**Data Handling (`src/data/`):**
- `loaders.py` - QuickDraw dataset loading, class mapping
- `preprocessing.py` - Image normalization, resizing
- `augmentation.py` - Data augmentation pipeline

**Utilities (`src/utils/`):**
- `metrics.py` - Custom metrics, evaluation functions
- `visualization.py` - Plotting, confusion matrices
- `config.py` - Configuration management

**Web Interface (`src/web/`):**
- `app.py` - Flask application
- `routes.py` - API endpoints
- `templates/` - HTML templates
- `static/` - CSS, JS, images

### Scripts Directory

Executable scripts that use the package:
- `train.py` - CLI for training models
- `evaluate.py` - CLI for evaluation
- `download_data.py` - Data download utility
- `serve.py` - Start web server

### Tests Directory

Organized by module:
- `test_core/` - Test core ML functionality
- `test_data/` - Test data loading/preprocessing
- `test_web/` - Test Flask endpoints
- `conftest.py` - Pytest fixtures

### Documentation Directory

Visible `docs/` instead of hidden `.documentation/`:
- `api.md` - API reference
- `architecture.md` - System design
- `installation.md` - Setup guide
- `troubleshooting.md` - Common issues

## Benefits of Proposed Structure

1. **Proper Python Package** - Can install with `pip install -e .`
2. **Clear Imports** - `from src.core import models`
3. **Separation of Concerns** - Scripts vs library code
4. **Standard Src-Layout** - Follows Python packaging best practices
5. **Testable** - Dedicated tests directory with proper structure
6. **Discoverable Docs** - `docs/` instead of hidden `.documentation/`
7. **Professional** - Used by major Python projects (pytest, setuptools, etc.)

## Migration Path

To migrate from current to proposed structure:

1. Reorganize `src/` into subpackages (core, data, utils, web)
2. Add `__init__.py` files to all packages
3. Move executable scripts to `scripts/` directory
4. Move `app/` contents to `src/web/`
5. Rename `.documentation/` to `docs/`
6. Create `tests/` directory structure
7. Create `pyproject.toml` for modern packaging
8. Update all imports throughout codebase

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

## Current Structure (For Reference)

The existing structure has these directories:
- `src/` - Flat directory with all Python modules (needs reorganization)
- `app/` - Flask web interface (should move to `src/web/`)
- `data/` - Dataset storage (keep as-is)
- `models/` - Trained models (keep as-is)
- `.documentation/` - Hidden documentation (rename to `docs/`)
- `scripts/` - Shell scripts (keep, but move Python scripts here)

**Key Changes Needed:**
1. Reorganize `src/` into subpackages
2. Move `app/` → `src/web/`
3. Rename `.documentation/` → `docs/`
4. Separate executable scripts from library code
5. Add proper `__init__.py` files
6. Create `tests/` directory

## Related Documentation

- [Architecture](architecture.md) - System design details
- [Roadmap](roadmap.md) - Development timeline
- [API Documentation](api.md) - API reference
- [Style Guide](../STYLE_GUIDE.md) - Code conventions
- [README](../README.md) - Project overview

*Project structure documentation for DoodleHunter v1.0*
