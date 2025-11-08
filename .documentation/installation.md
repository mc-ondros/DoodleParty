# DoodleHunter Installation Guide

**Purpose:** Complete installation and setup instructions.

**Status: Updated to match actual implementation** - Nov 2024

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for training)
- 2GB+ disk space for dataset and models
- Internet connection for downloading QuickDraw data

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/doodlehunter.git
cd doodlehunter
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- tensorflow>=2.13.0
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- scikit-learn>=1.3.0
- pillow>=10.0.0
- flask>=2.0.0
- flask-cors

### 4. Download QuickDraw Dataset

```bash
python scripts/data_processing/download_quickdraw_npy.py
```

This downloads NumPy bitmap files (28x28 pre-processed images) for selected categories to `data/raw/`.

### 5. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python -c "from src.data.loaders import load_data; print('Dataset module OK')"
```

## Configuration

### Environment Variables (Optional)

Create `.env` file in project root:

```bash
MODEL_PATH=models/quickdraw_model.h5
IMAGE_SIZE=28
THRESHOLD=0.5
FLASK_PORT=5000
FLASK_DEBUG=False
```

### Data Directory Structure

After installation, your data directory should look like:

```
data/
├── raw/
│   ├── penis.npy
│   ├── circle.npy
│   ├── square.npy
│   └── ...
└── processed/
    └── class_mapping.pkl
```

**Note:** The project uses pre-processed NumPy bitmap format (28x28 grayscale images) from Google's QuickDraw dataset, not raw NDJSON stroke data.

## Troubleshooting

### TensorFlow Installation Issues

**GPU Support:**
```bash
pip install tensorflow[and-cuda]  # For CUDA support
```

**CPU Only:**
```bash
pip install tensorflow-cpu  # Smaller package
```

### Memory Issues

If you encounter memory errors during data loading:

```bash
# Use --max-samples flag when training
python scripts/train.py --max-samples 5000
```

### Permission Errors

```bash
pip install --user -r requirements.txt
```

## Next Steps

1. Train a model: `python scripts/train.py --epochs 50 --batch-size 32`
2. Start web interface: `python src/web/app.py`
3. See [API Reference](api.md) for usage details

*Installation guide for DoodleHunter v1.0 - Updated Nov 2024*
