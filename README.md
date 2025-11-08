# DoodleHunter

**Binary classification for drawing content moderation**

ML system for classifying hand-drawn sketches using TensorFlow and QuickDraw dataset.
Uses 28x28 pixel images from Google's QuickDraw dataset (NumPy bitmap format).

**Status: Experimental** - This project is under active development. Documentation may not always match code.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start

**With Nix:**
```bash
nix develop
nix run .#web
```

**Without Nix:**
```bash
pip install -r requirements.txt
python scripts/data_processing/download_quickdraw_npy.py
python scripts/train.py
python src/web/app.py
```

**Note:** This project uses 28x28 pixel images from the QuickDraw dataset (NumPy bitmap format), not NDJSON. The dataset is pre-processed and doesn't require resizing.

Access at `http://localhost:5000`

## Documentation

- [Installation & Setup](.documentation/installation.md)
- [Nix Usage Guide](.documentation/nix-usage.md)
- [Architecture](.documentation/architecture.md)
- [API Reference](.documentation/api.md)
- [Development Roadmap](.documentation/roadmap.md)
- [Project Structure](.documentation/structure.md)
- [Code Style Guide](STYLE_GUIDE.md)

## Features

- Binary classification of hand-drawn sketches (28x28 pixel images)
- QuickDraw dataset integration (NumPy bitmap format)
- Multiple CNN architectures (Custom, ResNet50, MobileNetV3, EfficientNet)
- Flask web interface with three detection modes:
  - Standard single-image classification (<30ms)
  - Contour-based detection for shape isolation (~125ms)
  - Tile-based detection for content dilution prevention (<200ms, experimental)
- TensorFlow Lite INT8 optimization for Raspberry Pi 4 deployment
- OpenCV-based contour detection with hierarchical support (planned)

## Requirements

- Python 3.9+
- 4GB+ RAM (8GB for training)
- 2GB+ disk space
- See `requirements.txt` and `pyproject.toml` for dependencies

## Installation

```bash
git clone https://github.com/yourusername/doodlehunter.git
cd doodlehunter
pip install -r requirements.txt
python scripts/data_processing/download_quickdraw_ndjson.py
```

See [Installation Guide](.documentation/installation.md) for detailed setup.

## Usage

**Train Model:**
```bash
python scripts/train.py --epochs 50 --batch-size 32
```

**Run Web Interface:**
```bash
python src/web/app.py
```

**Evaluate Model:**
```bash
python scripts/evaluate.py --model models/quickdraw_model.h5
```

**Download Data:**
```bash
python scripts/data_processing/download_quickdraw_npy.py
```

See [API Reference](.documentation/api.md) for detailed usage.

## Contributing

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code standards and [Development Roadmap](.documentation/roadmap.md) for planned features.

## License

MIT License - see [LICENSE](LICENSE) for details.

