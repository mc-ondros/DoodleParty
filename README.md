# DoodleHunter

**Binary classification for drawing content moderation**

ML system for classifying hand-drawn sketches using TensorFlow and QuickDraw dataset.

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
python scripts/download_quickdraw.py
python scripts/train.py
python src/web/app.py
```

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

- Binary classification of hand-drawn sketches
- QuickDraw dataset integration
- CNN model with TensorFlow/Keras
- Flask web interface
- Threshold optimization and test-time augmentation

## Requirements

- Python 3.9+
- 4GB+ RAM (8GB for training)
- 2GB+ disk space
- See `requirements.txt` for dependencies

## Installation

```bash
git clone https://github.com/yourusername/doodlehunter.git
cd doodlehunter
pip install -r requirements.txt
python src/download_quickdraw.py
```

See [Installation Guide](.documentation/installation.md) for detailed setup.

## Usage

**Train Model:**
```bash
./train_max_accuracy.sh
# or: python src/train.py --epochs 50 --batch-size 32
```

**Run Web Interface:**
```bash
cd app && python app.py
```

**Evaluate Model:**
```bash
python src/evaluate.py --model models/quickdraw_classifier.h5
```

See [API Reference](.documentation/api.md) for detailed usage.

## Contributing

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code standards and [Development Roadmap](.documentation/roadmap.md) for planned features.

## License

MIT License - see [LICENSE](LICENSE) for details.
