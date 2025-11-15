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

### Lightweight QuickDraw Socket Canvas

For deployments where drawing events stream over `socket.io`, visit `http://localhost:5000/quickdraw-socket`. The viewer expects QuickDraw stroke vectors (`[[x...], [y...], [t...]]`) and listens on the `quickdraw.stroke`, `quickdraw.batch`, and `quickdraw.drawing` events. Use the `socketUrl` query parameter (for example `?socketUrl=http://your-host:3000`) if the socket server is hosted separately.

### Express-hosted Canvas Demo

If you prefer a vanilla Node + Express stack, run `npm install` (once) and then either `npm run serve:canvas` or `./start_express_canvas.sh` (make sure the script is executable). Both commands start a lightweight server on http://localhost:3000 that serves the same `quickdraw-canvas` page, serves the static assets from `src/web/static`, and emits sample QuickDraw strokes via `socket.io`. The canvas will automatically connect to the local socket server, but you can override it via `http://localhost:3000/?socketUrl=http://your-supplier:3000`.

To expose the canvas to other devices on your LAN, bind the server to `0.0.0.0` and set the port if needed (for example on a Raspberry Pi):

```bash
HOST=0.0.0.0 PORT=3000 npm run serve:canvas
```

Then open `http://<raspberry-pi-ip>:3000` from any machine on the same network. Make sure the port is allowed through any local firewalls.

### Drawing Sender Test Page

Use `http://<pi-ip>:3000/quickdraw-sender` to draw a sketch in the browser and emit the QuickDraw vectors (`[x[], y[], t[]]`) via `quickdraw.stroke`, `quickdraw.batch`, or `quickdraw.drawing`. This Express server relays those events to all connected clients (including the canvas) so the receiver renders them in real-time, and `quickdraw.clear` resets the canvas.

By default the canvas starts empty. Set `DEMO_MODE=1` before starting (e.g., `DEMO_MODE=1 ./start_express_canvas.sh`) to replay the built-in sample strokes and heartbeat line for demos.

See [API Reference](.documentation/api.md) for detailed usage.

## Contributing

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code standards and [Development Roadmap](.documentation/roadmap.md) for planned features.

## License

MIT License - see [LICENSE](LICENSE) for details.

