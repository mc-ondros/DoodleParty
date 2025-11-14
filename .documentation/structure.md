# DoodleParty Project Structure

**Purpose:** Documentation of file organization and directory structure.

**Status: Updated to match actual code**

## Table of Contents

### Project Organization
- [Current Structure](#current-structure)
  - [Structure Benefits](#structure-benefits)
- [Directory Details](#directory-details)
  - [Frontend (`src/`)](#frontend-src)
    - [Components (`src/components/`)](#components-srccomponents)
    - [Hooks (`src/hooks/`)](#hooks-srchooks)
    - [Services (`src/services/`)](#services-srcservices)
  - [Backend (`src_py/`)](#backend-src_py)
    - [Core ML (`src_py/core/`)](#core-ml-src_pycore)
    - [Data (`src_py/data/`)](#data-src_pydata)
    - [Web (`src_py/web/`)](#web-src_pyweb)
  - [Scripts (`scripts/`)](#scripts-scripts)
    - [Training (`scripts/training/`)](#training-scriptstraining)
    - [Evaluation (`scripts/evaluation/`)](#evaluation-scriptsevaluation)
    - [Data Processing (`scripts/data_processing/`)](#data-processing-scriptsdata_processing)
    - [Deployment (`scripts/deployment/`)](#deployment-scriptsdeployment)

### Code Organization
- [File Naming Conventions](#file-naming-conventions)
  - [TypeScript/JavaScript](#typescriptjavascript)
  - [Python](#python)
  - [Configuration](#configuration)
  - [Documentation](#documentation)
- [Import Examples](#import-examples)
  - [React Components](#react-components)
  - [Python ML](#python-ml)

### Architecture Flow
- [Data Flow Architecture](#data-flow-architecture)
  - [Frontend to Backend](#frontend-to-backend)
  - [Game Mode Flow](#game-mode-flow)

### Build & Deployment
- [Build and Deployment](#build-and-deployment)
  - [Development Build](#development-build)
  - [Production Build](#production-build)
  - [Docker Build](#docker-build)
  - [Kubernetes Deployment](#kubernetes-deployment)
- [Environment Configuration](#environment-configuration)
  - [Development (`.env.development`)](#development-envdevelopment)
  - [Production (`.env.production`)](#production-envproduction)

### Resources
- [Related Documentation](#related-documentation)

## Current Structure

The project follows best practices with clear separation of concerns:

```
DoodleParty/
├── src/                       # Source code
│   ├── components/            # React components
│   │   ├── DrawingCanvas.tsx  # Main drawing canvas
│   │   ├── DrawerView.tsx     # Drawer/sidebar
│   │   ├── InkMeter.tsx       # Ink level display
│   │   ├── ModerationShield.tsx # Moderation feedback
│   │   └── GameModeSelector.tsx # Game mode UI
│   ├── hooks/                 # React hooks
│   │   ├── useDraggable.tsx   # Drawing input handling
│   │   ├── useGameMode.tsx    # Game mode logic
│   │   └── useLeaderboard.tsx # Leaderboard state
│   ├── services/              # Business logic
│   │   ├── socketService.tsx  # WebSocket client
│   │   ├── moderationService.ts # ML integration
│   │   ├── gameService.ts     # Game mode logic
│   │   └── analyticsService.ts # Event tracking
│   ├── App.tsx                # Root component
│   ├── index.tsx              # Entry point
│   └── types.ts               # TypeScript types
├── public/                    # Static assets
│   ├── css/
│   │   └── main.css
│   ├── js/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── utils/
│   └── index.html
├── scripts/                   # Utility scripts
│   ├── training/              # ML training scripts
│   │   ├── train.py           # Main training CLI
│   │   ├── train_binary_classifier.py
│   │   └── train_mixed_dataset.py
│   ├── evaluation/            # Model evaluation
│   ├── data_processing/       # Data download/processing
│   │   ├── download_quickdraw_npy.py  # Download safe categories
│   │   ├── download_appendix.py       # Download explicit content
│   │   └── download_with_curl.sh      # Alternative curl-based downloader
│   └── deployment/            # Deployment scripts
├── src_py/                    # Python ML backend (renamed from src-py for Python compatibility)
│   ├── __init__.py            # Package init
│   ├── core/                  # ML core
│   │   ├── __init__.py
│   │   ├── models.py          # CNN architectures (Keras 3 compatible)
│   │   ├── training.py        # Training logic with Rich TUI
│   │   ├── inference.py       # Prediction
│   │   ├── shape_detection.py # Shape-based detection
│   │   ├── tile_detection.py  # Tile-based detection
│   │   └── optimization.py    # Model optimization
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py         # QuickDraw & Appendix dataset loading
│   │   └── augmentation.py    # Data augmentation
│   └── web/                   # Flask ML service
│       ├── __init__.py
│       ├── app.py             # Flask server
│       └── routes.py          # API routes
├── models/                    # Trained models
│   ├── quickdraw_model.h5
│   ├── quickdraw_model.tflite
│   └── quickdraw_model_int8.tflite
├── data/                      # Dataset storage
│   ├── raw/                   # QuickDraw NumPy files
│   └── processed/             # Processed data
├── .documentation/            # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── installation.md
│   ├── ml-pipeline.md
│   ├── roadmap.md
│   ├── structure.md
│   ├── troubleshooting.md
│   └── nix-usage.md
├── k8s/                       # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment-web.yaml
│   ├── deployment-ml.yaml
│   └── service.yaml
├── .github/                   # GitHub workflows
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
├── package.json               # Node.js dependencies
├── tsconfig.json              # TypeScript config
├── requirements.txt           # Python dependencies
├── requirements-rpi4.txt      # RPi4-specific deps
├── vite.config.ts             # Vite config
├── flake.nix                  # Nix flake (TensorFlow 2.20 + Keras 3)
├── shell.nix                  # Nix shell for non-flake users
├── module.nix                 # NixOS module
├── STYLE_GUIDE.md             # Code standards
└── README.md                  # Project overview
```

## Structure Benefits

1. ✅ **Organized `src/` directory** - Clear separation by functionality
2. ✅ **Proper package structure** - Can import as `from src_py.core import models`
3. ✅ **Separated scripts** - Utility scripts in `scripts/`, library code in `src_py/`
4. ✅ **TypeScript support** - Full type safety for React components
5. ✅ **Python ML backend** - Separate `src_py/` for ML pipeline (Python-compatible naming)
6. ✅ **Kubernetes ready** - K8s manifests for cloud deployment
7. ✅ **Modern tooling** - Vite for fast development, TypeScript for type safety
8. ✅ **NixOS support** - Reproducible development environment with Keras 3 compatibility

## Directory Details

### Frontend (`src/`)

**Components (`src/components/`):**
- `DrawingCanvas.tsx` - Main canvas component with drawing logic
- `DrawerView.tsx` - Side drawer with game mode selection
- `InkMeter.tsx` - Ink level indicator
- `ModerationShield.tsx` - Content moderation feedback UI
- `GameModeSelector.tsx` - Game mode selection interface
- `Leaderboard.tsx` - Real-time leaderboard display
- `Timer.tsx` - Game timer display

**Hooks (`src/hooks/`):**
- `useDraggable.tsx` - Mouse/touch drawing input handling
- `useGameMode.tsx` - Game mode state management
- `useLeaderboard.tsx` - Leaderboard data fetching
- `useSocket.tsx` - WebSocket connection management

**Services (`src/services/`):**
- `socketService.tsx` - Socket.io client wrapper
- `moderationService.ts` - ML inference API calls
- `gameService.ts` - Game mode logic
- `analyticsService.ts` - Event tracking

### Backend (`src_py/`)

**Note:** Renamed from `src-py/` to `src_py/` for Python import compatibility (hyphens not allowed in module names).

**Core ML (`src_py/core/`):**
- `models.py` - CNN architectures with Keras 3 compatibility (Custom, ResNet50, MobileNetV3, EfficientNet)
- `training.py` - Training loops with Rich TUI progress and callbacks
- `inference.py` - Single/batch prediction
- `shape_detection.py` - Shape-based detection with stroke awareness
- `tile_detection.py` - Tile-based grid detection
- `optimization.py` - Model quantization and optimization

**Data (`src_py/data/`):**
- `loaders.py` - QuickDraw & Quickdraw Appendix dataset loading
- `augmentation.py` - Data augmentation pipeline

**Web (`src_py/web/`):**
- `app.py` - Flask ML service with REST endpoints
- `routes.py` - API route definitions

### Scripts (`scripts/`)

**Training (`scripts/training/`):**
- `train.py` - CLI for model training

**Evaluation (`scripts/evaluation/`):**
- `evaluate.py` - Model evaluation
- `optimize_threshold.py` - Threshold optimization
- `benchmark_tile_detection.py` - Performance benchmarking

**Data Processing (`scripts/data_processing/`):**
- `download_quickdraw_npy.py` - Download safe QuickDraw categories (Python)
- `download_appendix.py` - Download/convert explicit content from Quickdraw Appendix (Python)
- `download_with_curl.sh` - Bash downloader for safe categories (better NixOS compatibility)
- `download_appendix.sh` - Bash downloader for explicit content (recommended for NixOS)
- `process_all_data_128x128.py` - Data preprocessing

**Deployment (`scripts/deployment/`):**
- `deploy-rpi4.sh` - RPi4 deployment script
- `deploy-cloud.sh` - Cloud deployment script

## File Naming Conventions

**TypeScript/JavaScript:**
- Components: `PascalCase.tsx`
- Services: `camelCase.ts`
- Hooks: `camelCase.tsx`
- Utilities: `camelCase.ts`

**Python:**
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`

**Configuration:**
- `package.json` - Node.js dependencies
- `tsconfig.json` - TypeScript configuration
- `vite.config.ts` - Vite build configuration
- `requirements.txt` - Python dependencies

**Documentation:**
- `README.md` - Project overview
- `STYLE_GUIDE.md` - Code standards
- `.documentation/*.md` - Technical documentation

## Import Examples

**React Components:**
```typescript
import { DrawingCanvas } from '@/components/DrawingCanvas';
import { useDraggable } from '@/hooks/useDraggable';
import { socketService } from '@/services/socketService';
```

**Python ML:**
```python
from src_py.core.models import build_custom_cnn
from src_py.core.inference import predict_image
from src_py.data.loaders import load_quickdraw_dataset
```

## Data Flow Architecture

### Frontend to Backend

```
React Component
    ↓
useDraggable Hook (Input Handling)
    ↓
socketService (WebSocket)
    ↓
Node.js Server (Express + Socket.io)
    ↓
moderationService (REST API)
    ↓
Flask ML Service (Python)
    ↓
TensorFlow Lite Model
    ↓
Moderation Result
    ↓ WebSocket Broadcast
All Connected Clients
```

### Game Mode Flow

```
GameModeSelector Component
    ↓
gameService (Game Logic)
    ↓
Socket.io Events
    ↓
Node.js Game Manager
    ↓
LLM Service (Prompt Generation)
    ↓
Leaderboard Update
    ↓ WebSocket Broadcast
All Players
```

## Build and Deployment

### Development Build

```bash
npm run dev
```

Starts Vite dev server with hot module replacement.

### Production Build

```bash
npm run build
```

Optimized bundle in `dist/` directory.

### Docker Build

```bash
docker build -t doodleparty:latest .
docker run -p 3000:3000 doodleparty:latest
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

## Environment Configuration

**Development (`.env.development`):**
```
VITE_API_URL=http://localhost:3000
VITE_ML_URL=http://localhost:5001
VITE_ENV=development
```

**Production (`.env.production`):**
```
VITE_API_URL=https://api.doodleparty.io
VITE_ML_URL=https://ml.doodleparty.io
VITE_ENV=production
```

## Related Documentation

- [Architecture](architecture.md) - System design
- [API Reference](api.md) - API documentation
- [Installation](installation.md) - Setup guide
- [README](../README.md) - Project overview
- [Testing Strategy](testing.md) - Testing approach and implementation

*Project structure documentation for DoodleParty v1.0*
