# DoodleHunter API Reference

**Purpose:** Complete API documentation for Flask endpoints and Python module interfaces.

**Status: Updated to match actual implementation** - Nov 2024

## Table of Contents

1. [Flask Web API](#flask-web-api)
2. [Python Module API](#python-module-api)
3. [Model Inference API](#model-inference-api)
4. [Error Handling](#error-handling)

## Flask Web API

**Base URL:** `http://localhost:5000`

### `GET /`

Serve the main drawing interface.

**Response:**
- HTML page with drawing canvas

**Status Codes:**
- `200 OK` - Page loaded successfully

### `POST /api/predict`

Classify a hand-drawn image using standard single-image prediction.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "PENIS" | "OTHER_SHAPE",
  "verdict_text": "Drawing looks like a penis! ✓",
  "confidence": 0.95,
  "raw_probability": 0.95,
  "threshold": 0.5,
  "model_info": "Binary classifier: penis vs 21 common shapes (TFLite INT8)",
  "drawing_statistics": {
    "response_time_ms": 45.2,
    "preprocess_time_ms": 12.3,
    "inference_time_ms": 28.7
  }
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid image data
- `500 Internal Server Error` - Model inference failed

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'
```

### `POST /api/predict/region`

Classify a hand-drawn image using region-based detection (contour extraction).

**Detection Method:** Uses OpenCV `findContours(RETR_TREE)` by default to isolate individual shapes with full hierarchical analysis, then classifies each contour independently. This prevents content dilution attacks where offensive content is mixed with innocent shapes, and detects nested offensive content inside benign shapes.

**Hierarchical Detection:** Analyzes parent-child relationships between contours to detect offensive content hidden inside benign shapes (e.g., offensive drawing inside a circle). Can optionally use `RETR_EXTERNAL` mode for faster detection without nested analysis.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "mode": "tree",  // Optional: "tree" (default) or "external"
  "min_contour_area": 100,  // Optional: minimum contour area to analyze
  "early_stopping": true  // Optional: stop on first high-confidence detection
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "PENIS" | "OTHER_SHAPE",
  "verdict_text": "Drawing looks like a penis! ✓ (Detected via region analysis)",
  "confidence": 0.87,
  "raw_probability": 0.87,
  "threshold": 0.5,
  "model_info": "Binary classifier with region-based detection (TFLite INT8)",
  "detection_details": {
    "num_patches_analyzed": 5,
    "early_stopped": false,
    "aggregation_strategy": "shape_max",
    "patch_predictions": [
      {"x": 0, "y": 0, "confidence": 0.87, "is_positive": true},
      {"x": 64, "y": 0, "confidence": 0.23, "is_positive": false}
    ]
  },
  "drawing_statistics": {
    "response_time_ms": 125.4,
    "preprocess_time_ms": 45.2,
    "inference_time_ms": 76.8
  }
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid image data
- `500 Internal Server Error` - Model inference failed

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict/region \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'
```

### `POST /api/predict/tile` (Experimental)

Classify using tile-based detection with grid partitioning. This is the most robust detection mode, designed to prevent content dilution attacks by analyzing the canvas in independent tiles.

**Detection Method:** Divides canvas into a fixed grid (e.g., 8x8 for 512x512 canvas using 64x64 tiles). Only re-analyzes "dirty" tiles affected by new strokes. Supports non-square canvas dimensions with dynamic grid calculation.

**Tile Sizes:**
- `64x64` (recommended): ~8x8 grid for 512x512 canvas, balanced performance
- `32x32` (high precision): ~16x16 grid, better for fine details, higher cost
- `128x128` (low budget): ~4x4 grid, minimal inference load

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "strokes": [
    {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 25}]}
  ],
  "tile_size": 64
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "PENIS" | "OTHER_SHAPE",
  "verdict_text": "Drawing looks like a penis! ✓ (Tile-based detection)",
  "confidence": 0.92,
  "raw_probability": 0.92,
  "threshold": 0.5,
  "model_info": "Binary classifier with tile-based detection (TFLite INT8)",
  "detection_details": {
    "num_tiles_analyzed": 12,
    "total_tiles": 64,
    "grid_size": 8,
    "cached": false,
    "aggregation_strategy": "tile_max"
  },
  "drawing_statistics": {
    "response_time_ms": 85.3,
    "preprocess_time_ms": 22.1,
    "inference_time_ms": 58.7
  }
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid image data
- `500 Internal Server Error` - Model inference failed or tile detector not available

### `POST /api/tile/reset`

Reset tile detector state and clear cached predictions.

**Request Body:** (empty)

**Response:**
```json
{
  "success": true,
  "message": "Tile detector reset"
}
```

**Status Codes:**
- `200 OK` - Reset successful
- `500 Internal Server Error` - Tile detector not available

### `GET /api/health`

Health check endpoint with model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "TFLite INT8 (Optimized)",
  "model_type": "TFLite",
  "threshold": 0.5,
  "region_detection_available": true
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `500 Internal Server Error` - Service unhealthy

## Python Module API

### Data Loading (`src/data/loaders.py`)

#### `load_data(data_dir, categories, max_samples_per_class=10000)`

Load QuickDraw dataset for training.

**Parameters:**
- `data_dir` (str): Path to data directory
- `categories` (list): List of category names to load
- `max_samples_per_class` (int): Maximum samples per category

**Returns:**
- `tuple`: (X_train, y_train, X_test, y_test, class_mapping)

**Example:**
```python
from src.data.loaders import QuickDrawDataset

dataset = QuickDrawDataset(data_dir='data/raw')
# Load specific categories
dataset.load_category('penis')
dataset.load_category('circle')
dataset.load_category('square')
# Get processed data
X_train, y_train, X_test, y_test = dataset.prepare_data()
```

#### `preprocess_image(image, target_size=(128, 128))`

Preprocess image for model input.

**Parameters:**
- `image` (np.ndarray): Input image
- `target_size` (tuple): Target dimensions

**Returns:**
- `np.ndarray`: Preprocessed image (normalized, resized)

### Model Architecture (`src/core/models.py`)

#### `create_cnn_model(input_shape=(128, 128, 1), num_classes=2)`

Create CNN model for binary classification.

**Parameters:**
- `input_shape` (tuple): Input image shape (H, W, C)
- `num_classes` (int): Number of output classes

**Returns:**
- `keras.Model`: Compiled model

**Example:**
```python
from src.core.models import create_cnn_model

model = create_cnn_model(
    input_shape=(128, 128, 1),
    num_classes=2
)
```

### Training (`scripts/train.py`, `src/core/training.py`)

#### `train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)`

Train the classification model.

**Parameters:**
- `model` (keras.Model): Model to train
- `X_train` (np.ndarray): Training images
- `y_train` (np.ndarray): Training labels
- `X_val` (np.ndarray): Validation images
- `y_val` (np.ndarray): Validation labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size

**Returns:**
- `keras.callbacks.History`: Training history

**Example:**
```python
from src.core.training import train_model

history = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=50,
    batch_size=32
)
```

### Evaluation (`scripts/evaluate.py`)

#### `evaluate_model(model, X_test, y_test)`

Evaluate model performance.

**Parameters:**
- `model` (keras.Model): Trained model
- `X_test` (np.ndarray): Test images
- `y_test` (np.ndarray): Test labels

**Returns:**
- `dict`: Metrics (accuracy, precision, recall, f1_score)

**Example:**
```python
# Run evaluation script
python scripts/evaluate.py --model models/quickdraw_model.h5

# Or use the module directly
from src.core.inference import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Model Inference API

### Prediction (`src/core/inference.py`)

#### `predict_single(model, image, threshold=0.5)`

Predict class for a single image.

**Parameters:**
- `model` (keras.Model): Trained model
- `image` (np.ndarray): Input image
- `threshold` (float): Classification threshold

**Returns:**
- `dict`: Prediction result

**Example:**
```python
from src.core.inference import predict_single

result = predict_single(
    model=model,
    image=test_image,
    threshold=0.5
)
# {'class': 'positive', 'confidence': 0.87, 'prediction': 1}
```

#### `predict_batch(model, images, threshold=0.5)`

Predict classes for multiple images.

**Parameters:**
- `model` (keras.Model): Trained model
- `images` (np.ndarray): Batch of images
- `threshold` (float): Classification threshold

**Returns:**
- `list`: List of prediction dictionaries

## Error Handling

### Error Response Format

**Flask API Errors:**
```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Invalid base64 image data",
    "details": "Could not decode image"
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_IMAGE` | Image data is invalid or corrupted | 400 |
| `MODEL_NOT_LOADED` | Model file not found or failed to load | 503 |
| `INFERENCE_FAILED` | Model prediction failed | 500 |
| `INVALID_INPUT` | Input validation failed | 400 |

### Exception Handling

**Python Modules:**

All functions follow consistent error handling:

```python
try:
    result = function_call()
except ValueError as e:
    # Input validation errors
    logger.error(f"Validation error: {e}")
    raise
except FileNotFoundError as e:
    # Missing files
    logger.error(f"File not found: {e}")
    raise
except Exception as e:
    # Unexpected errors
    logger.error(f"Unexpected error: {e}")
    raise
```

## Configuration

### Environment Variables

```bash
# Flask configuration
FLASK_PORT=5000
FLASK_DEBUG=False

# Model configuration
MODEL_PATH=models/quickdraw_model.h5
IMAGE_SIZE=128
THRESHOLD=0.5

# Data paths
DATA_DIR=data/raw
PROCESSED_DIR=data/processed
```

### Model Configuration

**Training Parameters:**
```python
# scripts/train.py
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
```

**Data Augmentation:**
```python
# src/data/augmentation.py
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1
```

## Examples

### Complete Training Workflow

```python
from src.data.loaders import load_data
from src.core.models import create_cnn_model
from src.core.training import train_model
from src.core.inference import evaluate_model

# Load data
X_train, y_train, X_test, y_test, mapping = load_data(
    data_dir='data/raw_ndjson',
    categories=['penis', 'circle', 'square', 'triangle'],
    max_samples_per_class=10000
)

# Create model
model = create_cnn_model(input_shape=(128, 128, 1))

# Train
history = train_model(
    model, X_train, y_train, X_test, y_test,
    epochs=50, batch_size=32
)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")

# Save
model.save('models/quickdraw_model.h5')
```

### Flask API Usage

```python
import requests
import base64

# Load image
with open('test_drawing.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'image': f'data:image/png;base64,{image_data}'}
)

result = response.json()
print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Related Documentation

- [Architecture](architecture.md) - System design
- [Installation](installation.md) - Setup guide
- [Testing](testing.md) - Testing strategy
- [README](../README.md) - Project overview

*API Reference for DoodleHunter v1.0*
