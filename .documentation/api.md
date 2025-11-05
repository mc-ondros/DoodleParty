# DoodleHunter API Reference

**Purpose:** Complete API documentation for Flask endpoints and Python module interfaces.

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

### `POST /predict`

Classify a hand-drawn image.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**Response:**
```json
{
  "class": "positive" | "negative",
  "confidence": 0.95,
  "threshold": 0.5
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid image data
- `500 Internal Server Error` - Model inference failed

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/quickdraw_classifier.h5"
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `503 Service Unavailable` - Model not loaded

## Python Module API

### Data Loading (`src/dataset.py`)

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
from src.dataset import load_data

X_train, y_train, X_test, y_test, mapping = load_data(
    data_dir='data/raw',
    categories=['penis', 'circle', 'square'],
    max_samples_per_class=5000
)
```

#### `preprocess_image(image, target_size=(128, 128))`

Preprocess image for model input.

**Parameters:**
- `image` (np.ndarray): Input image
- `target_size` (tuple): Target dimensions

**Returns:**
- `np.ndarray`: Preprocessed image (normalized, resized)

### Model Architecture (`src/models.py`)

#### `create_cnn_model(input_shape=(128, 128, 1), num_classes=2)`

Create CNN model for binary classification.

**Parameters:**
- `input_shape` (tuple): Input image shape (H, W, C)
- `num_classes` (int): Number of output classes

**Returns:**
- `keras.Model`: Compiled model

**Example:**
```python
from src.models import create_cnn_model

model = create_cnn_model(
    input_shape=(128, 128, 1),
    num_classes=2
)
```

### Training (`src/train.py`)

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
from src.train import train_model

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

### Evaluation (`src/evaluate.py`)

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
from src.evaluate import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Model Inference API

### Prediction (`src/predict.py`)

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
from src.predict import predict_single

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
MODEL_PATH=models/quickdraw_classifier.h5
IMAGE_SIZE=128
THRESHOLD=0.5

# Data paths
DATA_DIR=data/raw
PROCESSED_DIR=data/processed
```

### Model Configuration

**Training Parameters:**
```python
# src/train.py
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
```

**Data Augmentation:**
```python
# src/data_pipeline.py
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1
```

## Examples

### Complete Training Workflow

```python
from src.dataset import load_data
from src.models import create_cnn_model
from src.train import train_model
from src.evaluate import evaluate_model

# Load data
X_train, y_train, X_test, y_test, mapping = load_data(
    data_dir='data/raw',
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
model.save('models/quickdraw_classifier.h5')
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
