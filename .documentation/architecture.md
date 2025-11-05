# DoodleHunter System Architecture

**Purpose:** Technical documentation of system design, component interactions, and data flow for DoodleHunter binary classification system.

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Web Application Architecture](#web-application-architecture)
- [Training Pipeline](#training-pipeline)

## Overview

DoodleHunter uses a modular Python/TensorFlow architecture with three primary components:

1. **Training Pipeline** - Data loading, preprocessing, model training
2. **Flask Web Application** - Real-time drawing classification interface
3. **Model Inference** - CNN-based binary classification

The system prioritizes accuracy and ease of use for content moderation tasks.

**Key Design Decisions:**
- TensorFlow/Keras for ML framework (widely supported, easy deployment)
- Flask for web interface (lightweight, Python-native)
- QuickDraw dataset for training data (large, diverse, free)
- Binary classification (simplifies model and improves accuracy)

## System Components

### Component Overview

| Component | Technology | Responsibility | Location |
|-----------|-----------|----------------|----------|
| **Data Pipeline** | Python + NumPy | Data loading and preprocessing | `src/data/loaders.py`, `src/data/augmentation.py` |
| **Model Training** | TensorFlow/Keras | CNN model training | `scripts/train.py`, `src/core/training.py` |
| **Web Interface** | Flask + HTML5 Canvas | Drawing and classification UI | `src/web/app.py` |
| **Inference Engine** | TensorFlow/Keras | Real-time prediction | `src/core/inference.py` |

### Component Interactions

```mermaid
graph TB
    subgraph Training["Training Pipeline"]
        QD[QuickDraw Dataset] --> DP[Data Processing]
        DP --> MT[Model Training]
        MT --> TM[Trained Model<br/>.h5/.keras]
    end
    
    subgraph WebApp["Web Application"]
        DC[Drawing Canvas<br/>HTML5] --> MI[Model Inference]
        MI --> FS[Flask Server<br/>Port 5000]
        FS --> CR[Classification Result]
        CR --> DC
    end
    
    TM --> MI
```

## Data Flow

### Training Data Flow

```mermaid
graph LR
    A[Download QuickDraw<br/>NPY format] --> B[Load & Preprocess<br/>128x128, normalize]
    B --> C[Data Augmentation<br/>rotation, translation, zoom]
    C --> D[Train CNN Model<br/>binary crossentropy]
    D --> E[Save Model<br/>quickdraw_classifier.h5]
```

### Inference Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Canvas as HTML5 Canvas
    participant Server as Flask Server
    participant Model as CNN Model
    
    User->>Canvas: Draw on canvas
    Canvas->>Server: POST /predict (base64 image)
    Server->>Server: Preprocess image<br/>(decode, grayscale, resize, normalize)
    Server->>Model: Run inference
    Model->>Server: Confidence score (0.0-1.0)
    Server->>Server: Apply threshold (0.5)
    Server->>Canvas: Return {class, confidence}
    Canvas->>User: Display result
```

## Model Architecture

### CNN Architecture

```mermaid
graph TB
    Input[Input<br/>128x128x1 grayscale] --> Conv1[Conv2D 32, 3x3<br/>ReLU + MaxPool 2x2]
    Conv1 --> Conv2[Conv2D 64, 3x3<br/>ReLU + MaxPool 2x2]
    Conv2 --> Conv3[Conv2D 128, 3x3<br/>ReLU + MaxPool 2x2]
    Conv3 --> Flatten[Flatten]
    Flatten --> Dense1[Dense 128<br/>ReLU + Dropout 0.5]
    Dense1 --> Dense2[Dense 1<br/>Sigmoid]
    Dense2 --> Output[Output<br/>Binary probability]
```

**Model Specifications:**
- Input: 128x128 grayscale images
- Output: Single probability value (0.0-1.0)
- Loss: Binary crossentropy
- Optimizer: Adam (lr=0.001)
- Metrics: Accuracy, precision, recall

**Training Configuration:**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

### Data Preprocessing

**Image Preprocessing Pipeline:**
```python
def preprocess_image(image):
    # 1. Convert to grayscale
    image = image.convert('L')
    
    # 2. Invert colors (canvas has white bg, model expects black bg)
    img_array = 255 - np.array(image)
    
    # 3. Resize to 128x128
    image = Image.fromarray(img_array).resize((128, 128))
    
    # 4. Normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 5. Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array
```

## Web Application Architecture

### Flask Application Structure

**Directory Layout:**
- `src/web/app.py` - Flask server
- `src/web/templates/index.html` - Drawing interface
- `src/web/static/css/style.css` - Styles
- `src/web/static/js/canvas.js` - Canvas drawing logic

**Flask Routes:**
- `GET /` - Serve drawing interface
- `POST /predict` - Classify drawing
- `GET /health` - Health check

**Request/Response Flow:**

```mermaid
sequenceDiagram
    participant Browser
    participant Flask
    participant Model
    
    Browser->>Flask: GET /
    Flask->>Browser: HTML page with canvas
    Browser->>Browser: User draws
    Browser->>Flask: POST /predict {image: base64}
    Flask->>Flask: Preprocess image
    Flask->>Model: Inference
    Model->>Flask: Confidence score
    Flask->>Browser: {class, confidence}
    Browser->>Browser: Display result
```

## Training Pipeline

### Training Workflow

**Phase 1: Data Preparation**
```bash
# Download QuickDraw data
python scripts/data_processing/download_quickdraw_ndjson.py
```

**Data Organization:**
- `data/raw_ndjson/` - Downloaded NDJSON files (penis-raw.ndjson, circle-raw.ndjson, etc.)
- `data/processed/` - Processed data and class_mapping.pkl

**Phase 2: Model Training**
```python
# Train model
python scripts/train.py \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001

# Output: models/quickdraw_classifier.keras
```

**Phase 3: Evaluation**
```python
# Evaluate model
python scripts/evaluate.py \
  --model models/quickdraw_classifier.keras

# Outputs:
# - Accuracy, precision, recall, F1
# - Confusion matrix
# - ROC curve
```

### Data Augmentation

**Augmentation Techniques:**
- Rotation: ±15 degrees
- Translation: ±10% (width/height)
- Zoom: 90-110%
- Horizontal flip: 50% probability

**Implementation:**
```python
# See src/data/augmentation.py for implementation
from src.data.augmentation import create_augmentation_pipeline

datagen = create_augmentation_pipeline(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
```

## Performance Targets

| Metric | Target | Measured At |
|--------|--------|-------------|
| Training accuracy | >90% | Validation set |
| Inference time | <100ms | Single image |
| Model size | <50MB | Saved file |
| Web response time | <200ms | End-to-end |

## Scalability Considerations

**Current Limitations:**
- Single-threaded Flask server
- Model loaded in memory (not optimized for concurrent requests)
- No caching of predictions

**Future Improvements:**
- Use Gunicorn for multi-worker deployment
- Implement model caching and batching
- Add Redis for prediction caching
- Convert to TensorFlow Lite for faster inference
- Deploy on cloud (AWS Lambda, Google Cloud Run)

## Security Architecture

**Input Validation:**
- Validate base64 image format
- Limit image size (<5MB)
- Sanitize file paths
- Prevent path traversal attacks

**Error Handling:**
- Never expose stack traces to client
- Log errors server-side
- Return generic error messages

**Rate Limiting:**
- Limit requests per IP (optional)
- Prevent abuse of prediction endpoint

## Related Documentation

- [API Reference](api.md) - Detailed API documentation
- [Installation](installation.md) - Setup instructions
- [Testing](testing.md) - Testing strategy
- [README](../README.md) - Project overview

*Architecture documentation for DoodleHunter v1.0*
