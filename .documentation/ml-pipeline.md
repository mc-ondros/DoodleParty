# DoodleParty ML Pipeline Documentation

**Purpose:** Complete documentation of the content moderation ML pipeline, model architecture, and inference strategies.

**Status: Updated to match actual implementation**

## Table of Contents

### Model Architecture
- [Model Architecture](#model-architecture)
  - [CNN Architecture](#cnn-architecture)
    - [Custom CNN Architecture](#custom-cnn-architecture)
    - [Model Specifications](#model-specifications)
    - [Training Configuration](#training-configuration)
    - [Transfer Learning Models](#transfer-learning-models)

### Data Processing
- [Data Preprocessing](#data-preprocessing)
  - [Image Preprocessing Pipeline](#image-preprocessing-pipeline)
    - [Key Steps](#key-steps)

### Detection Strategies
- [Detection Strategies](#detection-strategies)
  - [Strategy 1: Standard Single-Image Classification](#strategy-1-standard-single-image-classification)
    - [Pipeline](#pipeline)
    - [Characteristics](#characteristics)
  - [Strategy 2: Contour-Based Detection (Production)](#strategy-2-contour-based-detection-production)
    - [Pipeline](#pipeline-1)
    - [Key Features](#key-features)
    - [Hierarchical Detection](#hierarchical-detection)
  - [Strategy 3: Tile-Based Detection (Robust)](#strategy-3-tile-based-detection-robust)
    - [Pipeline](#pipeline-2)
    - [Key Features](#key-features-1)
    - [Tile Sizes](#tile-sizes)
    - [Use Case](#use-case)
  - [Strategy 4: Shape-Based Detection (Stroke-Aware)](#strategy-4-shape-based-detection-stroke-aware)
    - [Pipeline Overview](#pipeline-overview)

### Performance Considerations
- [Performance Targets](#performance-targets)
  - [Raspberry Pi 4 Deployment](#raspberry-pi-4-deployment)
    - [Critical Optimizations](#critical-optimizations)
      - [Model Optimization (MANDATORY)](#model-optimization-mandatory)
      - [Inference Optimization](#inference-optimization)
      - [Memory Management](#memory-management)
      - [System-Level Optimizations](#system-level-optimizations)
    - [TensorFlow Lite Conversion Pipeline](#tensorflow-lite-conversion-pipeline)
    - [RPi4 Inference Implementation](#rpi4-inference-implementation)
    - [Expected Performance on RPi4](#expected-performance-on-rpi4)
    - [Deployment Checklist for RPi4](#deployment-checklist-for-rpi4)
      - [Software Setup](#software-setup)
      - [Hardware Setup](#hardware-setup)
      - [Performance Tuning](#performance-tuning)
      - [Validation](#validation)

### Training Pipeline
- [Training Pipeline](#training-pipeline)
  - [Phase 1: Data Preparation](#phase-1-data-preparation)
    - [Data Organization](#data-organization)
  - [Phase 2: Model Training](#phase-2-model-training)
  - [Phase 3: Evaluation](#phase-3-evaluation)
  - [Data Augmentation](#data-augmentation)
    - [Techniques](#techniques)
    - [Data Source](#data-source)

### API Integration
- [Inference API](#inference-api)
  - [Key Endpoints Summary](#key-endpoints-summary)

### Documentation Resources
- [Further Documentation](#further-documentation)
  - [Implementation](#implementation)
  - [Operations](#operations)
  - [Development](#development)

## Model Architecture

### CNN Architecture

**Framework:** TensorFlow/Keras with TFLite optimization
**Base Model:** Custom CNN (423K parameters) or transfer learning
**Input:** 28x28 grayscale images (QuickDraw native format)
**Output:** Binary probability (0.0-1.0)

### Custom CNN Architecture

```
Input (28x28x1 grayscale)
    ↓
Conv2D 32 filters, 3x3 kernel (padding='same')
ReLU + BatchNorm + MaxPool(2x2) + Dropout(0.25)
    ↓ (14x14x32)
Conv2D 64 filters, 3x3 kernel (padding='same')
ReLU + BatchNorm + MaxPool(2x2) + Dropout(0.25)
    ↓ (7x7x64)
Conv2D 128 filters, 3x3 kernel (padding='same')
ReLU + BatchNorm + Dropout(0.25)
    ↓ (7x7x128)
Flatten
    ↓
Dense 256 units
ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense 1 unit
Sigmoid (Binary Classification)
    ↓
Output (0.0-1.0 probability)
```

**Model Specifications:**
- Total Parameters: ~423K
- Training Loss: Binary crossentropy
- Optimizer: Adam (lr=0.001)
- Metrics: Accuracy, Precision, Recall
- Lightweight design for edge deployment

### Transfer Learning Models

Available alternatives for higher accuracy:
- **ResNet50:** 23.5M parameters (higher accuracy, slower)
- **MobileNetV3:** 5.4M parameters (balanced)
- **EfficientNet:** 5.3M parameters (efficient)

**Training Configuration:**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

## Data Preprocessing

### Image Preprocessing Pipeline

**Training Data (QuickDraw):**
```python
def load_quickdraw_category(file_path):
    # QuickDraw .npy files are already 28x28 grayscale bitmaps
    data = np.load(file_path)  # Shape: (N, 784) or (N, 28, 28)
    
    # Normalize to [0, 1]
    data = data.astype(np.float32) / 255.0
    
    # Reshape to (N, 28, 28, 1) if needed
    if data.ndim == 2:
        data = data.reshape(-1, 28, 28, 1)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    
    return data
```

**Inference (Canvas → 28x28):**
```python
def preprocess_canvas_image(image):
    # 1. Convert to grayscale
    image = image.convert('L')

    # 2. Invert colors (canvas has white bg, model expects black)
    img_array = 255 - np.array(image)

    # 3. Apply morphological dilation to thicken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_array = cv2.dilate(img_array, kernel, iterations=1)

    # 4. Resize to 28x28 (model input size)
    image = Image.fromarray(img_array).resize((28, 28), Image.LANCZOS)

    # 5. Normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 6. Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array
```

**Key Steps:**
1. **Training:** Use native 28x28 QuickDraw .npy format (no preprocessing needed)
2. **Inference:** Resize canvas to 28x28 to match model input
3. Grayscale conversion and color inversion
4. Morphological dilation to preserve thin strokes
5. Normalization to [0, 1] range

## Detection Strategies

### Strategy 1: Standard Single-Image Classification

**Endpoint:** `POST /api/predict`

**Pipeline:**
```
Canvas (512x512) → Preprocess → Resize to 28x28 → TFLite INT8 Model → Binary Classification
```

**Characteristics:**
- Fastest inference (<30ms on RPi4)
- Simplest implementation
- Vulnerable to content dilution attacks
- Best for: Quick validation, low-risk scenarios

**Latency:** <30ms per image

### Strategy 2: Contour-Based Detection (Production)

**Endpoint:** `POST /api/predict/region`

**Pipeline:**
```
Canvas → OpenCV findContours(RETR_TREE) → Extract Hierarchical Contours → 
Classify Each Contour → Detect Nested Content → Aggregate Results
```

**Key Features:**
- Isolates individual shapes for independent classification
- Uses `cv2.RETR_TREE` by default for full hierarchical detection
- Detects nested content (e.g., offensive drawing inside a benign circle)
- Filters small contours (noise reduction)
- Supports multiple aggregation strategies: MAX, MEAN, WEIGHTED_MEAN, VOTING, ANY_POSITIVE
- Early stopping on first positive detection
- Optional `cv2.RETR_EXTERNAL` mode for faster detection (outer boundaries only)

**Latency:** ~125-135ms for 5-10 contours (including nested analysis)

**Hierarchical Detection:**
- Analyzes parent-child relationships between contours
- Detects offensive content hidden inside benign shapes
- Full hierarchy support with `RETR_TREE` mode

### Strategy 3: Tile-Based Detection (Robust)

**Endpoint:** `POST /api/predict/tile`

**Pipeline:**
```
Canvas → Fixed Grid (e.g., 8x8 = 64 tiles) → Dirty Tile Tracking → 
Per-Tile Inference → Tile Caching → Aggregate Results
```

**Key Features:**
- Divides canvas into fixed-size tiles (configurable: 32x32, 64x64, 128x128)
- Supports non-square canvas dimensions (dynamic grid calculation)
- Dirty tile tracking: only re-analyze tiles affected by new strokes
- Tile caching: cache predictions for unchanged tiles (incremental updates <1ms)
- Robust against content dilution attacks (each tile analyzed independently)

**Tile Sizes:**
- `64x64` (recommended): ~8x8 grid for 512x512 canvas, balanced performance
- `32x32` (high precision): ~16x16 grid, better for fine details, higher cost
- `128x128` (low budget): ~4x4 grid, minimal inference load

**Latency:**
- Full grid (64 tiles): ~342ms (mock model), <200ms expected with TFLite INT8
- Incremental update (1-4 tiles): <0.1ms with caching
- Single tile: ~5ms

**Use Case:** High-security scenarios where users may attempt to hide offensive content by mixing with innocent shapes across the canvas

### Strategy 4: Shape-Based Detection (Stroke-Aware)

**Endpoint:** `POST /api/predict/shape`

**Purpose:** Detect coherent offensive objects (especially multi-part penis drawings) by using stroke history, spatial configuration, and model confidence scores.

**Pipeline Overview:**

1. **Stroke-Aware Shape Proposals**
   - Source: `stroke_history` from the web UI
   - Normalizes each stroke into points and timestamps
   - Clusters strokes via union-find:
     - Connect strokes if endpoints within spatial radius (default ≤32px)
     - OR temporal gap ≤900ms
   - Each connected component becomes a candidate shape with its own bounding box
   - Rationale: Captures user intent better than raw pixels

2. **Robust Contour Fallback**
   - If stroke-based clustering yields no shapes:
     - Grayscale + Otsu/adaptive thresholding
     - Morphological closing to connect fragmented strokes
     - Area and aspect-ratio filters for noise rejection
   - Guarantees reliable fallback even without stroke metadata

3. **Per-Shape Normalization and Scoring**
   - For each candidate shape:
     - Crop with margin, preserve aspect ratio, normalize to model input size
     - Supports both Keras and TFLite backends
     - Converts arbitrary output tensors into stable scalar offensive score in [0,1]
   - Results captured as `ShapeInfo` with bounding box, confidence, area, etc.

4. **Grouping Logic (Merging Related Shapes)**
   - Operates on shapes flagged as positive by caller
   - Builds undirected graph where shapes are linked if:
     - Center distance ≤80px, OR
     - IoU ≥0.05
   - Each connected component becomes merged object with group score = max member confidence

5. **Penis-Specific Cluster Heuristic**
   - First, try strict grouping:
     - If any merged group contains shapes ≥ threshold → positive verdict
   - If no strict-positive cluster:
     - Identify shapes with confidence close to threshold (e.g. ≥0.45 for threshold 0.5)
     - If 3 or more such shapes form single merged cluster:
       - Treat as one coherent offensive object (classic shaft + two balls pattern)
       - Promote overall verdict to positive
       - Boost group confidence (capped below 1.0) to reflect combined evidence
   - Intent: Avoid "three 49% negatives" outcome for clear penis drawings

**Latency:** ~95ms with stroke awareness

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "stroke_history": [
    {
      "points": [
        {"x": 120, "y": 260, "t": 1730980000000},
        {"x": 160, "y": 260, "t": 1730980000100}
      ],
      "timestamp": 1730980000000
    }
  ],
  "min_shape_area": 100
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "APPROVED",
  "confidence": 0.12,
  "detection_details": {
    "num_shapes_analyzed": 3,
    "shape_predictions": [
      {
        "shape_id": 0,
        "x": 180,
        "y": 260,
        "width": 80,
        "height": 160,
        "confidence": 0.15,
        "is_positive": false,
        "area": 12800
      }
    ],
    "grouped_boxes": []
  }
}
```

## Performance Targets

### Raspberry Pi 4 Deployment

| Metric | Target | Critical? | Notes |
|--------|--------|-----------|-------|
| **Inference latency** | **<50ms** | **YES** | Per-stroke moderation must be fast |
| **Model size** | **<5MB** | **YES** | INT8 quantization mandatory (5MB vs 50MB) |
| **Memory usage** | **<500MB** | **YES** | ML service only; leaves 1.5-2GB for users |
| **Accuracy retention** | **>88%** | **YES** | After INT8 quantization |
| **Cold start time** | **<3s** | NO | Model loading on startup |
| **Multi-patch inference** | **<200ms** | YES | Tile/shape detection with 10+ patches |
| **CPU utilization** | **<80%** | NO | Avoid thermal throttling |
| **Batch throughput** | **50 req/sec** | YES | At <50ms latency per request |

### Development Environment

| Metric | Target |
|--------|--------|
| Training accuracy | >90% |
| Inference time | <100ms |
| Model size | <50MB |

## Raspberry Pi 4 Optimization

### Critical Optimizations

**1. Model Optimization (MANDATORY)**
- **INT8 Quantization:** Reduce model size by 4x and inference time by 2-4x
- **TensorFlow Lite Conversion:** Use optimized ARM runtime
- **Model Pruning:** Remove redundant weights (target 30-50% sparsity)
- **Architecture Simplification:** Consider reducing layer depth if needed

**2. Inference Optimization**
- **TFLite Interpreter:** Use `tflite_runtime` (lighter than full TensorFlow)
- **XNNPACK Delegate:** Enable ARM NEON SIMD acceleration
- **Thread Pool:** Configure TFLite to use 4 threads (all RPi4 cores)
- **Batch Inference:** Process multiple patches in single forward pass

**3. Memory Management**
- **Lazy Loading:** Load model only when needed
- **Memory Mapping:** Use mmap for model loading
- **Garbage Collection:** Explicit cleanup after inference
- **Swap Configuration:** Disable or minimize swap usage

**4. System-Level Optimizations**
- **CPU Governor:** Set to `performance` mode (avoid throttling)
- **Thermal Management:** Ensure adequate cooling (heatsink + fan)
- **Process Priority:** Run inference with higher priority
- **Minimal OS:** Use Raspberry Pi OS Lite (no desktop environment)

### TensorFlow Lite Conversion Pipeline

```python
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('quickdraw_model.h5')

# Create representative dataset for calibration
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 128, 128, 1).astype(np.float32)]

# Convert with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save optimized model
with open('quickdraw_model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

### RPi4 Inference Implementation

```python
import numpy as np
import tflite_runtime.interpreter as tflite

class RPi4Inference:
    def __init__(self, model_path):
        # Load TFLite model with optimizations
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=4  # Use all 4 cores
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, image):
        # Preprocess to uint8 (matches quantized model)
        input_data = np.array(image, dtype=np.uint8)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            input_data
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output[0][0]
```

### Expected Performance on RPi4

| Optimization Stage | Inference Time | Model Size | Accuracy |
|-------------------|----------------|------------|----------|
| Baseline (TF Keras) | ~300-500ms | 45MB | 92% |
| TFLite FP32 | ~150-200ms | 45MB | 92% |
| TFLite INT8 | **~30-50ms** | **<5MB** | ~90% |
| TFLite INT8 + XNNPACK | **~20-40ms** | **<5MB** | ~90% |

### Deployment Checklist for RPi4

**Software Setup:**
- [ ] Install Raspberry Pi OS Lite (64-bit)
- [ ] Update system: `sudo apt update && sudo apt upgrade`
- [ ] Install Python 3.9+ and pip
- [ ] Install `tflite_runtime` (not full TensorFlow)
- [ ] Install minimal dependencies: numpy, flask, pillow
- [ ] Convert model to TFLite INT8 format
- [ ] Copy optimized model to RPi4

**Hardware Setup:**
- [ ] Install heatsink and active cooling fan
- [ ] Verify power supply is 5V 3A minimum
- [ ] Use quality microSD card (UHS-I A1 or better)

**Performance Tuning:**
- [ ] Configure CPU governor to `performance`
- [ ] Disable unnecessary system services
- [ ] Disable swap or set swappiness to 10
- [ ] Set process priority for inference

**Validation:**
- [ ] Benchmark inference latency on actual hardware
- [ ] Verify accuracy retention (>88% required)
- [ ] Test under sustained load (thermal throttling check)
- [ ] Monitor CPU temperature (should stay <75°C)
- [ ] Monitor memory usage (<500MB target)

## Training Pipeline

### Phase 1: Data Preparation

```bash
# Download QuickDraw data (NumPy bitmap format, 28x28 pre-processed)
python scripts/data_processing/download_quickdraw_npy.py
```

**Data Organization:**
- `data/raw/` - Downloaded NumPy bitmap files (penis.npy, circle.npy, etc.)
- Format: Native 28x28 grayscale bitmaps from Google's QuickDraw dataset (no preprocessing needed)

### Phase 2: Model Training

**Option A: Train with .npy only (28x28 native)**
```bash
python scripts/training/train_binary_classifier.py \
  --data-dir data/raw \
  --epochs 30 \
  --batch-size 32 \
  --max-samples 10000
```

**Option B: Train with mixed datasets (.npy + appendix)**
```bash
# Combines QuickDraw .npy (28x28) + Appendix (128x128→28x28)
python scripts/training/train_mixed_dataset.py \
  --npy-dir data/raw \
  --appendix-dir data/appendix \
  --epochs 30 \
  --max-npy-samples 10000 \
  --max-appendix-samples 5000
```

**Outputs:** `models/quickdraw_binary_28x28.h5` or `models/quickdraw_mixed_28x28.h5`

**Note:** Appendix images (128x128) are automatically downscaled to 28x28 using `cv2.INTER_AREA` for optimal quality while maintaining model efficiency.

### Phase 3: Evaluation

```bash
python scripts/evaluate.py \
  --model models/quickdraw_model.h5
```

**Outputs:**
- Accuracy, precision, recall, F1
- Confusion matrix
- ROC curve

### Data Augmentation

**Techniques:**
- Rotation: ±15 degrees
- Translation: ±10% (width/height)
- Zoom: 90-110%
- Horizontal flip: 50% probability

**Data Sources:**
- **QuickDraw .npy:** Native 28x28 grayscale bitmaps (primary source)
- **QuickDraw Appendix:** 128x128 images (automatically downscaled to 28x28)
- Categories: penis (positive) + 21 common shapes (negative)
- See `docs/MIXED_DATASET_TRAINING.md` for mixed dataset details

## Inference API

*For complete endpoint details, request/response formats, and error handling, see the [API Reference](api.md#ml-inference-api).*

### Key Endpoints Summary

- `POST /api/predict` - Standard single-image classification
- `POST /api/predict/shape` - Shape-based detection with stroke awareness
- `POST /api/predict/tile` - Tile-based detection with grid partitioning
- `GET /api/health` - ML service health check

## Further Documentation

### Implementation
- [API Reference](api.md#ml-inference-api) - Complete inference API endpoints
- [Architecture Overview](architecture.md#content-moderation-pipeline) - System integration details

### Operations
- [Installation Guide](installation.md) - Complete deployment instructions
- [Nix Usage Guide](nix-usage.md) - NixOS-specific deployment instructions

### Development
- [Testing Strategy](testing.md) - ML component testing
- [Project Structure](structure.md) - Code organization

*ML Pipeline documentation for DoodleParty v1.0*
