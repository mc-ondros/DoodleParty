# DoodleHunter 🎨

A high-resolution CNN binary classifier that distinguishes between penis drawings and common shapes from the QuickDraw dataset.

## Overview

**DoodleHunter** uses TensorFlow/Keras to train a convolutional neural network (CNN) for binary classification:
- **Positive class (1)**: Penis drawings from custom NDJSON dataset
- **Negative class (0)**: 21 common shapes from Google's QuickDraw dataset

The model achieves **97.25% accuracy** on 128×128 high-resolution images rendered from vector strokes.

## Project Structure

```
├── app/                          # Flask web application
│   ├── app.py                   # Backend API server
│   ├── static/                  # CSS, JavaScript
│   ├── templates/               # HTML templates
│   └── requirements.txt         # Web app dependencies
├── data/
│   ├── raw/                     # Raw NDJSON stroke data (21 QuickDraw classes)
│   └── processed/               # Preprocessed 128×128 numpy arrays
│       ├── X_train.npy         # Training images (40,320 samples, 2.5GB)
│       ├── X_test.npy          # Test images (10,080 samples, 630MB)
│       ├── y_train.npy         # Training labels
│       ├── y_test.npy          # Test labels
│       └── class_mapping.pkl   # Class to index mapping
├── models/
│   └── quickdraw_model.h5      # Trained CNN model (296MB, 25.8M params)
├── scripts/
│   ├── data_processing/        # Data preparation scripts
│   └── visualization/          # Visualization utilities
├── src/
│   ├── dataset.py              # Data loading and preprocessing
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference and evaluation
│   └── models.py               # Model architecture definitions
├── docs/
│   └── ML_API_DOCUMENTATION.md # Comprehensive API documentation
├── quickdraw_appendix/         # Custom penis drawing dataset
└── requirements.txt            # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mc-ondros/DoodleHunter.git
cd DoodleHunter
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download and Process Data

```bash
# Download QuickDraw NDJSON files (vector strokes, not pre-rendered)
python scripts/data_processing/download_quickdraw_ndjson.py

# Process all data to 128×128 from vector strokes
python scripts/data_processing/process_all_data_128x128.py

# Generate training/test splits (80/20)
python scripts/data_processing/regenerate_training_data.py
```

### 2. Train the Model

```bash
# Train for 50 epochs with batch size 32
python src/train.py

# Model saved to: models/quickdraw_model.h5
# Training takes ~8 hours on CPU (Intel with AVX512)
```

### 3. Run the Web Interface

```bash
cd app
flask run --host=0.0.0.0 --port=5000

# Or use the launcher script:
bash run_interface.sh

# Access at: http://localhost:5000
```

### 4. Make Predictions via API

```bash
# Using curl
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'

# Response:
# {
#   "success": true,
#   "verdict": "PENIS",
#   "confidence": 0.9234,
#   "raw_probability": 0.9234
# }
```

## Model Architecture

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Layer (type)               ┃ Output Shape        ┃    Param #   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Conv2D + BatchNorm (×3)    │ (None, 28, 28, 128) │    93,696    │
│ MaxPooling2D + Dropout     │                     │              │
├────────────────────────────┼─────────────────────┼──────────────┤
│ Flatten                    │ (None, 100352)      │         0    │
├────────────────────────────┼─────────────────────┼──────────────┤
│ Dense + BatchNorm          │ (None, 256)         │ 25,690,368   │
│ Dropout                    │                     │              │
├────────────────────────────┼─────────────────────┼──────────────┤
│ Dense + BatchNorm          │ (None, 128)         │    32,896    │
│ Dropout                    │                     │              │
├────────────────────────────┼─────────────────────┼──────────────┤
│ Dense (sigmoid)            │ (None, 1)           │       129    │
└────────────────────────────┴─────────────────────┴──────────────┘
Total params: 25,818,499 (98.49 MB)
Trainable params: 25,817,281
```

**Key Features**:
- Input: 128×128 grayscale images
- Binary classification with sigmoid output
- Per-image normalization prevents brightness shortcuts
- Optimized with Adam, binary crossentropy loss

## Dataset

### Positive Class: Penis Drawings
- **Source**: Custom NDJSON dataset (quickdraw_appendix/)
- **Samples**: 25,209 drawings
- **Format**: Vector strokes rendered to 128×128 bitmaps
- **Stroke Width**: 12px on 256×256 canvas → 6px at 128×128

### Negative Class: QuickDraw Common Shapes
- **Source**: Google QuickDraw dataset NDJSON files
- **Categories** (21 classes, 1,200 samples each):
  - airplane, apple, arm, banana, bird, boomerang
  - cat, circle, cloud, dog, drill, fish, flower
  - house, moon, pencil, square, star, sun, tree, triangle
- **Total**: 25,200 drawings
- **Rendering**: Same pipeline as positive class (consistent 6px strokes)

### Preprocessing Pipeline
1. Render vector strokes at 256×256 with 12px width
2. Downsample to 128×128 using LANCZOS filter (preserves sharpness)
3. Normalize to [0, 1] range
4. **Per-image normalization**: `(x - mean) / std` then rescale to [0, 1]
5. Result: Grey background (~0.45), white strokes (~0.9-1.0)

### Data Split
- **Training**: 40,320 samples (50% positive, 50% negative)
- **Testing**: 10,080 samples (stratified split)
- **Augmentation**: None (model generalizes well without it)

## Performance Metrics

**Test Set Results**:
- **Accuracy**: 97.25% (9,802 / 10,080 correct)
- **Penis Detection**: ~95% probability on true positives
- **Shape Detection**: ~3-7% probability on true negatives
- **Inference Time**: ~70ms per image (CPU), ~4ms per image (batched)

**Model Characteristics**:
- Robust to drawing style variations
- Handles different stroke thicknesses well (5-8px effective width)
- Minimal false positives on common shapes
- No data augmentation needed - generalizes well

## Web Interface

The included Flask web app provides an interactive drawing canvas:

**Features**:
- 512×512 HTML5 canvas for smooth drawing
- Adjustable brush size (5-50px, default 24px)
- Real-time predictions with confidence scores
- Automatic preprocessing matching training pipeline
- ~100ms total response time (drawing → result)

**Technical Details**:
- Canvas: White background, black strokes
- Preprocessing: Inverts colors, resizes to 128×128, applies per-image normalization
- API: RESTful JSON endpoints for predictions and health checks

## Documentation

See [`docs/ML_API_DOCUMENTATION.md`](docs/ML_API_DOCUMENTATION.md) for comprehensive API documentation including:
- Detailed preprocessing pipeline
- Drawing guidelines for optimal results  
- Common issues and solutions
- Python and JavaScript usage examples
- Batch processing and real-time inference

## Troubleshooting

**Model predicts everything as "OTHER_SHAPE"**
- ✓ Increase brush size (use 24px+ on 512×512 canvas)
- ✓ Verify strokes are thick enough (>10% bright pixels after preprocessing)

**Low confidence on clear drawings**
- ✓ Ensure per-image normalization is applied
- ✓ Check that preprocessed mean ≈ 0.5

**Poor performance on new drawings**
- ✓ Match training data stroke width (~6px at 128×128)
- ✓ Use continuous strokes, avoid scattered dots

## Future Improvements

- [ ] Multi-class classification to identify specific shapes
- [ ] GPU optimization for faster inference
- [ ] Model quantization for mobile deployment
- [ ] Additional training data from user submissions
- [ ] A/B testing for threshold optimization

## License

MIT

## References

- QuickDraw Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
