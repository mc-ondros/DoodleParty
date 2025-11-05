# DoodleHunter ML API Documentation

## Overview

The DoodleHunter ML model is a binary classifier trained to distinguish between penis drawings and common shapes from the QuickDraw dataset. It uses a Convolutional Neural Network (CNN) with 25.8M parameters trained on 128×128 high-resolution images.

## Model Specifications

- **Architecture**: CNN with 3 convolutional blocks + 2 dense layers
- **Input Shape**: `(1, 128, 128, 1)` - grayscale images
- **Output**: Single sigmoid probability `[0.0, 1.0]`
  - `≥ 0.5`: Classified as **penis** (positive class)
  - `< 0.5`: Classified as **common shape** (negative class)
- **Performance**: 97.25% accuracy on test set
- **Response Time**: ~70ms per prediction (CPU)

## Training Data Format

### Image Specifications

All training images are preprocessed to match this format:

```
Size: 128×128 pixels
Channels: 1 (grayscale)
Background: Black (value 0 before normalization)
Strokes: White (value 255 before normalization)
Stroke Width: ~6px effective width at 128×128 resolution
Format: Float32 array, normalized
```

### Preprocessing Pipeline

1. **Raw Drawing** (vector strokes)
   - Rendered at 256×256 with 12px stroke width
   - Black background, white strokes

2. **Downsampling**
   - Resize 256×256 → 128×128 using LANCZOS filter
   - Preserves stroke quality

3. **Normalization to [0, 1]**
   ```python
   img_array = img_array.astype(np.float32) / 255.0
   ```

4. **Per-Image Normalization** (Critical!)
   ```python
   # Center around mean and scale by std
   img_flat = img_array.flatten()
   if img_flat.std() > 0.01:
       img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
       img_array = (img_array + 3) / 6  # Rescale to ~[0, 1]
       img_array = np.clip(img_array, 0, 1)
   ```
   
   **Why?** This prevents the model from using brightness as a classification shortcut. After normalization:
   - Background pixels: ~0.43-0.48 (grey, not black)
   - Stroke pixels: ~0.9-1.0 (bright)
   - Mean: ~0.5 for all images

5. **Final Shape**
   ```python
   img_array = img_array.reshape(1, 128, 128, 1)
   ```

## Using the Model

### Web API Endpoint

**Endpoint**: `POST /api/predict`

**Request Format**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**Response Format**:
```json
{
  "success": true,
  "verdict": "PENIS" | "OTHER_SHAPE",
  "verdict_text": "Human-readable description",
  "confidence": 0.9234,
  "raw_probability": 0.9234,
  "threshold": 0.5,
  "model_info": "Binary classifier: penis vs 21 common shapes"
}
```

### Python Direct Usage

```python
import numpy as np
from tensorflow import keras
from PIL import Image
from io import BytesIO
import base64

# Load model
model = keras.models.load_model('models/quickdraw_model.h5')

def preprocess_image(image_data):
    """
    Preprocess canvas image for model prediction.
    
    Args:
        image_data: Base64 encoded PNG image (e.g., from HTML canvas)
    
    Returns:
        Preprocessed numpy array (1, 128, 128, 1)
    """
    # Decode base64
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Load image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert('L')  # Convert to grayscale
    
    # Invert colors (canvas has white bg/black strokes)
    img_array = np.array(image, dtype=np.uint8)
    img_array = 255 - img_array
    
    # Resize to 128x128
    image_inverted = Image.fromarray(img_array, mode='L')
    image_resized = image_inverted.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Normalize to [0, 1]
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Per-image normalization (CRITICAL - must match training!)
    img_flat = img_array.flatten()
    if img_flat.std() > 0.01:
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        img_array = (img_array + 3) / 6
        img_array = np.clip(img_array, 0, 1)
    
    # Reshape for model input
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# Make prediction
img_array = preprocess_image(base64_image_data)
probability = model.predict(img_array, verbose=0)[0][0]

if probability >= 0.5:
    print(f"PENIS - Confidence: {probability:.2%}")
else:
    print(f"OTHER SHAPE - Confidence: {(1-probability):.2%}")
```

## Drawing Guidelines for Best Results

### Canvas Setup

For optimal results, use these settings when collecting drawings:

```javascript
// Recommended canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Canvas size: 512×512 for smooth drawing
canvas.width = 512;
canvas.height = 512;

// White background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 512, 512);

// Stroke settings
ctx.strokeStyle = '#000';  // Black strokes
ctx.lineWidth = 24;         // 24px at 512×512 = 6px at 128×128
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
```

### Critical Requirements

1. **Stroke Width**: Must be thick enough!
   - At 512×512: Use 20-30px brush
   - At 256×256: Use 10-15px brush
   - At 128×128: Use 5-8px brush
   - **Too thin strokes will fail!** Model expects ~6px effective width

2. **Color Inversion**: Canvas and model use opposite conventions
   - **Canvas**: White background, black strokes
   - **Model**: Black background, white strokes (inverted during preprocessing)

3. **Image Quality**: Use PNG format with lossless compression

4. **Drawing Style**: Model trained on single-stroke continuous drawings
   - Clear, connected lines
   - Avoid scattered dots or multiple disconnected shapes

## Common Issues and Solutions

### Issue: Model classifies everything as "OTHER_SHAPE"

**Cause**: Strokes too thin

**Solution**: 
- Increase brush size (24px+ on 512×512 canvas)
- Verify preprocessed image has >10% bright pixels
- Check: `np.sum(img_array > 0.6) / img_array.size > 0.10`

### Issue: Model gives low confidence on clear drawings

**Cause**: Missing per-image normalization

**Solution**:
- Ensure preprocessing applies per-image normalization
- Check mean is ~0.5 after preprocessing
- Background should be grey (~0.45), not black (0)

### Issue: Model performs poorly on new drawings

**Cause**: Distribution shift from training data

**Solution**:
- Match training data style (continuous strokes)
- Use similar stroke widths
- Avoid over-detailed or sketchy drawings

## Model Files

- **Model**: `models/quickdraw_model.h5` (296MB)
- **Class Mapping**: `data/processed/class_mapping.pkl`
- **Training History**: `models/training_history.pkl`

## Training Dataset

- **Positive Class**: 25,209 penis drawings from custom NDJSON dataset
- **Negative Classes**: 25,200 drawings from QuickDraw (1,200 per class)
  - airplane, apple, arm, banana, bird, boomerang, cat, circle, cloud, dog
  - drill, fish, flower, house, moon, pencil, square, star, sun, tree, triangle

## Performance Metrics

- **Test Accuracy**: 97.25%
- **Training Samples**: 40,320 (50/50 positive/negative split)
- **Test Samples**: 10,080
- **Model Parameters**: 25,818,499 trainable parameters

## Version Info

- **Model Version**: 1.0 (trained Nov 5, 2025)
- **TensorFlow**: 2.x with Keras
- **Python**: 3.12+
- **Resolution**: 128×128 (high-res, rendered from vector strokes)

## Example Usage Scenarios

### Scenario 1: Web Drawing App

```javascript
// Get base64 from canvas
const imageData = canvas.toDataURL('image/png');

// Send to API
const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image: imageData})
});

const result = await response.json();
console.log(result.verdict, result.confidence);
```

### Scenario 2: Batch Processing

```python
import numpy as np
from tensorflow import keras

model = keras.models.load_model('models/quickdraw_model.h5')

# Load preprocessed images (already normalized)
images = np.load('images_to_classify.npy')  # Shape: (N, 128, 128, 1)

# Batch predict
probabilities = model.predict(images, batch_size=32)

# Classify
predictions = (probabilities >= 0.5).astype(int)
```

### Scenario 3: Real-time Webcam

```python
import cv2
import numpy as np

model = keras.models.load_model('models/quickdraw_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert to grayscale, resize, preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    
    # Apply preprocessing (invert, normalize, etc.)
    processed = preprocess_image_from_array(resized)
    
    # Predict
    prob = model.predict(processed, verbose=0)[0][0]
    
    # Display
    cv2.putText(frame, f"Prob: {prob:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('DoodleHunter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Support

For issues or questions:
- Check preprocessing matches training pipeline
- Verify stroke width is adequate (>10% bright pixels)
- Ensure per-image normalization is applied
- Test with known training samples first
