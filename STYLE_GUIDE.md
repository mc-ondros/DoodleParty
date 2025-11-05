# DoodleHunter - Code Style Guide

This guide defines the consistent coding and documentation standards for all DoodleHunter modules. It covers ML training scripts (Python/TensorFlow), web interface (Python/Flask), and data processing components.

## Prohibited Characters in Comments

**Never use these in comments:**
- `===` (triple equals)
- `---` (triple dashes)

Use these alternatives instead:
- For separation: blank comment lines
- For emphasis: **bold** or _italic_ markdown

### Python Examples

```python
# Good - uses blank line for separation
# Validate image array shape

if not image_array.shape == (128, 128, 1):
    return False

# Bad - uses prohibited characters
# Validate image array shape

if not image_array.shape == (128, 128, 1):
    return False
```

## File Header Comments

**Format:**

```python
"""
QuickDraw Binary Classifier Training Script

TensorFlow/Keras model training for binary classification of drawings.
Handles data loading, augmentation, model training, and evaluation.

Related:
- src/dataset.py (data loading and preprocessing)
- src/models.py (model architectures)
- src/data_pipeline.py (data augmentation pipeline)
- scripts/train.sh (training orchestration)

Exports:
- train_model, evaluate_model, save_model
"""
```

**Rules:**
1. First line: Brief description of the module
2. Blank comment line after first line
3. Bullet points use `-` character (standard markdown)
4. Each section separated by blank comment line
5. Include Usage section where relevant

## Code Comments

### Standalone Comments (above code)

```python
# Normalize to [0, 1] range for model input
image_array = np.array(image, dtype=np.float32) / 255.0
```

**Rules:**
1. Single line whenever possible
2. Multi-line if necessary, each line starts with `#`
3. No trailing comments after closing braces
4. Never use `===` or `---` in comments

## Comment Content Guidelines

### Do:
- ✓ Explain **why**, not **what**
- ✓ Document edge cases and gotchas
- ✓ Reference external resources
- ✓ Note data format expectations

### Don't:
- ✗ Use `===` or `---` in comments
- ✗ State the obvious (`i = 0  # Set i to 0`)
- ✗ Write overly verbose comments

## Special Comment Types

### TODOs

```python
# TODO: Add data augmentation support
# TODO(DoodleHunter-team): Implement multi-class classification
```

### Notes

```python
# NOTE: This requires TensorFlow 2.13+
```

### Security

```python
# SECURITY: Validate all user input before processing
# Prevent path traversal in file uploads
```

## Multi-Line Comments

```python
# First line of explanation
# Second line of explanation
# Third line of explanation
code = value
```

**Rules:**
1. Each line starts with `#` and one space
2. No blank lines within multi-line comment
3. Blank line before code block
4. Never use `===` or `---`

## Examples

### Good

```python
# Cache strategy: LRU with 1000 entry limit
# Eviction happens at 90% capacity
make_cache = size: ...
```

### Bad

```python
make_cache = size: ...  # makes a cache of size
```

## Naming Conventions

### Files

- Python: `snake_case.py`
- Shell scripts: `kebab-case.sh`

**Examples:**

- `data_pipeline.py`
- `appendix_loader.py`
- `train-model.sh`

### Code Elements

**Classes:**

- PascalCase

```python
class DataPipeline:
    pass

class ModelTrainer:
    pass
```

**Functions:**

- snake_case

```python
def preprocess_image(image):
    pass

def load_quickdraw_data():
    pass
```

**Constants:**

- UPPER_SNAKE_CASE

```python
IMAGE_SIZE = 128
MAX_SAMPLES_PER_CLASS = 10000
DEFAULT_BATCH_SIZE = 32
```

## Code Formatting

### Indentation

- Python: **4 spaces**
- Shell scripts: **2 spaces**

### Line Length

- Maximum: **100 characters**

### Quotes

- Single quotes (`'`) for all Python code
- Double quotes only for JSON and docstrings

```python
message = 'Hello, DoodleHunter!'
config = {'batch_size': 32}  # JSON format
```

### Trailing Commas

- Always use in multi-line arrays/dicts

```python
config = {
    'image_size': 128,
    'batch_size': 32,
    'epochs': 50,
}
```

## Import Ordering

### Python

1. Standard library
2. Third-party packages (TensorFlow, NumPy, etc.)
3. Local imports

```python
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .dataset import load_data
from .models import create_model
```

## Language-Specific Guidelines

### Python

**Type Hints:**

```python
from typing import Tuple, Optional
import numpy as np

def preprocess_image(
    image: np.ndarray,
    target_size: int = 128
) -> Tuple[np.ndarray, bool]:
    """
    Preprocesses image for model input.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size for resizing

    Returns:
        Tuple of (preprocessed_image, is_valid)
    """
    if image.shape[0] < target_size or image.shape[1] < target_size:
        return None, False
    
    processed = resize_image(image, target_size)
    return processed, True
```

**Docstrings:**

- Google-style docstrings
- Include Args, Returns, Raises sections

```python
def train_model(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 50
) -> keras.Model:
    """
    Trains binary classification model on QuickDraw data.

    Args:
        train_data: Training images (N, H, W, C)
        train_labels: Binary labels (N,)
        epochs: Number of training epochs

    Returns:
        Trained Keras model

    Raises:
        ValueError: If data shapes are incompatible
    """
    if train_data.shape[0] != train_labels.shape[0]:
        raise ValueError('Data and labels must have same length')
    
    model = create_model()
    model.fit(train_data, train_labels, epochs=epochs)
    return model
```

### Flask

**Route Structure:**

```python
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for drawing classification.
    
    Expects JSON with base64 encoded image.
    Returns prediction with confidence score.
    """
    try:
        data = request.get_json()
        image = preprocess_image(data['image'])
        prediction = model.predict(image)
        
        return jsonify({
            'class': idx_to_class[prediction],
            'confidence': float(prediction[0]),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

## TensorFlow/Keras Conventions

### Model Definition

```python
def create_model(input_shape: Tuple[int, int, int] = (128, 128, 1)) -> keras.Model:
    """
    Creates CNN model for binary classification.
    
    Args:
        input_shape: Input image shape (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    
    return model
```

### Data Pipeline

```python
# Good - use tf.data API for efficient data loading
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Bad - loading all data into memory
all_images = np.concatenate([load_batch(i) for i in range(100)])
```

## Security Guidelines

### Input Validation

```python
def validate_image_data(image_data: str) -> bool:
    """Validate base64 image data from user input."""
    # Check required fields
    if not image_data or not isinstance(image_data, str):
        return False
    
    # Prevent excessively large payloads
    if len(image_data) > MAX_IMAGE_SIZE:
        return False
    
    # Validate base64 format
    try:
        base64.b64decode(image_data.split(',')[1])
    except Exception:
        return False
    
    return True
```

### File Operations

```python
# Good - use Path for safe file operations
from pathlib import Path

def load_model_safe(model_name: str) -> keras.Model:
    """Load model with path validation."""
    models_dir = Path(__file__).parent / 'models'
    model_path = models_dir / f'{model_name}.h5'
    
    # Prevent path traversal
    if not model_path.resolve().is_relative_to(models_dir.resolve()):
        raise ValueError('Invalid model path')
    
    return keras.models.load_model(model_path)

# Bad - vulnerable to path traversal
def load_model_unsafe(model_name: str):
    return keras.models.load_model(f'models/{model_name}.h5')
```

## Documentation

### Project Documentation

Keep documentation in existing files:

- `README.md` - Project overview
- `STYLE_GUIDE.md` - This file
- Individual script docstrings - Implementation details

**Rule:** No new `.md` files. Extend existing files instead.

## Code Review Checklist

### Documentation & Comments
- [ ] File headers follow standard format
- [ ] Comments explain **why** not **what**
- [ ] Standalone comments preferred over inline
- [ ] No obvious/redundant comments
- [ ] Data format expectations documented
- [ ] TODO/FIXME comments use proper format

### Formatting
- [ ] 4-space indent for Python
- [ ] Line length ≤100 characters
- [ ] No `---` in comments/markdown
- [ ] Trailing commas in multi-line structures

### Naming
- [ ] Files: snake_case.py
- [ ] Classes: PascalCase
- [ ] Functions: snake_case
- [ ] Constants: UPPER_SNAKE_CASE

### Language-Specific
- [ ] Python: Type hints and docstrings
- [ ] TensorFlow: Efficient data pipelines
- [ ] Flask: Proper error handling

### Security
- [ ] Input validation present
- [ ] No hardcoded secrets
- [ ] Safe file operations (use Path)
- [ ] Error handling prevents info leaks

### ML Best Practices
- [ ] Data normalization documented
- [ ] Model input/output shapes specified
- [ ] Random seeds set for reproducibility
- [ ] Validation data properly separated
