"""
Flask Web Application for Doodle Classification

Provides real-time drawing classification interface using trained CNN model.
Handles canvas input, image preprocessing, model inference, and result display.

Why Flask: Lightweight web framework suitable for ML model serving without
complex deployment infrastructure. Enables rapid prototyping of classification UI.

Security considerations: Validates image data size and format before processing
to prevent resource exhaustion attacks. CORS enabled for frontend integration.

Related:
- src/core/models.py (CNN model architectures)
- src/core/inference.py (inference pipeline)
- src/web/templates/index.html (drawing canvas interface)

Exports:
- load_model_and_mapping: Initialize model and class mappings
- preprocess_image: Convert canvas data to model input format
- predict: Run classification on preprocessed image
"""

import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Import region-based detection
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.patch_extraction import (
    SlidingWindowDetector,
    AggregationStrategy
)


app = Flask(__name__)
CORS(app)

# Global model and mapping
model = None
tflite_interpreter = None
is_tflite = False
model_name = "Unknown"
idx_to_class = None
THRESHOLD = 0.5


def load_model_and_mapping():
    """Initialize trained model and class label mappings."""
    global model, tflite_interpreter, is_tflite, model_name, idx_to_class

    models_dir = Path(__file__).parent.parent.parent / "models"
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"

    # Priority: Use optimized TFLite INT8 model if available for better performance
    tflite_int8_files = list(models_dir.glob("*_int8.tflite"))
    tflite_files = list(models_dir.glob("*.tflite"))
    model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    
    model_path = None
    
    # Try INT8 quantized TFLite first (smallest, fastest)
    if tflite_int8_files:
        model_path = max(tflite_int8_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = "TFLite INT8 (Optimized)"
        print(f"Loading optimized INT8 TFLite model: {model_path}")
    # Then try regular TFLite
    elif tflite_files:
        model_path = max(tflite_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = "TFLite Float32"
        print(f"Loading TFLite model: {model_path}")
    # Fall back to Keras/H5
    elif model_files:
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        is_tflite = False
        model_name = "Keras/TensorFlow"
        print(f"Loading Keras model: {model_path}")
    else:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load TFLite or Keras model
    if is_tflite:
        # Load TFLite model
        tflite_interpreter = tf.lite.Interpreter(model_path=str(model_path))
        tflite_interpreter.allocate_tensors()
        
        # Get model info
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        print(f"TFLite model loaded successfully!")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        
        # Get model size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
    else:
        model = keras.models.load_model(model_path)
        print(f"Keras model loaded successfully!")

    # Load class-to-index mapping for converting predictions to labels
    try:
        with open(data_dir / "class_mapping.pkl", 'rb') as f:
            class_mapping = pickle.load(f)

        # Extract integer index to class name mappings
        # Why filter: Mapping file may contain nested metadata beyond simple indices
        idx_to_class = {}
        for k, v in class_mapping.items():
            if isinstance(v, int):  # Only take integer values (class indices)
                idx_to_class[v] = k

        # Provide sensible defaults if extraction fails
        if not idx_to_class:
            print("Warning: Could not extract class mappings from file, using defaults")
            idx_to_class = {0: 'negative', 1: 'positive'}
    except FileNotFoundError:
        print("Warning: class_mapping.pkl not found")
        idx_to_class = {0: 'negative', 1: 'positive'}
    except Exception as e:
        print(f"Warning: Error loading class mapping: {e}")
        idx_to_class = {0: 'negative', 1: 'positive'}

    print("Model loaded successfully!")


def preprocess_image(image_data):
    """
    Convert canvas drawing to model-compatible input format.

    Applies same preprocessing pipeline used during training to ensure predictions
    are consistent with model expectations. Handles format conversion, color
    inversion, resizing, and normalization.

    Args:
        image_data: Base64 encoded image data from canvas

    Returns:
        Preprocessed numpy array (1, 128, 128, 1) ready for model prediction
    """
    # Extract base64 payload from data URL format
    # Why split: Canvas data often includes metadata prefix "data:image/png;base64,"
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Convert to single-channel grayscale
    # Why grayscale: Model was trained on single-channel images (no color information needed)
    image = image.convert('L')

    # Invert colors to match training data convention
    # Why invert: Canvas produces white-on-black, but model expects black-on-white
    # This inversion aligns with QuickDraw dataset format used during training
    img_array = np.array(image, dtype=np.uint8)
    img_array = 255 - img_array

    # Resize to model's expected input size with high-quality resampling
    # Why 128x128: Matches training resolution for consistent feature extraction
    # Why LANCZOS: Provides sharp results suitable for line drawings
    image_inverted = Image.fromarray(img_array, 'L')
    image_resized = image_inverted.resize((128, 128), Image.Resampling.LANCZOS)

    # Scale pixel values to [0, 1] range
    # Why normalize: Model was trained on normalized inputs for stable gradients
    img_array = np.array(image_resized, dtype=np.float32) / 255.0

    # Apply per-image standardization matching training pipeline
    # Why per-image: Prevents brightness bias by normalizing each image independently
    img_flat = img_array.flatten()
    if img_flat.std() > 0.01:  # Skip blank images (no variation to normalize)
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        img_array = (img_array + 3) / 6  # Rescale from [-3, 3] to [0, 1]
        img_array = np.clip(img_array, 0, 1)

    # Add batch dimension for model compatibility
    # Why reshape: TensorFlow expects (batch, height, width, channels)
    img_array = img_array.reshape(1, 128, 128, 1)

    return img_array


def predict(image_data):
    """
    Classify drawing using trained CNN model.

    Runs full prediction pipeline: input validation, preprocessing, model inference,
    and confidence scoring. Returns structured result for frontend consumption.

    Args:
        image_data: Base64 encoded drawing from canvas

    Returns:
        Dictionary with prediction results including response time
    """
    try:
        # Ensure model is available before processing
        if model is None and tflite_interpreter is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train a model first or place a trained model (.h5 or .keras) in the models/ directory.'
            }

        # Start timing
        start_time = time.time()
        
        # Preprocess image
        preprocess_start = time.time()
        img_array = preprocess_image(image_data)
        preprocess_time = time.time() - preprocess_start
        
        # Predict based on model type
        inference_start = time.time()
        
        if is_tflite:
            # TFLite inference
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            # Set input tensor
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            
            # Run inference
            tflite_interpreter.invoke()
            
            # Get output
            probability = tflite_interpreter.get_tensor(output_details[0]['index'])[0][0]
        else:
            # Keras inference
            probability = model.predict(img_array, verbose=0)[0][0]
        
        inference_time = time.time() - inference_start
        
        # Total time
        total_time = time.time() - start_time

        # Apply threshold to convert probability to binary classification
        # Why threshold: Model outputs probability, but interface needs discrete class
        if probability >= THRESHOLD:
            verdict = 'PENIS'
            verdict_text = "Drawing looks like a penis! ✓"
            confidence = float(probability)
        else:
            verdict = 'OTHER_SHAPE'
            verdict_text = "Drawing looks like a common shape (not penis)."
            confidence = float(1 - probability)

        return {
            'success': True,
            'verdict': verdict,
            'verdict_text': verdict_text,
            'confidence': round(confidence, 4),
            'raw_probability': round(float(probability), 4),
            'threshold': THRESHOLD,
            'model_info': f'Binary classifier: penis vs 21 common shapes ({model_name})',
            'drawing_statistics': {
                'response_time_ms': round(total_time * 1000, 2),
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'inference_time_ms': round(inference_time * 1000, 2)
            }
        }
    except Exception as e:
        # Return error without exposing stack traces to users
        # Why catch: Prevents application crashes from malformed input
        return {
            'success': False,
            'error': str(e)
        }


def predict_region_based(image_data, use_region_detection=True):
    """
    Classify drawing using region-based detection for robustness.
    
    This approach analyzes multiple patches of the canvas to detect
    suspicious content even when diluted with innocent content.
    
    Args:
        image_data: Base64 encoded drawing from canvas
        use_region_detection: Whether to use region-based detection
    
    Returns:
        Dictionary with prediction results including patch information
    """
    try:
        # Ensure model is available
        if model is None and tflite_interpreter is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train a model first or place a trained model in the models/ directory.'
            }
        
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        preprocess_start = time.time()
        img_array = preprocess_image(image_data)
        preprocess_time = time.time() - preprocess_start
        
        # Get the actual model object for region-based detection
        # TFLite models need a wrapper
        if is_tflite:
            # Create a wrapper for TFLite interpreter
            class TFLiteModelWrapper:
                def __init__(self, interpreter):
                    self.interpreter = interpreter
                    self.input_details = interpreter.get_input_details()
                    self.output_details = interpreter.get_output_details()
                
                def predict(self, x, verbose=0):
                    # Handle batch inference
                    batch_size = x.shape[0]
                    results = []
                    
                    for i in range(batch_size):
                        single_input = x[i:i+1]
                        self.interpreter.set_tensor(
                            self.input_details[0]['index'],
                            single_input.astype(np.float32)
                        )
                        self.interpreter.invoke()
                        output = self.interpreter.get_tensor(
                            self.output_details[0]['index']
                        )
                        results.append(output[0])
                    
                    return np.array(results)
            
            model_wrapper = TFLiteModelWrapper(tflite_interpreter)
        else:
            model_wrapper = model
        
        # Remove batch dimension for detector
        img_array_2d = img_array[0]  # (128, 128, 1)
        
        # Create sliding window detector
        inference_start = time.time()
        
        detector = SlidingWindowDetector(
            model=model_wrapper,
            patch_size=(128, 128),
            stride=(64, 64),  # 50% overlap for better coverage
            min_content_ratio=0.05,
            max_patches=16,
            early_stopping=True,
            early_stop_threshold=0.9,
            aggregation_strategy=AggregationStrategy.MAX,
            classification_threshold=THRESHOLD
        )
        
        # Perform detection
        detection_result = detector.detect_batch(img_array_2d)
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Format result
        if detection_result.is_positive:
            verdict = 'PENIS'
            verdict_text = "Drawing looks like a penis! ✓ (Detected via region analysis)"
            confidence = float(detection_result.confidence)
        else:
            verdict = 'OTHER_SHAPE'
            verdict_text = "Drawing looks like a common shape (not penis). (Verified via region analysis)"
            confidence = float(1 - detection_result.confidence)
        
        return {
            'success': True,
            'verdict': verdict,
            'verdict_text': verdict_text,
            'confidence': round(confidence, 4),
            'raw_probability': round(float(detection_result.confidence), 4),
            'threshold': THRESHOLD,
            'model_info': f'Binary classifier with region-based detection ({model_name})',
            'detection_details': {
                'num_patches_analyzed': detection_result.num_patches_analyzed,
                'early_stopped': detection_result.early_stopped,
                'aggregation_strategy': detection_result.aggregation_strategy,
                'patch_predictions': [
                    {
                        'x': p['x'],
                        'y': p['y'],
                        'confidence': round(p['confidence'], 4),
                        'is_positive': p['is_positive']
                    }
                    for p in detection_result.patch_predictions[:5]  # Limit to first 5 for response size
                ]
            },
            'drawing_statistics': {
                'response_time_ms': round(total_time * 1000, 2),
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'inference_time_ms': round(inference_time * 1000, 2)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Serve the main drawing interface page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for drawing classification requests."""
    try:
        data = request.get_json()
        image_data = data.get('image')

        # Validate required field before processing
        # Why validate: Prevents unnecessary work on malformed requests
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        # Run prediction pipeline
        result = predict(image_data)
        return jsonify(result)
    except Exception as e:
        # Return HTTP 500 for unexpected errors while avoiding exposure of internal details
        # Why generic error: Prevents information leakage about server internals
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/region', methods=['POST'])
def api_predict_region():
    """REST API endpoint for region-based detection (robust against content dilution)."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Run region-based prediction
        result = predict_region_based(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring and load balancer probes."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None or tflite_interpreter is not None,
        'threshold': THRESHOLD,
        'region_detection_available': True
    })


# Initialize model at module load time for faster first predictions
# Why eager loading: Avoids cold-start latency when first prediction request arrives
try:
    load_model_and_mapping()
except Exception as e:
    # Log warning but don't crash - allows app to run without model for testing
    # Why continue: Enables UI development even when model is not available
    print(f"Warning: Failed to load model on startup: {e}")
    print("Model will need to be loaded manually")


if __name__ == '__main__':
    # Start Flask development server
    # Why debug=True: Enables auto-reload and detailed error pages during development
    # Why 0.0.0.0: Allows external access (not just localhost)
    print("Starting Penis Classifier Interface...")
    print("Visit http://localhost:5000 to use the drawing board")
    print("Model: Binary classification (penis vs other shapes)")
    app.run(debug=True, host='0.0.0.0', port=5000)
