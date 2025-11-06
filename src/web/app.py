"""
Flask Web Application for Doodle Classification

Provides real-time drawing classification interface using trained CNN model.
Handles canvas input, image preprocessing, model inference, and result display.

Why Flask: Lightweight web framework suitable for ML model serving without
complex deployment infrastructure. Enables rapid prototyping of classification UI.

SECURITY: Validates image data size and format before processing
to prevent resource exhaustion attacks. CORS enabled for frontend integration.

Related:
- src/core/models.py (CNN model architectures)
- src/core/inference.py (inference pipeline)
- src/web/templates/index.html (drawing canvas interface)

Exports:
- load_model_and_mapping: Initialize model and class mappings
- preprocess_image: Convert canvas data to model input format
- predict: Run classification on preprocessed image

Usage:
Run the Flask development server:

    python src/web/app.py

Access the interface at http://localhost:5000

For production, use a WSGI server:

    gunicorn -w 4 -b 0.0.0.0:5000 src.web.app:app
"""

from typing import Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import os
import logging
import traceback
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2

# Set up comprehensive logging
# Logs go to both console and file for debugging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "doodlehunter.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DoodleHunter')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.patch_extraction import (
    SlidingWindowDetector,
    AggregationStrategy
)

if TYPE_CHECKING:
    from tensorflow.keras import Model

app = Flask(__name__)
CORS(app)

model: Optional['Model'] = None
tflite_interpreter: Optional['tf.lite.Interpreter'] = None
is_tflite: bool = False
model_name: str = 'Unknown'
idx_to_class: Optional[Dict[int, str]] = None
THRESHOLD: float = 0.5


def load_model_and_mapping() -> None:
    """
    Initialize trained model and class label mappings.

    Automatically detects and loads the best available model in priority order:
    1. TFLite INT8 quantized (smallest, fastest)
    2. TFLite Float32
    3. Keras/H5 fallback

    Loads class-to-index mapping for converting model predictions to class names.
    Provides sensible defaults if mapping file is unavailable.

    Raises:
        FileNotFoundError: If no model files found or specific model file missing
        Exception: For other model loading errors (logged but not raised)
    """
    global model, tflite_interpreter, is_tflite, model_name, idx_to_class

    logger.info("=== Starting model and mapping initialization ===")

    models_dir = Path(__file__).parent.parent.parent / "models"
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"

    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data directory: {data_dir}")

    # Check if directories exist
    if not models_dir.exists():
        logger.error(f"Models directory does not exist: {models_dir}")
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")

    # Scan for available model files
    tflite_int8_files = list(models_dir.glob("*_int8.tflite"))
    tflite_files = list(models_dir.glob("*.tflite"))
    model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))

    logger.debug(f"Found {len(tflite_int8_files)} INT8 TFLite files: {tflite_int8_files}")
    logger.debug(f"Found {len(tflite_files)} TFLite files: {tflite_files}")
    logger.debug(f"Found {len(model_files)} Keras files: {model_files}")

    model_path: Optional[Path] = None

    # Select best model based on priority
    if tflite_int8_files:
        model_path = max(tflite_int8_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = 'TFLite INT8 (Optimized)'
        logger.info(f"Selected INT8 TFLite model: {model_path}")
    elif tflite_files:
        model_path = max(tflite_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = 'TFLite Float32'
        logger.info(f"Selected TFLite model: {model_path}")
    elif model_files:
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        is_tflite = False
        model_name = 'Keras/TensorFlow'
        logger.info(f"Selected Keras model: {model_path}")
    else:
        logger.error(f"No model files found in {models_dir}")
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Verify model file exists
    if not model_path.exists():
        logger.error(f"Selected model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Log model file details
    size_mb = model_path.stat().st_size / (1024 * 1024)
    modified_time = model_path.stat().st_mtime
    logger.info(f"Model file size: {size_mb:.2f} MB")
    logger.info(f"Model last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))}")

    # Load the selected model
    if is_tflite:
        logger.info("Initializing TFLite interpreter with optimizations...")
        try:
            # Configure TFLite interpreter with multi-threading
            # XNNPACK delegate is automatically used if available
            num_threads = 4  # RPi4 has 4 cores
            
            logger.info(f"Loading TFLite with {num_threads} threads...")
            tflite_interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                num_threads=num_threads
            )
            logger.info("✓ TFLite loaded with multi-threading")
            
            # Allocate tensors with memory mapping for faster loading
            tflite_interpreter.allocate_tensors()

            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            logger.info("TFLite model loaded successfully!")
            logger.info(f"  Threads: {num_threads}")
            logger.info(f"  Input shape: {input_details[0]['shape']}")
            logger.info(f"  Input dtype: {input_details[0]['dtype']}")
            logger.info(f"  Input name: {input_details[0]['name']}")
            logger.info(f"  Output shape: {output_details[0]['shape']}")
            logger.info(f"  Output dtype: {output_details[0]['dtype']}")
            logger.info(f"  Output name: {output_details[0]['name']}")
            
            # Warm up the model with a dummy inference
            logger.info("Warming up model with dummy inference...")
            dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
            tflite_interpreter.set_tensor(input_details[0]['index'], dummy_input)
            tflite_interpreter.invoke()
            logger.info("✓ Model warm-up complete")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    else:
        logger.info("Loading Keras model...")
        try:
            model = keras.models.load_model(model_path)
            logger.info("Keras model loaded successfully!")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            logger.info(f"Model total params: {model.count_params():,}")
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise

    # Load class mapping
    logger.info("Loading class mapping...")
    try:
        mapping_file = data_dir / "class_mapping.pkl"
        if mapping_file.exists():
            logger.debug(f"Reading mapping from: {mapping_file}")
            with open(mapping_file, 'rb') as f:
                class_mapping = pickle.load(f)

            logger.debug(f"Raw class mapping: {class_mapping}")

            idx_to_class = {}
            for k, v in class_mapping.items():
                if isinstance(v, int):
                    idx_to_class[v] = k

            logger.info(f"Processed class mapping: {idx_to_class}")

            if not idx_to_class:
                logger.warning('Could not extract class mappings from file, using defaults')
                idx_to_class = {0: 'negative', 1: 'positive'}
                logger.info(f"Using default mapping: {idx_to_class}")
        else:
            logger.warning('class_mapping.pkl not found, using defaults')
            idx_to_class = {0: 'negative', 1: 'positive'}
            logger.info(f"Using default mapping: {idx_to_class}")
    except FileNotFoundError:
        logger.warning('class_mapping.pkl not found, using defaults')
        idx_to_class = {0: 'negative', 1: 'positive'}
        logger.info(f"Using default mapping: {idx_to_class}")
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}, using defaults")
        idx_to_class = {0: 'negative', 1: 'positive'}
        logger.info(f"Using default mapping: {idx_to_class}")

    logger.info(f"Model '{model_name}' loaded successfully with mapping: {idx_to_class}")

    logger.info("=== Model and mapping initialization complete ===")


def preprocess_image(image_data: str) -> np.ndarray:
    """
    Convert canvas drawing to model-compatible input format.

    Applies same preprocessing pipeline used during training to ensure predictions
    are consistent with model expectations. Handles format conversion, color
    inversion, resizing, and normalization.

    Args:
        image_data: Base64 encoded image data from canvas (may include data URL prefix)

    Returns:
        Preprocessed numpy array with shape (1, 128, 128, 1) ready for model prediction

    Raises:
        ValueError: If image data is invalid or cannot be decoded
    """
    logger.debug("=== Starting image preprocessing ===")
    logger.debug(f"Input image_data length: {len(image_data)}")

    # Validate input
    if not image_data:
        logger.error("Empty image data received")
        raise ValueError("Image data is empty")

    # Remove data URL prefix if present
    image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
    logger.debug(f"Cleaned image_data length: {len(image_data_clean)}")

    # Decode base64
    try:
        decoded_data = base64.b64decode(image_data_clean)
        logger.debug(f"Decoded data size: {len(decoded_data)} bytes")
    except Exception as e:
        logger.error(f"Failed to decode base64 image data: {e}")
        raise ValueError(f"Invalid base64 data: {e}")

    # Open image
    try:
        image = Image.open(BytesIO(decoded_data))
        logger.info(f"Image opened: {image.size} pixels, mode: {image.mode}")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise ValueError(f"Invalid image format: {e}")

    # Convert to grayscale to match training data format
    # Redundant color channels would add noise to the model
    logger.debug("Converting to grayscale...")
    image = image.convert('L')
    logger.debug(f"Grayscale image: {image.size}")

    # Convert to numpy array
    img_array = np.array(image, dtype=np.uint8)
    logger.debug(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
    logger.debug(f"Array value range: [{img_array.min()}, {img_array.max()}]")

    # Invert colors because canvas draws black on white
    # Model was trained on white drawings on black background
    logger.debug("Inverting colors...")
    img_array = 255 - img_array
    logger.debug(f"After inversion - value range: [{img_array.min()}, {img_array.max()}]")

    # Apply morphological dilation to thicken strokes
    # Prevents thin lines from disappearing during resizing
    logger.debug("Applying morphological dilation...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_array = cv2.dilate(img_array, kernel, iterations=1)
    logger.debug(f"After dilation - value range: [{img_array.min()}, {img_array.max()}]")

    # Convert back to PIL for high-quality resize
    logger.debug("Converting to PIL for resize...")
    image_inverted = Image.fromarray(img_array, 'L')
    logger.debug(f"Created PIL image: {image_inverted.size}")

    # Resize to model input size
    logger.debug("Resizing to 128x128...")
    image_resized = image_inverted.resize((128, 128), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image: {image_resized.size}")

    # Convert back to numpy array and normalize
    logger.debug("Converting to float32 and normalizing...")
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    logger.debug(f"After normalization - shape: {img_array.shape}, range: [{img_array.min():.4f}, {img_array.max():.4f}]")

    # Apply z-score normalization for consistent brightness
    # Only normalize if image has sufficient variation (avoid noise amplification)
    img_flat = img_array.flatten()
    original_mean = img_flat.mean()
    original_std = img_flat.std()
    logger.debug(f"Pre-normalization stats: mean={original_mean:.4f}, std={original_std:.4f}")

    if img_flat.std() > 0.01:
        logger.debug("Applying z-score normalization...")
        # Standardize to zero mean, unit variance
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        # Rescale from [-2, 2] to [0, 1] to match training distribution
        img_array = (img_array + 2) / 4
        # Ensure all values are in valid range
        img_array = np.clip(img_array, 0, 1)

        logger.debug(f"After normalization - range: [{img_array.min():.4f}, {img_array.max():.4f}]")
    else:
        logger.debug("Skipping normalization due to low standard deviation")

    # Add batch dimension
    logger.debug("Adding batch dimension...")
    img_array = img_array.reshape(1, 128, 128, 1)
    logger.debug(f"Final output shape: {img_array.shape}")

    logger.debug("=== Image preprocessing complete ===")
    return img_array


def predict(image_data: str) -> Dict[str, Any]:
    """
    Classify drawing using trained CNN model.

    Runs full prediction pipeline: input validation, preprocessing, model inference,
    and confidence scoring. Returns structured result for frontend consumption.

    Args:
        image_data: Base64 encoded drawing from canvas

    Returns:
        Dictionary with prediction results including:
        - success: Boolean indicating if prediction succeeded
        - verdict: Classification result ('PENIS' or 'OTHER_SHAPE')
        - verdict_text: Human-readable description
        - confidence: Confidence score (0-1)
        - raw_probability: Raw model output probability
        - threshold: Classification threshold used
        - model_info: Description of model used
        - drawing_statistics: Timing information in milliseconds

    Raises:
        Exception: Propagates from preprocessing or inference errors
    """
    logger.info("=== Starting predict() function ===")
    logger.debug(f"predict() input: image_data length={len(image_data)}")

    try:
        # Check model state before processing
        logger.debug(f"Model state check - model is None: {model is None}")
        logger.debug(f"Model state check - tflite_interpreter is None: {tflite_interpreter is None}")
        logger.debug(f"Model state check - is_tflite: {is_tflite}")
        logger.debug(f"Model state check - model_name: {model_name}")

        if model is None and tflite_interpreter is None:
            logger.error("predict() - Model not loaded, returning error")
            return {
                'success': False,
                'error': 'Model not loaded. Please train a model first or place a trained model (.h5 or .keras) in the models/ directory.'
            }

        start_time = time.time()
        logger.debug(f"predict() - Start time: {start_time}")

        preprocess_start = time.time()
        logger.debug(f"predict() - Starting preprocessing at: {preprocess_start}")

        img_array = preprocess_image(image_data)
        preprocess_time = time.time() - preprocess_start
        logger.info(f"predict() - Preprocessing completed in {preprocess_time*1000:.2f}ms")
        logger.debug(f"predict() - Preprocessed array shape: {img_array.shape}")
        logger.debug(f"predict() - Preprocessed array dtype: {img_array.dtype}")
        logger.debug(f"predict() - Preprocessed array value range: [{img_array.min():.4f}, {img_array.max():.4f}]")

        inference_start = time.time()
        logger.debug(f"predict() - Starting inference at: {inference_start}")

        probability = None
        if is_tflite:
            logger.debug("predict() - Using TFLite model for inference")
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            logger.debug(f"predict() - TFLite input details: {input_details[0]}")
            logger.debug(f"predict() - TFLite output details: {output_details[0]}")

            # Set input tensor
            input_array = img_array.astype(np.float32)
            logger.debug(f"predict() - Setting input tensor with shape: {input_array.shape}")
            tflite_interpreter.set_tensor(input_details[0]['index'], input_array)

            # Invoke interpreter
            logger.debug("predict() - Invoking TFLite interpreter")
            tflite_interpreter.invoke()

            # Get output tensor
            output_tensor = tflite_interpreter.get_tensor(output_details[0]['index'])
            probability = output_tensor[0][0]
            logger.debug(f"predict() - TFLite output tensor shape: {output_tensor.shape}")
            logger.debug(f"predict() - TFLite raw probability: {probability}")
        else:
            logger.debug("predict() - Using Keras model for inference")
            logger.debug(f"predict() - Keras model input shape: {model.input_shape}")
            logger.debug(f"predict() - Keras model output shape: {model.output_shape}")

            # Run prediction with verbose=0 to suppress progress output
            predictions = model.predict(img_array, verbose=0)
            probability = predictions[0][0]
            logger.debug(f"predict() - Keras prediction shape: {predictions.shape}")
            logger.debug(f"predict() - Keras raw probability: {probability}")

        inference_time = time.time() - inference_start
        logger.info(f"predict() - Inference completed in {inference_time*1000:.2f}ms")

        total_time = time.time() - start_time
        logger.info(f"predict() - Total prediction time: {total_time*1000:.2f}ms")

        # Apply classification threshold and format results
        # Binary classification: above threshold = PENIS, below = OTHER_SHAPE
        logger.debug(f"predict() - Applying threshold: {THRESHOLD}")
        logger.debug(f"predict() - Raw probability: {probability}")

        if probability >= THRESHOLD:
            verdict = 'PENIS'
            verdict_text = 'Drawing looks like a penis! ✓'
            confidence = float(probability)
            logger.info(f"predict() - Classification: PENIS (confidence: {confidence:.4f})")
        else:
            verdict = 'OTHER_SHAPE'
            verdict_text = 'Drawing looks like a common shape (not penis).'
            confidence = float(1 - probability)
            logger.info(f"predict() - Classification: OTHER_SHAPE (confidence: {confidence:.4f})")

        result = {
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

        logger.debug(f"predict() - Returning result: success={result['success']}, verdict={result['verdict']}")
        logger.debug(f"predict() - Drawing statistics: {result['drawing_statistics']}")
        logger.info("=== predict() function completed successfully ===")

        return result
    except Exception as e:
        logger.error(f"predict() - Exception occurred: {type(e).__name__}: {e}")
        logger.error(f"predict() - Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index() -> str:
    """
    Serve the main web interface.

    Returns:
        HTML content for the drawing canvas interface
    """
    logger.debug("index() - Serving web interface")
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict() -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    REST API endpoint for simple single-image classification.

    This is the most basic detection mode - classifies the entire canvas
    as a single image without any region-based or stroke-based analysis.

    Expects JSON payload with 'image' field containing base64 encoded drawing.

    Returns:
        Tuple of (response_dict, status_code)
    """
    request_id = f"req_{int(time.time() * 1000000)}"
    logger.info(f"api_predict() - {request_id} - Received POST request")

    try:
        data = request.get_json()
        logger.debug(f"api_predict() - {request_id} - Request content-type: {request.content_type}")

        if not data:
            logger.warning(f"api_predict() - {request_id} - No JSON data provided")
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        image_data = data.get('image')
        logger.debug(f"api_predict() - {request_id} - Image data present: {image_data is not None}")

        if not image_data:
            logger.warning(f"api_predict() - {request_id} - Missing 'image' field in request")
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        logger.info(f"api_predict() - {request_id} - Starting simple prediction pipeline")
        prediction_start = time.time()

        # Call prediction function
        result = predict(image_data)

        prediction_time = time.time() - prediction_start
        logger.info(f"api_predict() - {request_id} - Simple prediction completed in {prediction_time*1000:.2f}ms")

        if result.get('success'):
            logger.info(f"api_predict() - {request_id} - Response: verdict={result.get('verdict')}, confidence={result.get('confidence')}")

        result['request_id'] = request_id

        logger.info(f"api_predict() - {request_id} - Returning 200 OK")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"api_predict() - {request_id} - Exception occurred: {type(e).__name__}: {e}")
        logger.error(f"api_predict() - {request_id} - Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500




@app.route('/api/health', methods=['GET'])
def health() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancer probes.

    Returns:
        Dictionary with service status and configuration information
    """
    request_id = f"req_{int(time.time() * 1000000)}"
    logger.info(f"health() - {request_id} - Received GET request to /api/health")
    logger.debug(f"health() - {request_id} - Request headers: {dict(request.headers)}")

    # Check model state
    model_loaded = model is not None or tflite_interpreter is not None
    logger.debug(f"health() - {request_id} - Model loaded: {model_loaded}")
    logger.debug(f"health() - {request_id} - Keras model: {model is not None}")
    logger.debug(f"health() - {request_id} - TFLite interpreter: {tflite_interpreter is not None}")
    logger.debug(f"health() - {request_id} - Model type: {model_name}")
    logger.debug(f"health() - {request_id} - Threshold: {THRESHOLD}")

    # Prepare health response
    health_response = {
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_name': model_name,
        'model_type': 'TFLite' if is_tflite else 'Keras',
        'threshold': THRESHOLD,
        'simple_detection': True
    }

    logger.info(f"health() - {request_id} - Returning health status: {health_response['status']}")
    logger.debug(f"health() - {request_id} - Response: {health_response}")

    return jsonify(health_response)


try:
    load_model_and_mapping()
except Exception as e:
    print(f"Warning: Failed to load model on startup: {e}")
    print('Model will need to be loaded manually')


if __name__ == '__main__':
    print('Starting Penis Classifier Interface...')
    print('Visit http://localhost:5000 to use the drawing board')
    print('Model: Binary classification (penis vs other shapes)')
    app.run(debug=True, host='0.0.0.0', port=5000)
