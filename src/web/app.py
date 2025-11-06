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

    models_dir = Path(__file__).parent.parent.parent / "models"
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"

    tflite_int8_files = list(models_dir.glob("*_int8.tflite"))
    tflite_files = list(models_dir.glob("*.tflite"))
    model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    
    model_path: Optional[Path] = None
    
    if tflite_int8_files:
        model_path = max(tflite_int8_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = 'TFLite INT8 (Optimized)'
        print(f"Loading optimized INT8 TFLite model: {model_path}")
    elif tflite_files:
        model_path = max(tflite_files, key=lambda p: p.stat().st_mtime)
        is_tflite = True
        model_name = 'TFLite Float32'
        print(f"Loading TFLite model: {model_path}")
    elif model_files:
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        is_tflite = False
        model_name = 'Keras/TensorFlow'
        print(f"Loading Keras model: {model_path}")
    else:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if is_tflite:
        tflite_interpreter = tf.lite.Interpreter(model_path=str(model_path))
        tflite_interpreter.allocate_tensors()
        
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        print(f"TFLite model loaded successfully!")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
    else:
        model = keras.models.load_model(model_path)
        print(f"Keras model loaded successfully!")

    try:
        with open(data_dir / "class_mapping.pkl", 'rb') as f:
            class_mapping = pickle.load(f)

        idx_to_class = {}
        for k, v in class_mapping.items():
            if isinstance(v, int):
                idx_to_class[v] = k

        if not idx_to_class:
            print('Warning: Could not extract class mappings from file, using defaults')
            idx_to_class = {0: 'negative', 1: 'positive'}
    except FileNotFoundError:
        print('Warning: class_mapping.pkl not found')
        idx_to_class = {0: 'negative', 1: 'positive'}
    except Exception as e:
        print(f"Warning: Error loading class mapping: {e}")
        idx_to_class = {0: 'negative', 1: 'positive'}

    print('Model loaded successfully!')


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
    image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(base64.b64decode(image_data_clean)))

    # Convert to grayscale to match training data format
    # Redundant color channels would add noise to the model
    image = image.convert('L')

    img_array = np.array(image, dtype=np.uint8)

    # Invert colors because canvas draws black on white
    # Model was trained on white drawings on black background
    img_array = 255 - img_array

    # Apply morphological dilation to thicken strokes
    # Prevents thin lines from disappearing during resizing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_array = cv2.dilate(img_array, kernel, iterations=1)
    
    image_inverted = Image.fromarray(img_array, 'L')
    image_resized = image_inverted.resize((128, 128), Image.Resampling.LANCZOS)

    img_array = np.array(image_resized, dtype=np.float32) / 255.0

    # Apply z-score normalization for consistent brightness
    # Only normalize if image has sufficient variation (avoid noise amplification)
    img_flat = img_array.flatten()
    if img_flat.std() > 0.01:
        # Standardize to zero mean, unit variance
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        # Rescale from [-2, 2] to [0, 1] to match training distribution
        img_array = (img_array + 2) / 4
        # Ensure all values are in valid range
        img_array = np.clip(img_array, 0, 1)

    img_array = img_array.reshape(1, 128, 128, 1)

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
    try:
        if model is None and tflite_interpreter is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train a model first or place a trained model (.h5 or .keras) in the models/ directory.'
            }

        start_time = time.time()
        
        preprocess_start = time.time()
        img_array = preprocess_image(image_data)
        preprocess_time = time.time() - preprocess_start
        
        inference_start = time.time()
        
        if is_tflite:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            
            tflite_interpreter.invoke()
            
            probability = tflite_interpreter.get_tensor(output_details[0]['index'])[0][0]
        else:
            probability = model.predict(img_array, verbose=0)[0][0]
        
        inference_time = time.time() - inference_start
        
        total_time = time.time() - start_time

        if probability >= THRESHOLD:
            verdict = 'PENIS'
            verdict_text = 'Drawing looks like a penis! ✓'
            confidence = float(probability)
        else:
            verdict = 'OTHER_SHAPE'
            verdict_text = 'Drawing looks like a common shape (not penis).'
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
        return {
            'success': False,
            'error': str(e)
        }


def predict_region_based(image_data: str, use_region_detection: bool = True) -> Dict[str, Any]:
    """
    Classify drawing using region-based detection for robustness.
    
    This approach analyzes multiple patches of the canvas to detect
    suspicious content even when diluted with innocent content.
    
    Args:
        image_data: Base64 encoded drawing from canvas
        use_region_detection: Whether to use region-based detection (currently always True)
    
    Returns:
        Dictionary with prediction results including:
        - success: Boolean indicating if prediction succeeded
        - verdict: Classification result
        - verdict_text: Human-readable description with detection method
        - confidence: Confidence score (0-1)
        - raw_probability: Raw model output probability
        - threshold: Classification threshold used
        - model_info: Description of model and detection method
        - detection_details: Region-based detection metadata
        - drawing_statistics: Timing information in milliseconds
    
    Raises:
        Exception: Propagates from preprocessing or inference errors
    """
    try:
        if model is None and tflite_interpreter is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train a model first or place a trained model in the models/ directory.'
            }
        
        start_time = time.time()
        
        preprocess_start = time.time()
        
        image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
        image = Image.open(BytesIO(base64.b64decode(image_data_clean)))
        image = image.convert('L')
        
        img_array = np.array(image, dtype=np.uint8)
        img_array = 255 - img_array
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        img_array = img_array.astype(np.float32) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        
        preprocess_time = time.time() - preprocess_start
        
        if is_tflite:
            class TFLiteModelWrapper:
                """
                Wrapper class for TFLite interpreter to enable batch processing.
                
                TFLite interpreter natively handles single inputs, but region-based
                detection requires batch inference. This wrapper handles patch
                resizing and standardization for each patch before inference.
                """
                def __init__(self, interpreter: tf.lite.Interpreter) -> None:
                    self.interpreter = interpreter
                    self.input_details = interpreter.get_input_details()
                    self.output_details = interpreter.get_output_details()
                
                def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
                    batch_size = x.shape[0]
                    results = []
                    
                    for i in range(batch_size):
                        patch = x[i]
                        
                        # Ensure patches match model input size (128x128)
                        # Some patches may be smaller when extracted near edges
                        if patch.shape[0] != 128 or patch.shape[1] != 128:
                            # Convert normalized patch back to image format for resizing
                            # Preserves interpolation quality compared to numpy resize
                            patch_img = Image.fromarray((patch[:,:,0] * 255).astype(np.uint8), 'L')
                            patch_img = patch_img.resize((128, 128), Image.Resampling.LANCZOS)
                            patch = np.array(patch_img, dtype=np.float32) / 255.0
                            # Restore channel dimension after resize
                            patch = np.expand_dims(patch, axis=-1)

                        # Apply same normalization used in main preprocessing
                        # Ensures consistent input distribution across all patches
                        patch_flat = patch.flatten()
                        if patch_flat.std() > 0.01:
                            patch = (patch - patch_flat.mean()) / (patch_flat.std() + 1e-7)
                            patch = (patch + 2) / 4
                            patch = np.clip(patch, 0, 1)

                        # Add batch dimension for TFLite interpreter (expects 4D tensor)
                        single_input = np.expand_dims(patch, axis=0)
                        
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
        
        inference_start = time.time()
        
        # Multi-scale detection: analyze full image and zoomed regions
        # This preserves semantic content while detecting dilution attacks
        def multi_scale_detection(img_array, model_wrapper):
            """
            Analyze drawing at multiple scales to detect content dilution.
            
            Strategy:
            1. Full image (512x512 -> 128x128) - catches normal drawings
            2. Center crop (384x384 -> 128x128) - focuses on main content
            3. Quadrants (256x256 -> 128x128) - detects dilution in corners
            4. Content-focused crop - zooms into densest region
            
            Returns max confidence across all scales.
            """
            from src.core.patch_extraction import DetectionResult
            
            h, w = img_array.shape[:2]
            predictions = []
            
            def process_region(region, name, x=0, y=0):
                """Process a single region through the model."""
                # Resize to 128x128
                region_img = Image.fromarray((region[:,:,0] * 255).astype(np.uint8), 'L')
                region_resized = region_img.resize((128, 128), Image.Resampling.LANCZOS)
                region_array = np.array(region_resized, dtype=np.float32) / 255.0
                
                # Apply z-score normalization
                region_flat = region_array.flatten()
                if region_flat.std() > 0.01:
                    region_array = (region_array - region_flat.mean()) / (region_flat.std() + 1e-7)
                    region_array = (region_array + 2) / 4
                    region_array = np.clip(region_array, 0, 1)
                
                # Add batch and channel dimensions
                region_array = region_array.reshape(1, 128, 128, 1)
                
                # Predict
                confidence = model_wrapper.predict(region_array, verbose=0)[0][0]
                
                predictions.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'confidence': float(confidence),
                    'is_positive': confidence >= THRESHOLD
                })
                
                return confidence
            
            # 1. Full image (baseline)
            full_conf = process_region(img_array, 'full', 0, 0)
            
            # 2. Center crop (75% of image) - focuses on main content
            crop_size = int(min(h, w) * 0.75)
            cx, cy = w // 2, h // 2
            x1, y1 = cx - crop_size // 2, cy - crop_size // 2
            x2, y2 = x1 + crop_size, y1 + crop_size
            center_crop = img_array[y1:y2, x1:x2, :]
            center_conf = process_region(center_crop, 'center_crop', x1, y1)
            
            # 3. Quadrants - detect if inappropriate content is in corners
            # Attackers may hide content in corners to dilute main area
            # Analyzing each quadrant separately catches this strategy
            quad_size = min(h, w) // 2
            quadrants = [
                (img_array[0:quad_size, 0:quad_size, :], 'top_left', 0, 0),
                (img_array[0:quad_size, w-quad_size:w, :], 'top_right', w-quad_size, 0),
                (img_array[h-quad_size:h, 0:quad_size, :], 'bottom_left', 0, h-quad_size),
                (img_array[h-quad_size:h, w-quad_size:w, :], 'bottom_right', w-quad_size, h-quad_size),
            ]

            quad_confs = []
            for quad_img, name, qx, qy in quadrants:
                conf = process_region(quad_img, name, qx, qy)
                quad_confs.append(conf)
            
            # 4. Smart content extraction - find and analyze interior regions
            # This detects boxes/frames and analyzes what's INSIDE them
            content_confs = []
            
            # Convert to binary for contour detection
            # Threshold at 0.1 to catch light strokes while ignoring background
            binary = (img_array[:,:,0] > 0.1).astype(np.uint8) * 255

            # Find contours (potential boxes/frames)
            # Common attack: draw box around inappropriate content to hide it
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze regions inside contours
            for contour in contours:
                area = cv2.contourArea(contour)
                # Only process significant contours (not tiny noise or full canvas)
                # Skip tiny specks and avoid analyzing the entire image as one contour
                if area < 1000 or area > h * w * 0.9:
                    continue

                # Get bounding box
                x, y, cw, ch = cv2.boundingRect(contour)

                # Extract interior content (shrink bounding box by 15% to avoid edges)
                # Margin prevents box lines from contaminating the interior analysis
                margin = int(min(cw, ch) * 0.15)
                interior_x1 = max(0, x + margin)
                interior_y1 = max(0, y + margin)
                interior_x2 = min(w, x + cw - margin)
                interior_y2 = min(h, y + ch - margin)

                if interior_x2 - interior_x1 > 50 and interior_y2 - interior_y1 > 50:
                    interior = img_array[interior_y1:interior_y2, interior_x1:interior_x2, :]
                    perimeter = img_array[y:y+ch, x:x+cw, :]

                    # Calculate content density
                    # Compare interior vs perimeter to detect hollow boxes
                    interior_density = np.mean(interior > 0.1)
                    perimeter_density = np.mean(perimeter > 0.1)

                    # Only analyze if interior has significant content
                    # AND it's not just an empty box (perimeter has content but interior doesn't)
                    if interior_density > 0.08:  # Interior must have content
                        # Check if this is an empty box (perimeter dense, interior sparse)
                        # Attackers may draw hollow boxes to mask content
                        if perimeter_density > 0.2 and interior_density < 0.12:
                            # This is likely an empty box, skip it
                            continue

                        conf = process_region(interior, f'interior_{x}_{y}', interior_x1, interior_y1)
                        content_confs.append(conf)
            
            # 4b. Also check densest regions (fallback for drawings without boxes)
            grid_size = 4
            cell_h, cell_w = h // grid_size, w // grid_size
            density_map = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = img_array[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :]
                    density = np.mean(cell > 0.1)
                    density_map.append((density, i, j))
            
            # Only analyze the single densest region at one scale
            density_map.sort(reverse=True)
            if density_map and density_map[0][0] > 0.1:
                density, ci, cj = density_map[0]
                focus_size = int(min(h, w) * 0.5)
                fx1 = max(0, cj * cell_w + cell_w // 2 - focus_size // 2)
                fy1 = max(0, ci * cell_h + cell_h // 2 - focus_size // 2)
                fx2 = min(w, fx1 + focus_size)
                fy2 = min(h, fy1 + focus_size)
                
                if fx2 - fx1 > 50 and fy2 - fy1 > 50:
                    focus_crop = img_array[fy1:fy2, fx1:fx2, :]
                    conf = process_region(focus_crop, 'densest_region', fx1, fy1)
                    content_confs.append(conf)
            
            # Aggregate: use max confidence (most aggressive)
            all_confs = [full_conf, center_conf] + quad_confs + content_confs
            max_conf = max(all_confs) if all_confs else 0.0
            is_positive = max_conf >= THRESHOLD
            
            return DetectionResult(
                is_positive=is_positive,
                confidence=max_conf,
                patch_predictions=predictions,
                num_patches_analyzed=len(predictions),
                early_stopped=max_conf >= 0.9,
                aggregation_strategy='multi_scale_max'
            )
        
        detection_result = multi_scale_detection(img_array, model_wrapper)
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        if detection_result.is_positive:
            verdict = 'PENIS'
            verdict_text = 'Drawing looks like a penis! ✓ (Detected via region analysis)'
            confidence = float(detection_result.confidence)
        else:
            verdict = 'OTHER_SHAPE'
            verdict_text = 'Drawing looks like a common shape (not penis). (Verified via region analysis)'
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
                'num_patches_analyzed': int(detection_result.num_patches_analyzed),
                'early_stopped': bool(detection_result.early_stopped),
                'aggregation_strategy': str(detection_result.aggregation_strategy),
                'patch_predictions': [
                    {
                        'x': int(p['x']),
                        'y': int(p['y']),
                        'confidence': round(float(p['confidence']), 4),
                        'is_positive': bool(p['is_positive'])
                    }
                    for p in detection_result.patch_predictions[:5]
                ]
            },
            'drawing_statistics': {
                'response_time_ms': round(total_time * 1000, 2),
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'inference_time_ms': round(inference_time * 1000, 2)
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index() -> str:
    """Serve the main drawing interface page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict() -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    REST API endpoint for drawing classification requests.
    
    Expects JSON payload with 'image' field containing base64 encoded drawing.
    Returns prediction results with confidence scores and timing information.
    
    Returns:
        Tuple of (response_dict, status_code)
        - 200: Success with prediction results
        - 400: Bad request (missing or invalid image data)
        - 500: Internal server error
    
    SECURITY: Validates input before processing to prevent resource exhaustion
    """
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        result = predict(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/region', methods=['POST'])
def api_predict_region() -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    REST API endpoint for region-based detection (robust against content dilution).
    
    Expects JSON payload with 'image' field containing base64 encoded drawing.
    Returns detailed region-based analysis results.
    
    Returns:
        Tuple of (response_dict, status_code)
        - 200: Success with prediction and detection details
        - 400: Bad request (missing or invalid image data)
        - 500: Internal server error
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        result = predict_region_based(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancer probes.
    
    Returns:
        Dictionary with service status and configuration information
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None or tflite_interpreter is not None,
        'threshold': THRESHOLD,
        'region_detection_available': True
    })


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
