#!/usr/bin/env python3
"""
Local ML Server for Inappropriate Content Detection
Runs a Flask server that receives images and returns classification results
"""

import os
import sys
import io
import json
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', './models/content_detector.h5')
ML_INPUT_SIZE = int(os.getenv('ML_INPUT_SIZE', '128'))
ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.7'))
SERVER_PORT = int(os.getenv('ML_SERVER_PORT', '5000'))

# Global prediction engine
model = None


class TFLitePredictor:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = tuple(self.input_details[0]['shape'])
        
        # Log quantization parameters
        logger.info(f"Input details: {self.input_details[0]}")
        logger.info(f"Output details: {self.output_details[0]}")

    def predict(self, image_array: np.ndarray) -> float:
        # Set input tensor
        input_details = self.input_details[0]
        
        # Handle input quantization if needed
        if input_details['dtype'] == np.uint8:
            # Quantize input
            input_scale, input_zero_point = input_details['quantization']
            if input_scale > 0:
                # Quantize: quantized = input / scale + zero_point
                image_array = (image_array / input_scale + input_zero_point).astype(np.uint8)
                logger.debug(f"Input quantized to uint8: range [{image_array.min()}, {image_array.max()}]")
        
        self.interpreter.set_tensor(input_details['index'], image_array)
        self.interpreter.invoke()
        
        # Get output and dequantize if needed
        output_details = self.output_details[0]
        output = self.interpreter.get_tensor(output_details['index'])
        
        # Handle output dequantization
        if output_details['dtype'] == np.uint8 or output_details['dtype'] == np.int8:
            # Dequantize output
            output_scale, output_zero_point = output_details['quantization']
            if output_scale > 0:
                # Dequantize: dequantized = (quantized - zero_point) * scale
                output = (output.astype(np.float32) - output_zero_point) * output_scale
                logger.debug(f"Output dequantized: {output[0][0]:.6f} (raw: {self.interpreter.get_tensor(output_details['index'])[0][0]})")
        
        result = float(output[0][0])
        logger.debug(f"Final prediction: {result:.6f}")
        return result


def load_model():
    """Load the ML model"""
    global model
    
    try:
        logger.info(f"Loading ML model from {ML_MODEL_PATH}")
        
        if not os.path.exists(ML_MODEL_PATH):
            logger.warning(f"Model file not found at {ML_MODEL_PATH}")
            logger.warning("Using mock predictions for testing")
            return None
        
        if ML_MODEL_PATH.lower().endswith('.tflite'):
            model = TFLitePredictor(ML_MODEL_PATH)
            logger.info(f"TFLite model loaded successfully. Input shape: {model.input_shape}")
            return model

        model = tf.keras.models.load_model(ML_MODEL_PATH)
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using mock predictions for testing")
        return None


def decode_base64_image(image_data: str) -> Image.Image:
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to model-ready numpy array
    
    Expects grayscale image with black background (0) and white strokes (255)
    matching QuickDraw training format.
    """
    try:
        # Convert to grayscale (L mode) - single channel
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size
        image = image.resize((ML_INPUT_SIZE, ML_INPUT_SIZE), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to 0-1 range (TFLite will quantize if needed)
        img_array = img_array / 255.0
        
        # Add channel dimension: (128, 128) -> (128, 128, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Add batch dimension: (128, 128, 1) -> (1, 128, 128, 1)
        img_array = np.expand_dims(img_array, axis=0)

        logger.debug(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.debug(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        logger.debug(f"Mean: {img_array.mean():.3f}, Std: {img_array.std():.3f}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def extract_request_payload():
    payload = {}
    json_payload = request.get_json(silent=True)
    if isinstance(json_payload, dict):
        payload.update(json_payload)

    for key in request.form:
        payload[key] = request.form.get(key)

    return payload


def parse_request_image(payload: dict) -> Image.Image:
    if 'image' in request.files:
        image_file = request.files['image']
        image_file.stream.seek(0)
        return Image.open(image_file.stream)

    image_data = payload.get('image_data')
    if not image_data:
        raise ValueError('Missing image_data in request payload')

    return decode_base64_image(image_data)


def parse_bbox(bbox_raw):
    if not bbox_raw:
        return {}
    if isinstance(bbox_raw, dict):
        return bbox_raw
    try:
        return json.loads(bbox_raw)
    except (TypeError, ValueError):
        return {}


def predict(image_array):
    """
    Run prediction on preprocessed image
    
    Args:
        image_array: Preprocessed numpy array
        
    Returns:
        dict with prediction results
    """
    global model
    
    try:
        if model is None:
            # Mock prediction for testing
            logger.warning("⚠️ Using mock prediction (model not loaded)")
            # Random prediction for testing
            score = np.random.random()
            is_inappropriate = score > ML_CONFIDENCE_THRESHOLD
            
            logger.debug(f"Mock prediction: score={score:.3f}, threshold={ML_CONFIDENCE_THRESHOLD}")
            
            return {
                'is_inappropriate': bool(is_inappropriate),
                'confidence': float(score),
                'category': 'inappropriate' if is_inappropriate else 'safe',
                'mock': True
            }
        
        logger.debug(f"Running inference with model type: {type(model).__name__}")
        
        if isinstance(model, TFLitePredictor):
            score = model.predict(image_array)
            logger.debug(f"TFLite prediction score: {score:.3f}")
        else:
            predictions = model.predict(image_array, verbose=0)
            score = float(predictions[0][0])
            logger.debug(f"Model prediction score: {score:.3f}")
        
        is_inappropriate = score > ML_CONFIDENCE_THRESHOLD
        logger.debug(f"Threshold: {ML_CONFIDENCE_THRESHOLD}, Result: {'INAPPROPRIATE' if is_inappropriate else 'SAFE'}")
        
        return {
            'is_inappropriate': bool(is_inappropriate),
            'confidence': float(score),
            'category': 'inappropriate' if is_inappropriate else 'safe',
            'mock': False
        }
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}", exc_info=True)
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_info = {
        'loaded': model is not None,
        'type': type(model).__name__ if model else None,
        'path': ML_MODEL_PATH,
        'exists': os.path.exists(ML_MODEL_PATH) if ML_MODEL_PATH else False
    }
    
    if isinstance(model, TFLitePredictor):
        # Convert numpy int32 to Python int for JSON serialization
        model_info['input_shape'] = [int(x) for x in model.input_shape]
    elif model is not None and hasattr(model, 'input_shape'):
        model_info['input_shape'] = str(model.input_shape)
    
    response = {
        'status': 'healthy',
        'model': model_info,
        'config': {
            'input_size': int(ML_INPUT_SIZE),
            'threshold': float(ML_CONFIDENCE_THRESHOLD),
            'port': int(SERVER_PORT)
        },
        'endpoints': {
            'classify': '/classify (POST)',
            'batch_classify': '/batch_classify (POST)',
            'health': '/health (GET)'
        }
    }
    
    logger.info("Health check requested")
    return jsonify(response)


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify image for inappropriate content
    
    Expected JSON:
    {
        "image_data": "base64_encoded_image",
        "session_id": "optional_session_id",
        "bbox": {"minX": 0, "maxX": 100, "minY": 0, "maxY": 100}
    }
    
    Returns JSON:
    {
        "is_inappropriate": true/false,
        "confidence": 0.0-1.0,
        "category": "inappropriate" or "safe"
    }
    """
    request_start_time = __import__('time').time()
    
    try:
        payload = extract_request_payload()
        image = parse_request_image(payload)
        session_id = payload.get('session_id', payload.get('sessionId', 'unknown'))
        bbox = parse_bbox(payload.get('bbox'))

        logger.info('='*60)
        logger.info(f"📥 CLASSIFICATION REQUEST")
        logger.info(f"Session: {session_id}")
        logger.info(f"Image size: {image.size}, mode: {image.mode}")
        logger.info(f"Bounding box: {bbox}")

        # Preprocess image
        preprocess_start = __import__('time').time()
        image_array = preprocess_image(image)
        preprocess_time = (__import__('time').time() - preprocess_start) * 1000
        logger.info(f"✓ Preprocessing completed in {preprocess_time:.2f}ms")
        logger.info(f"  Array shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Run prediction
        predict_start = __import__('time').time()
        result = predict(image_array)
        predict_time = (__import__('time').time() - predict_start) * 1000
        logger.info(f"✓ Prediction completed in {predict_time:.2f}ms")

        total_time = (__import__('time').time() - request_start_time) * 1000
        
        # Log result with visual indicator
        status_icon = "⚠️" if result['is_inappropriate'] else "✅"
        status_color = "INAPPROPRIATE" if result['is_inappropriate'] else "SAFE"
        
        logger.info(f"{status_icon} RESULT: {status_color}")
        logger.info(f"  Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        logger.info(f"  Category: {result['category']}")
        logger.info(f"  Mock: {result.get('mock', False)}")
        logger.info(f"  Total time: {total_time:.2f}ms")
        logger.info('='*60)

        result['session_id'] = session_id
        result['bbox'] = bbox
        result['processing_time_ms'] = round(total_time, 2)

        return jsonify(result)

    except Exception as e:
        total_time = (__import__('time').time() - request_start_time) * 1000
        logger.error('='*60)
        logger.error(f"❌ ERROR in classify endpoint ({total_time:.2f}ms)")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error('='*60)
        return jsonify({
            'error': str(e),
            'is_inappropriate': False,  # Safe default on error
            'confidence': 0.0,
            'category': 'error',
            'processing_time_ms': round(total_time, 2)
        }), 500


@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    """
    Classify multiple images in batch
    
    Expected JSON:
    {
        "images": [
            {"image_data": "base64...", "id": "obj1"},
            {"image_data": "base64...", "id": "obj2"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'error': 'Missing images in request'}), 400
        
        images = data['images']
        results = []
        
        logger.info(f"Batch classification request with {len(images)} images")
        
        for img_data in images:
            try:
                image = decode_base64_image(img_data['image_data'])
                image_array = preprocess_image(image)
                result = predict(image_array)
                result['id'] = img_data.get('id', 'unknown')
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {img_data.get('id')}: {e}")
                results.append({
                    'id': img_data.get('id', 'unknown'),
                    'error': str(e),
                    'is_inappropriate': False
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in batch_classify endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🤖 ML CONTENT DETECTION SERVER")
    print("=" * 60)
    print(f"📁 Model path: {ML_MODEL_PATH}")
    print(f"📏 Input size: {ML_INPUT_SIZE}×{ML_INPUT_SIZE}")
    print(f"🎯 Confidence threshold: {ML_CONFIDENCE_THRESHOLD}")
    print(f"🌐 Server port: {SERVER_PORT}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("Starting ML Content Detection Server")
    logger.info("=" * 60)
    logger.info(f"Model path: {ML_MODEL_PATH}")
    logger.info(f"Input size: {ML_INPUT_SIZE}x{ML_INPUT_SIZE}")
    logger.info(f"Confidence threshold: {ML_CONFIDENCE_THRESHOLD}")
    logger.info(f"Server port: {SERVER_PORT}")
    logger.info("=" * 60)
    
    # Load model
    model_loaded = load_model()
    
    if model_loaded is None:
        print("⚠️  WARNING: No model loaded - using MOCK predictions")
        print("   To use real predictions, set ML_MODEL_PATH to a valid model file")
    else:
        print("✅ Model loaded successfully")
    
    print("=" * 60)
    print(f"🚀 Server starting on http://0.0.0.0:{SERVER_PORT}")
    print(f"📊 Health check: http://localhost:{SERVER_PORT}/health")
    print(f"🔍 Classify endpoint: http://localhost:{SERVER_PORT}/classify")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    # Start server
    logger.info(f"Starting Flask server on http://0.0.0.0:{SERVER_PORT}")
    logger.info("Server ready - waiting for requests...")
    
    app.run(
        host='0.0.0.0',
        port=SERVER_PORT,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
