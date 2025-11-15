#!/usr/bin/env python3
"""
Local ML Server for Inappropriate Content Detection
Runs a Flask server that receives images and returns classification results
"""

import os
import sys
import io
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
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

# Global model variable
model = None


def load_model():
    """Load the ML model"""
    global model
    
    try:
        logger.info(f"Loading ML model from {ML_MODEL_PATH}")
        
        if not os.path.exists(ML_MODEL_PATH):
            logger.warning(f"Model file not found at {ML_MODEL_PATH}")
            logger.warning("Using mock predictions for testing")
            return None
        
        model = tf.keras.models.load_model(ML_MODEL_PATH)
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using mock predictions for testing")
        return None


def preprocess_image(image_data):
    """
    Preprocess image for ML model
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        numpy array ready for model input
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((ML_INPUT_SIZE, ML_INPUT_SIZE), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to 0-1
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.debug(f"Preprocessed image shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


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
            logger.warning("Using mock prediction (model not loaded)")
            # Random prediction for testing
            score = np.random.random()
            is_inappropriate = score > ML_CONFIDENCE_THRESHOLD
            
            return {
                'is_inappropriate': bool(is_inappropriate),
                'confidence': float(score),
                'category': 'inappropriate' if is_inappropriate else 'safe',
                'mock': True
            }
        
        # Run actual prediction
        predictions = model.predict(image_array, verbose=0)
        score = float(predictions[0][0])
        
        is_inappropriate = score > ML_CONFIDENCE_THRESHOLD
        
        return {
            'is_inappropriate': bool(is_inappropriate),
            'confidence': float(score),
            'category': 'inappropriate' if is_inappropriate else 'safe',
            'mock': False
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': ML_MODEL_PATH,
        'input_size': ML_INPUT_SIZE,
        'threshold': ML_CONFIDENCE_THRESHOLD
    })


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
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({
                'error': 'Missing image_data in request'
            }), 400
        
        image_data = data['image_data']
        session_id = data.get('session_id', 'unknown')
        bbox = data.get('bbox', {})
        
        logger.info(f"Classification request from session: {session_id}")
        logger.debug(f"Bounding box: {bbox}")
        
        # Preprocess image
        image_array = preprocess_image(image_data)
        
        # Run prediction
        result = predict(image_array)
        
        logger.info(f"Prediction result: {result['category']} (confidence: {result['confidence']:.3f})")
        
        # Add metadata
        result['session_id'] = session_id
        result['bbox'] = bbox
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'is_inappropriate': False  # Safe default on error
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
                image_array = preprocess_image(img_data['image_data'])
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
    logger.info("=" * 60)
    logger.info("Starting ML Content Detection Server")
    logger.info("=" * 60)
    logger.info(f"Model path: {ML_MODEL_PATH}")
    logger.info(f"Input size: {ML_INPUT_SIZE}x{ML_INPUT_SIZE}")
    logger.info(f"Confidence threshold: {ML_CONFIDENCE_THRESHOLD}")
    logger.info(f"Server port: {SERVER_PORT}")
    logger.info("=" * 60)
    
    # Load model
    load_model()
    
    # Start server
    logger.info(f"Starting Flask server on http://localhost:{SERVER_PORT}")
    logger.info("Press Ctrl+C to stop")
    
    app.run(
        host='0.0.0.0',
        port=SERVER_PORT,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
