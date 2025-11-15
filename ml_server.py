#!/usr/bin/env python3
"""
Local ML Server for Content Detection
Runs a Flask server that receives drawing images and classifies them for inappropriate content.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
import sys
import logging
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.model_loader import load_model, predict
except ImportError:
    print("Warning: Could not import model_loader. Using mock predictions.")
    load_model = None
    predict = None

# Configuration
PORT = 5000
HOST = '0.0.0.0'  # Listen on all interfaces for network access
MODEL_PATH = 'models/quickdraw_model_int8.tflite'  # Default model path (quantized QuickDraw model)
INAPPROPRIATE_THRESHOLD = 0.7  # Confidence threshold for inappropriate classification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global model variable
model = None


def load_ml_model():
    """Load the ML model at startup"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Using mock predictions.")
        return None
    
    try:
        if load_model is not None:
            model = load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return model
        else:
            logger.warning("Model loader not available. Using mock predictions.")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def preprocess_image(image_bytes):
    """
    Preprocess image for model inference
    
    Args:
        image_bytes: Raw image bytes from request
        
    Returns:
        numpy array ready for model prediction
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def mock_predict(image_array):
    """
    Mock prediction for testing without a real model
    Returns random predictions
    """
    import random
    
    # For testing, randomly classify as inappropriate ~10% of the time
    is_inappropriate = random.random() < 0.1
    confidence = random.uniform(0.5, 0.95) if is_inappropriate else random.uniform(0.05, 0.4)
    
    logger.info(f"Mock prediction: inappropriate={is_inappropriate}, confidence={confidence:.3f}")
    
    return {
        'is_inappropriate': is_inappropriate,
        'confidence': float(confidence),
        'class_name': 'inappropriate' if is_inappropriate else 'appropriate',
        'mock': True
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })


@app.route('/classify', methods=['POST'])
def classify_drawing():
    """
    Main classification endpoint
    
    Expects:
        - image: PNG image file (form-data)
        - sessionId: User session ID (form-data, optional)
        - bbox: Bounding box JSON string (form-data, optional)
        
    Returns:
        JSON with classification result:
        {
            'is_inappropriate': bool,
            'confidence': float,
            'class_name': str
        }
    """
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Get optional metadata
        session_id = request.form.get('sessionId', 'unknown')
        bbox_str = request.form.get('bbox', None)
        
        logger.info(f"Classification request from session {session_id}")
        if bbox_str:
            logger.debug(f"Bounding box: {bbox_str}")
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        logger.debug(f"Image shape: {img_array.shape}")
        
        # Predict
        if model is not None and predict is not None:
            # Real model prediction
            prediction = predict(model, img_array)
            
            # Extract results
            is_inappropriate = prediction['confidence'] >= INAPPROPRIATE_THRESHOLD
            result = {
                'is_inappropriate': bool(is_inappropriate),
                'confidence': float(prediction['confidence']),
                'class_name': prediction['class_name'],
                'mock': False
            }
        else:
            # Mock prediction
            result = mock_predict(img_array)
        
        logger.info(f"Result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config():
    """Get server configuration"""
    return jsonify({
        'threshold': INAPPROPRIATE_THRESHOLD,
        'model_path': MODEL_PATH,
        'model_loaded': model is not None
    })


@app.route('/config', methods=['POST'])
def update_config():
    """Update server configuration"""
    global INAPPROPRIATE_THRESHOLD
    
    data = request.get_json()
    
    if 'threshold' in data:
        INAPPROPRIATE_THRESHOLD = float(data['threshold'])
        logger.info(f"Updated threshold to {INAPPROPRIATE_THRESHOLD}")
    
    return get_config()


def main():
    """Start the ML server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DoodleParty ML Content Detection Server')
    parser.add_argument('--port', type=int, default=PORT, help='Port to run server on')
    parser.add_argument('--host', type=str, default=HOST, help='Host to bind to')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--threshold', type=float, default=INAPPROPRIATE_THRESHOLD, 
                       help='Confidence threshold for inappropriate classification')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Update globals
    global MODEL_PATH, INAPPROPRIATE_THRESHOLD
    MODEL_PATH = args.model
    INAPPROPRIATE_THRESHOLD = args.threshold
    
    # Load model
    logger.info("Starting ML server...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Threshold: {INAPPROPRIATE_THRESHOLD}")
    
    load_ml_model()
    
    # Start server
    logger.info(f"Server starting on {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
