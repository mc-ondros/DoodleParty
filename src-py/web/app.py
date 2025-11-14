"""
Flask ML service with REST endpoints for DoodleParty.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
import base64
from PIL import Image
import os
import sys

# Add src-py root to import path for core modules
CURRENT_DIR = os.path.dirname(__file__)
SRC_PY_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_PY_ROOT not in sys.path:
    sys.path.insert(0, SRC_PY_ROOT)

from core.inference import InferenceEngine  # type: ignore
from routes import create_routes  # type: ignore

app = Flask(__name__)
CORS(app)

# Initialize inference engine if model is available
MODEL_PATH = os.environ.get('DOODLEPARTY_MODEL', os.path.abspath(os.path.join(CURRENT_DIR, '../../models/model.h5')))
_inference_engine = None
if os.path.exists(MODEL_PATH):
    try:
        _inference_engine = InferenceEngine(MODEL_PATH)
    except Exception:
        _inference_engine = None

# Register API blueprint
app.register_blueprint(create_routes(_inference_engine))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': _inference_engine is not None}), 200


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a drawing (legacy endpoint).
    
    Expected JSON:
    {
        "image": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        if _inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if image_data.startswith('data:image') else image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        predictions = _inference_engine.predict_single(image_array)
        
        # For binary models, return probability
        offensive_prob = list(predictions.values())[0] if predictions else 0.0
        return jsonify({
            'predictions': predictions,
            'offensive_prob': float(offensive_prob)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/moderate', methods=['POST'])
def moderate():
    """
    Moderate a drawing for inappropriate content (legacy endpoint).
    
    Expected JSON:
    {
        "image": "base64_encoded_image",
        "threshold": 0.5
    }
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        threshold = data.get('threshold', 0.5)
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        if _inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if image_data.startswith('data:image') else image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_array = np.array(image).astype(np.float32) / 255.0
        
        predictions = _inference_engine.predict_single(image_array)
        offensive_prob = list(predictions.values())[0] if predictions else 0.0
        
        return jsonify({
            'status': 'unsafe' if offensive_prob > threshold else 'safe',
            'confidence': float(offensive_prob),
            'reason': 'Offensive content detected' if offensive_prob > threshold else None
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    """
    Classify multiple drawings (legacy endpoint).
    
    Expected JSON:
    {
        "images": ["base64_image1", "base64_image2", ...]
    }
    """
    try:
        data = request.get_json()
        images_data = data.get('images', [])
        
        if not images_data:
            return jsonify({'error': 'No images provided'}), 400
        if _inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        results = []
        for image_data in images_data:
            try:
                image_bytes = base64.b64decode(image_data.split(',')[1] if image_data.startswith('data:image') else image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('L')
                image_array = np.array(image).astype(np.float32) / 255.0
                
                predictions = _inference_engine.predict_single(image_array)
                offensive_prob = list(predictions.values())[0] if predictions else 0.0
                results.append({
                    'predictions': predictions,
                    'offensive_prob': float(offensive_prob)
                })
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
