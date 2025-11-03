"""
Flask app for testing DoodleHunter ML model with a drawing interface.
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
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Global model and mapping
model = None
idx_to_class = None
THRESHOLD = 0.5


def load_model_and_mapping():
    """Load trained model and class mapping."""
    global model, idx_to_class
    
    model_path = Path(__file__).parent.parent / "models" / "quickdraw_model.h5"
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    try:
        with open(data_dir / "class_mapping.pkl", 'rb') as f:
            class_mapping = pickle.load(f)
        
        # Handle nested structure - extract only the simple key-value mappings
        idx_to_class = {}
        for k, v in class_mapping.items():
            if isinstance(v, int):  # Only take integer values (class indices)
                idx_to_class[v] = k
        
        # Fallback if no valid mappings found
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
    Preprocess canvas image data for model prediction.
    
    Args:
        image_data: Base64 encoded image data from canvas
    
    Returns:
        Preprocessed numpy array suitable for model input
    """
    # Decode base64 image
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale and resize to 28x28
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array


def predict(image_data):
    """
    Make prediction on drawn image.
    
    Args:
        image_data: Base64 encoded image data
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # Predict
        probability = model.predict(img_array, verbose=0)[0][0]
        
        # Determine class based on threshold
        if probability >= THRESHOLD:
            verdict = 'IN-DISTRIBUTION'
            verdict_text = "Looks like a QuickDraw doodle! ✓"
            confidence = float(probability)
        else:
            verdict = 'OUT-OF-DISTRIBUTION'
            verdict_text = "Doesn't match QuickDraw style. ✗"
            confidence = float(1 - probability)
        
        return {
            'success': True,
            'verdict': verdict,
            'verdict_text': verdict_text,
            'confidence': round(confidence, 4),
            'raw_probability': round(float(probability), 4),
            'threshold': THRESHOLD
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        result = predict(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'threshold': THRESHOLD
    })


if __name__ == '__main__':
    # Load model on startup
    load_model_and_mapping()
    
    # Run Flask app
    print("Starting DoodleHunter Interface...")
    print("Visit http://localhost:5000 to use the drawing board")
    app.run(debug=True, host='0.0.0.0', port=5000)
