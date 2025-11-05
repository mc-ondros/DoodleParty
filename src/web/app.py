"""
Flask app for binary classification of drawings: penis vs other shapes.
Uses a model trained on QuickDraw dataset to classify drawings.
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
    
    models_dir = Path(__file__).parent.parent / "models"
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Try to find the most recent model file
    model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Use the most recently modified model
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
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
    Converts to 128x128 with black background and white strokes to match training data.
    
    Args:
        image_data: Base64 encoded image data from canvas
    
    Returns:
        Preprocessed numpy array suitable for model input (1, 128, 128, 1)
    """
    # Decode base64 image
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Canvas has WHITE background and BLACK strokes
    # Model expects BLACK background and WHITE strokes
    # So we need to invert: 255 - pixel_value
    img_array = np.array(image, dtype=np.uint8)
    img_array = 255 - img_array  # Invert colors
    
    # Resize to 128x128 using high-quality LANCZOS
    image_inverted = Image.fromarray(img_array, mode='L')
    image_resized = image_inverted.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to 0-1
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Apply per-image normalization (same as training)
    img_flat = img_array.flatten()
    if img_flat.std() > 0.01:  # Only normalize if there's variation
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        img_array = (img_array + 3) / 6  # Rescale to approximately 0-1 range
        img_array = np.clip(img_array, 0, 1)
    
    # Reshape for model input (batch_size=1, height=128, width=128, channels=1)
    img_array = img_array.reshape(1, 128, 128, 1)
    
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
            'model_info': 'Binary classifier: penis vs 21 common shapes'
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


# Load model on startup (works for both flask run and direct execution)
try:
    load_model_and_mapping()
except Exception as e:
    print(f"Warning: Failed to load model on startup: {e}")
    print("Model will need to be loaded manually")


if __name__ == '__main__':
    # Run Flask app
    print("Starting Penis Classifier Interface...")
    print("Visit http://localhost:5000 to use the drawing board")
    print("Model: Binary classification (penis vs other shapes)")
    app.run(debug=True, host='0.0.0.0', port=5000)
