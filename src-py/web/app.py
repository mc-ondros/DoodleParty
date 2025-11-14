"""
Flask ML service with REST endpoints for DoodleParty.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a drawing.
    
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
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # TODO: Run inference
        # predictions = inference_engine.predict_single(image_array)
        
        return jsonify({
            'label': 'example',
            'confidence': 0.95
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/moderate', methods=['POST'])
def moderate():
    """
    Moderate a drawing for inappropriate content.
    
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
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # TODO: Run moderation
        
        return jsonify({
            'status': 'safe',
            'confidence': 0.98,
            'reason': None
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    """
    Classify multiple drawings.
    
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
        
        results = []
        for image_data in images_data:
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('L')
                image_array = np.array(image).astype(np.float32) / 255.0
                
                # TODO: Run inference
                results.append({
                    'label': 'example',
                    'confidence': 0.95
                })
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
