"""
API route definitions for Flask ML service.

Endpoints:
- GET /api/health - Health check
- POST /api/predict - Single image classification
- POST /api/predict/shape - Shape-based detection with stroke awareness
- POST /api/predict/tile - Tile-based detection
- POST /api/predict/region - Contour-based region detection
"""

from flask import Blueprint, request, jsonify
import numpy as np
import base64
import io
from PIL import Image


def create_routes(inference_engine, moderation_engine=None):
    """
    Create API route blueprints.
    
    Args:
        inference_engine: InferenceEngine instance for classification
        moderation_engine: Optional moderation engine instance
    
    Returns:
        Blueprint for routes
    """
    api = Blueprint('api', __name__, url_prefix='/api')
    
    def decode_base64_image(image_b64: str) -> np.ndarray:
        """
        Decode base64 image string to numpy array.
        Handles both data URLs and raw base64.
        """
        # Remove data URL prefix if present
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        return np.array(image, dtype=np.float32) / 255.0
    
    @api.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'service': 'ml-inference',
            'version': '1.0.0'
        }), 200
    
    @api.route('/status', methods=['GET'])
    def status():
        """Get API status."""
        return jsonify({
            'status': 'online',
            'version': '1.0.0',
            'model': 'custom_cnn',
            'endpoints': [
                '/api/health',
                '/api/predict',
                '/api/predict/shape',
                '/api/predict/tile',
                '/api/predict/region'
            ]
        }), 200
    
    @api.route('/predict', methods=['POST'])
    def predict():
        """
        Single image classification endpoint.
        
        Expected JSON:
        {
            "image": "base64_encoded_image"
        }
        
        Response:
        {
            "success": true,
            "verdict": "APPROVED" or "REJECTED",
            "confidence": 0.95,
            "model_version": "1.0.0"
        }
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            # Decode and preprocess image
            image = decode_base64_image(data['image'])
            
            # Run inference
            if inference_engine is None:
                return jsonify({'error': 'Inference engine not initialized'}), 500
            
            predictions = inference_engine.predict_single(image)
            
            # For binary classification, use the offensive probability
            offensive_prob = list(predictions.values())[0] if predictions else 0.0
            threshold = data.get('threshold', 0.5)
            
            return jsonify({
                'success': True,
                'verdict': 'REJECTED' if offensive_prob > threshold else 'APPROVED',
                'confidence': float(offensive_prob),
                'model_version': '1.0.0'
            }), 200
        
        except ValueError as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': f'Inference failed: {str(e)}'}), 500
    
    @api.route('/predict/shape', methods=['POST'])
    def predict_shape():
        """
        Shape-based detection with stroke awareness.
        
        Expected JSON:
        {
            "image": "base64_encoded_image",
            "stroke_history": [
                {
                    "points": [{"x": 120, "y": 260, "t": 1730980000000}, ...],
                    "timestamp": 1730980000000
                }
            ],
            "min_shape_area": 100
        }
        
        Response:
        {
            "success": true,
            "verdict": "APPROVED" or "REJECTED",
            "confidence": 0.12,
            "detection_details": {
                "num_shapes_analyzed": 3,
                "shape_predictions": [...],
                "grouped_boxes": [...]
            }
        }
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            image = decode_base64_image(data['image'])
            stroke_history = data.get('stroke_history', [])
            min_shape_area = data.get('min_shape_area', 100)
            threshold = data.get('threshold', 0.5)
            
            # TODO: Implement shape-based detection
            # This would use stroke_history to cluster strokes into shapes
            # Then classify each shape independently
            
            return jsonify({
                'success': True,
                'verdict': 'APPROVED',
                'confidence': 0.0,
                'detection_details': {
                    'num_shapes_analyzed': 0,
                    'shape_predictions': [],
                    'grouped_boxes': []
                }
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Shape detection failed: {str(e)}'}), 500
    
    @api.route('/predict/tile', methods=['POST'])
    def predict_tile():
        """
        Tile-based detection with grid partitioning.
        
        Expected JSON:
        {
            "image": "base64_encoded_image",
            "tile_size": 64
        }
        
        Response:
        {
            "success": true,
            "verdict": "APPROVED" or "REJECTED",
            "confidence": 0.95,
            "tile_analysis": {
                "grid_size": [8, 8],
                "num_dirty_tiles": 5,
                "max_tile_confidence": 0.95
            }
        }
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            image = decode_base64_image(data['image'])
            tile_size = data.get('tile_size', 64)
            threshold = data.get('threshold', 0.5)
            
            # TODO: Implement tile-based detection
            # This would divide the image into tiles and classify each
            
            return jsonify({
                'success': True,
                'verdict': 'APPROVED',
                'confidence': 0.0,
                'tile_analysis': {
                    'grid_size': [8, 8],
                    'num_dirty_tiles': 0,
                    'max_tile_confidence': 0.0
                }
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Tile detection failed: {str(e)}'}), 500
    
    @api.route('/predict/region', methods=['POST'])
    def predict_region():
        """
        Contour-based region detection.
        
        Expected JSON:
        {
            "image": "base64_encoded_image",
            "contour_mode": "RETR_TREE"
        }
        
        Response:
        {
            "success": true,
            "verdict": "APPROVED" or "REJECTED",
            "confidence": 0.95,
            "region_analysis": {
                "num_contours": 5,
                "max_confidence": 0.95,
                "contour_predictions": [...]
            }
        }
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            image = decode_base64_image(data['image'])
            threshold = data.get('threshold', 0.5)
            
            # TODO: Implement contour-based detection
            # This would extract contours and classify each independently
            
            return jsonify({
                'success': True,
                'verdict': 'APPROVED',
                'confidence': 0.0,
                'region_analysis': {
                    'num_contours': 0,
                    'max_confidence': 0.0,
                    'contour_predictions': []
                }
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Region detection failed: {str(e)}'}), 500
    
    @api.route('/classify', methods=['POST'])
    def classify():
        """Classify a drawing from base64 image (legacy endpoint)."""
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            image = decode_base64_image(data['image'])
            predictions = inference_engine.predict_single(image)
            
            return jsonify({
                'success': True,
                'predictions': predictions
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @api.route('/moderate', methods=['POST'])
    def moderate():
        """
        Moderate a drawing for inappropriate content (legacy endpoint).
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing image in request'}), 400
            
            image = decode_base64_image(data['image'])
            predictions = inference_engine.predict_single(image)
            
            # For binary classification
            offensive_prob = list(predictions.values())[0] if predictions else 0.0
            threshold = data.get('threshold', 0.5)
            
            return jsonify({
                'success': True,
                'status': 'unsafe' if offensive_prob > threshold else 'safe',
                'confidence': float(offensive_prob),
                'reason': 'Offensive content detected' if offensive_prob > threshold else None
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return api
