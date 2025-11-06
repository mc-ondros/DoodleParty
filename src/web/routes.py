"""API routes for DoodleHunter web application.

Defines all REST API endpoints for the Flask application.
Separates route definitions from application logic for better organization.

Related:
- src/web/app.py (Flask application and prediction logic)
- src/core/inference.py (model inference)

Exports:
- All route handlers are registered with the Flask app
"""

from flask import Blueprint, request, jsonify, render_template

# Create blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/predict', methods=['POST'])
def predict():
    """REST API endpoint for drawing classification requests."""
    from src.web.app import predict as predict_func
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        result = predict_func(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/predict/region', methods=['POST'])
def predict_region():
    """REST API endpoint for region-based detection."""
    from src.web.app import predict_region_based
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        result = predict_region_based(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring and load balancer probes."""
    from src.web.app import model, tflite_interpreter, THRESHOLD
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None or tflite_interpreter is not None,
        'threshold': THRESHOLD,
        'region_detection_available': True
    })


def register_routes(app):
    """Register all routes with the Flask application."""
    app.register_blueprint(api_bp)
    
    @app.route('/')
    def index():
        return render_template('index.html')


if __name__ == '__main__':
    print('API Routes Module')
    print('Routes are registered with the Flask app via register_routes().')
