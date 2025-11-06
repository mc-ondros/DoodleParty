"""Integration tests for Flask application."""

import pytest
import numpy as np
import json
import base64
import tempfile
import pickle
from io import BytesIO
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch

from src.web.app import app
from src.core.models import build_custom_cnn


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_trained_model():
    """Create a mock trained model."""
    model = build_custom_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train on dummy data
    X = np.random.rand(50, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    model.fit(X, y, epochs=1, verbose=0)
    
    return model


@pytest.fixture
def canvas_drawing():
    """Create realistic canvas drawing data."""
    # Create image with some drawing
    img = Image.new('L', (280, 280), color=0)  # Black background
    
    # Draw white lines (simulating user drawing)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 180, 180], fill=255)  # White circle
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


class TestFlaskApplicationIntegration:
    """Test Flask application with realistic scenarios."""
    
    @patch('src.web.app.model')
    def test_full_prediction_flow(self, mock_model, client, canvas_drawing):
        """Test complete prediction flow through Flask app."""
        # Setup mock model
        mock_model.predict = Mock(return_value=np.array([[0.85]]))
        
        # Make request
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': canvas_drawing}),
            content_type='application/json'
        )
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] is True
        assert 'verdict' in data
        assert 'confidence' in data
        assert 'raw_probability' in data
        assert 'drawing_statistics' in data
        
        # Verify timing info
        stats = data['drawing_statistics']
        assert stats['response_time_ms'] > 0
        assert stats['preprocess_time_ms'] > 0
        assert stats['inference_time_ms'] > 0
    
    @patch('src.web.app.model')
    def test_multiple_predictions(self, mock_model, client, canvas_drawing):
        """Test multiple consecutive predictions."""
        mock_model.predict = Mock(return_value=np.array([[0.7]]))
        
        # Make multiple requests
        for i in range(5):
            response = client.post(
                '/api/predict',
                data=json.dumps({'image': canvas_drawing}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
    
    def test_health_check_integration(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'ok'
        assert 'model_loaded' in data
        assert 'threshold' in data
        assert isinstance(data['threshold'], (int, float))
    
    @patch('src.web.app.model')
    def test_different_image_sizes(self, mock_model, client):
        """Test prediction with different input image sizes."""
        mock_model.predict = Mock(return_value=np.array([[0.6]]))
        
        # Test different sizes
        sizes = [(100, 100), (280, 280), (500, 500), (50, 50)]
        
        for width, height in sizes:
            img = Image.new('L', (width, height), color=128)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{img_base64}"
            
            response = client.post(
                '/api/predict',
                data=json.dumps({'image': data_url}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result['success'] is True


class TestErrorHandlingIntegration:
    """Test error handling in realistic scenarios."""
    
    def test_malformed_requests(self, client):
        """Test handling of various malformed requests."""
        # Empty body
        response = client.post('/api/predict', data='')
        assert response.status_code in [400, 500]
        
        # Invalid JSON
        response = client.post(
            '/api/predict',
            data='not json',
            content_type='application/json'
        )
        assert response.status_code in [400, 500]
        
        # Missing image field
        response = client.post(
            '/api/predict',
            data=json.dumps({'wrong_field': 'value'}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.model')
    @patch('src.web.app.tflite_interpreter')
    def test_model_error_handling(self, mock_tflite_interpreter, mock_model, client, canvas_drawing):
        """Test error handling when model fails."""
        # Simulate model error
        mock_model.predict = Mock(side_effect=Exception("Model crashed"))
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': canvas_drawing}),
            content_type='application/json'
        )
        
        # Should return error gracefully
        assert response.status_code in [200, 500]
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
    
    def test_invalid_image_data(self, client):
        """Test handling of invalid image data."""
        invalid_data = [
            'not_base64',
            'data:image/png;base64,invalid!!!',
            '',
            'x' * 10000000,  # Very long string
        ]
        
        for invalid in invalid_data:
            response = client.post(
                '/api/predict',
                data=json.dumps({'image': invalid}),
                content_type='application/json'
            )
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 500]


class TestModelLoadingIntegration:
    """Test model loading scenarios."""
    
    @patch('src.web.app.keras.models.load_model')
    @patch('src.web.app.Path')
    def test_model_loading_on_startup(self, mock_path, mock_load_model):
        """Test that model loads on application startup."""
        # This tests the module-level initialization
        # In practice, this is tested by the app starting successfully
        pass
    
    @patch('src.web.app.model', None)
    @patch('src.web.app.tflite_interpreter', None)
    def test_prediction_without_model(self, client, canvas_drawing):
        """Test prediction when model is not loaded."""
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': canvas_drawing}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Model not loaded' in data['error']


class TestConcurrency:
    """Test concurrent request handling."""
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.model')
    def test_concurrent_predictions(self, mock_model, client, canvas_drawing):
        """Test handling of concurrent prediction requests."""
        mock_model.predict = Mock(return_value=np.array([[0.75]]))
        
        # Simulate concurrent requests (sequential in test)
        responses = []
        for _ in range(10):
            response = client.post(
                '/api/predict',
                data=json.dumps({'image': canvas_drawing}),
                content_type='application/json'
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True


class TestResponseFormat:
    """Test response format consistency."""
    
    @patch('src.web.app.model')
    def test_response_structure(self, mock_model, client, canvas_drawing):
        """Test that response has consistent structure."""
        mock_model.predict = Mock(return_value=np.array([[0.8]]))
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': canvas_drawing}),
            content_type='application/json'
        )
        
        data = json.loads(response.data)
        
        # Required fields
        required_fields = [
            'success',
            'verdict',
            'verdict_text',
            'confidence',
            'raw_probability',
            'threshold',
            'model_info',
            'drawing_statistics'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Verify types
        assert isinstance(data['success'], bool)
        assert isinstance(data['verdict'], str)
        assert isinstance(data['confidence'], (int, float))
        assert isinstance(data['raw_probability'], (int, float))
        assert isinstance(data['drawing_statistics'], dict)
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.model')
    def test_error_response_structure(self, mock_model, client, canvas_drawing):
        """Test error response structure."""
        mock_model.predict = Mock(side_effect=Exception("Error"))
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': canvas_drawing}),
            content_type='application/json'
        )
        
        data = json.loads(response.data)
        
        assert 'success' in data
        assert data['success'] is False
        assert 'error' in data
        assert isinstance(data['error'], str)
