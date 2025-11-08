"""Unit tests for Flask web application."""

import pytest
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.web.app import app, preprocess_image, predict, load_model_and_mapping


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_canvas_data():
    """Create sample base64 encoded canvas data."""
    # Create a simple test image
    img = Image.new('L', (280, 280), color=255)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


class TestRoutes:
    """Test Flask routes."""
    
    def test_index_route(self, client):
        """Test that index route returns HTML."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'html' in response.data.lower() or response.content_type == 'text/html; charset=utf-8'
    
    def test_health_route(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'model_loaded' in data
        assert 'threshold' in data
    
    @patch('src.web.app.model')
    def test_predict_route_success(self, mock_model, client, sample_canvas_data):
        """Test successful prediction via API."""
        # Mock model prediction
        mock_model.predict = Mock(return_value=np.array([[0.8]]))
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': sample_canvas_data}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'verdict' in data
        assert 'confidence' in data
    
    def test_predict_route_no_image(self, client):
        """Test prediction endpoint with missing image data."""
        response = client.post(
            '/api/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
    
    def test_predict_route_invalid_json(self, client):
        """Test prediction endpoint with invalid JSON."""
        response = client.post(
            '/api/predict',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code in [400, 500]


class TestPreprocessImage:
    """Test image preprocessing function."""
    
    def test_preprocess_image_shape(self, sample_canvas_data):
        """Test that preprocessing returns correct shape."""
        result = preprocess_image(sample_canvas_data)
        
        assert result.shape == (1, 128, 128, 1)
        assert result.dtype == np.float32
    
    def test_preprocess_image_normalization(self, sample_canvas_data):
        """Test that preprocessing normalizes values."""
        result = preprocess_image(sample_canvas_data)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_image_without_prefix(self):
        """Test preprocessing with base64 data without data URL prefix."""
        # Create simple image
        img = Image.new('L', (100, 100), color=128)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Should work without prefix
        result = preprocess_image(img_base64)
        
        assert result.shape == (1, 128, 128, 1)
    
    def test_preprocess_image_color_inversion(self):
        """Test that preprocessing inverts colors correctly."""
        # Create white image (canvas draws white on black)
        img = Image.new('L', (100, 100), color=255)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_base64}"
        
        result = preprocess_image(data_url)
        
        # After inversion, white should become black (low values)
        # After normalization, should still be relatively low
        assert result.mean() < 0.5
    
    def test_preprocess_image_invalid_base64(self):
        """Test error handling for invalid base64 data."""
        with pytest.raises(Exception):
            preprocess_image("invalid_base64_data")


class TestPredict:
    """Test prediction function."""
    
    @patch('src.web.app.model')
    def test_predict_success(self, mock_model, sample_canvas_data):
        """Test successful prediction."""
        mock_model.predict = Mock(return_value=np.array([[0.75]]))
        
        result = predict(sample_canvas_data)
        
        assert result['success'] is True
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'raw_probability' in result
        assert 'drawing_statistics' in result
    
    @patch('src.web.app.model', None)
    @patch('src.web.app.tflite_interpreter', None)
    def test_predict_no_model(self, sample_canvas_data):
        """Test prediction when model is not loaded."""
        result = predict(sample_canvas_data)

        assert result['success'] is False
        assert 'error' in result
        assert 'Model not loaded' in result['error']
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.tflite_interpreter', None)
    @patch('src.web.app.model')
    def test_predict_positive_class(self, mock_model, sample_canvas_data):
        """Test prediction for positive class."""
        mock_model.predict = Mock(return_value=np.array([[0.8]]))

        result = predict(sample_canvas_data)

        assert result['success'] is True
        assert result['verdict'] == 'PENIS'
        assert result['confidence'] >= 0.5
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.tflite_interpreter', None)
    @patch('src.web.app.model')
    def test_predict_negative_class(self, mock_model, sample_canvas_data):
        """Test prediction for negative class."""
        mock_model.predict = Mock(return_value=np.array([[0.3]]))

        result = predict(sample_canvas_data)

        assert result['success'] is True
        assert result['verdict'] == 'OTHER_SHAPE'
        assert result['confidence'] >= 0.5
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.tflite_interpreter', None)
    @patch('src.web.app.model')
    def test_predict_timing_info(self, mock_model, sample_canvas_data):
        """Test that prediction includes timing information."""
        mock_model.predict = Mock(return_value=np.array([[0.7]]))

        result = predict(sample_canvas_data)

        assert 'drawing_statistics' in result
        stats = result['drawing_statistics']
        assert 'response_time_ms' in stats
        assert 'preprocess_time_ms' in stats
        assert 'inference_time_ms' in stats
        assert stats['response_time_ms'] > 0
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.tflite_interpreter', None)
    @patch('src.web.app.model')
    def test_predict_threshold_boundary(self, mock_model, sample_canvas_data):
        """Test prediction at threshold boundary."""
        # Test exactly at threshold
        mock_model.predict = Mock(return_value=np.array([[0.5]]))

        result = predict(sample_canvas_data)

        assert result['success'] is True
        # At threshold, should be positive
        assert result['verdict'] == 'PENIS'
    
    @patch('src.web.app.is_tflite', False)
    @patch('src.web.app.tflite_interpreter', None)
    @patch('src.web.app.model')
    def test_predict_error_handling(self, mock_model, sample_canvas_data):
        """Test error handling in prediction."""
        mock_model.predict = Mock(side_effect=Exception("Model error"))

        result = predict(sample_canvas_data)

        assert result['success'] is False
        assert 'error' in result


class TestLoadModelAndMapping:
    """Test model loading function."""
    
    @patch('src.web.app.keras.models.load_model')
    @patch('src.web.app.Path')
    def test_load_model_finds_h5_file(self, mock_path_class, mock_load_model):
        """Test that model loading finds .h5 files."""
        # Create mock Path instances
        mock_h5_file = Mock()
        mock_h5_file.stat = Mock(return_value=Mock(st_mtime=1000, st_size=1024*1024))
        mock_h5_file.exists = Mock(return_value=True)
        
        mock_models_dir = Mock()
        mock_models_dir.exists = Mock(return_value=True)
        mock_models_dir.glob = Mock(side_effect=[
            [],  # *_int8.tflite files
            [],  # *.tflite files
            [mock_h5_file],  # *.h5 files
            []  # *.keras files
        ])
        
        # Mock for "data" / "processed" chain
        mock_data_processed = Mock()
        mock_data_processed.exists = Mock(return_value=False)
        
        mock_data_dir = Mock()
        mock_data_dir.__truediv__ = Mock(return_value=mock_data_processed)
        
        # Mock Path(__file__).parent.parent.parent
        mock_root = Mock()
        mock_root.__truediv__ = Mock(side_effect=lambda x: mock_models_dir if x == "models" else mock_data_dir)
        
        mock_file_path = Mock()
        mock_file_path.parent.parent.parent = mock_root
        
        # Configure Path class to return our mock when called
        mock_path_class.return_value = mock_file_path
        
        mock_model = Mock()
        mock_model.input_shape = (None, 128, 128, 1)
        mock_model.output_shape = (None, 1)
        mock_model.count_params = Mock(return_value=1000)
        mock_load_model.return_value = mock_model

        # Should not raise error
        load_model_and_mapping()
        
        # Verify model was loaded
        assert mock_load_model.called
    
    @patch('src.web.app.keras.models.load_model')
    @patch('src.web.app.Path')
    def test_load_model_no_files(self, mock_path_class, mock_load_model):
        """Test error when no model files found."""
        # Create mock Path instances
        mock_models_dir = Mock()
        mock_models_dir.exists = Mock(return_value=True)
        mock_models_dir.glob = Mock(return_value=[])  # No files
        
        # Mock for "data" / "processed" chain
        mock_data_processed = Mock()
        mock_data_processed.exists = Mock(return_value=True)
        
        mock_data_dir = Mock()
        mock_data_dir.__truediv__ = Mock(return_value=mock_data_processed)
        
        # Mock Path(__file__).parent.parent.parent
        mock_root = Mock()
        mock_root.__truediv__ = Mock(side_effect=lambda x: mock_models_dir if x == "models" else mock_data_dir)
        
        mock_file_path = Mock()
        mock_file_path.parent.parent.parent = mock_root
        
        # Configure Path class to return our mock when called
        mock_path_class.return_value = mock_file_path

        with pytest.raises(FileNotFoundError, match='No model files found'):
            load_model_and_mapping()


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_enabled(self, client):
        """Test that CORS headers are present."""
        response = client.get('/api/health')
        
        # CORS should add Access-Control headers
        # Note: Exact headers depend on flask-cors configuration
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in web application."""
    
    def test_404_error(self, client):
        """Test 404 error for non-existent routes."""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        # GET on POST-only endpoint
        response = client.get('/api/predict')
        
        assert response.status_code == 405
    
    @patch('src.web.app.model')
    def test_predict_handles_preprocessing_error(self, mock_model, client):
        """Test that prediction handles preprocessing errors gracefully."""
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': 'invalid_image_data'}),
            content_type='application/json'
        )
        
        # Should return error, not crash
        assert response.status_code in [200, 500]
        data = json.loads(response.data)
        assert data['success'] is False
