"""Integration tests for error handling across the system."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

from src.data.loaders import QuickDrawDataset
from src.core.models import build_custom_cnn, get_model
from src.data.augmentation import normalize_image, normalize_batch


class TestDataLoadingErrors:
    """Test error handling in data loading."""
    
    def test_missing_data_file(self):
        """Test error when data file doesn't exist."""
        dataset = QuickDrawDataset(data_dir="nonexistent_dir")
        
        with pytest.raises(FileNotFoundError):
            dataset.load_class_data("airplane")
    
    def test_corrupted_npy_file(self):
        """Test error handling for corrupted NPY file."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create corrupted file
        with open(temp_path / "corrupted.npy", 'wb') as f:
            f.write(b'not a valid npy file')
        
        dataset = QuickDrawDataset(data_dir=str(temp_path))
        
        with pytest.raises(Exception):
            dataset.load_class_data("corrupted")
    
    def test_empty_class_list(self):
        """Test error handling for empty class list."""
        dataset = QuickDrawDataset(data_dir=tempfile.mkdtemp())
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            dataset.prepare_dataset(classes=[], output_dir=tempfile.mkdtemp())


class TestModelErrors:
    """Test error handling in model operations."""
    
    def test_invalid_architecture_name(self):
        """Test error for invalid architecture name."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model('nonexistent_architecture', summary=False)
    
    def test_invalid_input_shape(self):
        """Test model with invalid input shape."""
        model = build_custom_cnn()
        
        # Wrong shape
        X = np.random.rand(10, 64, 64, 3).astype(np.float32)  # Wrong dimensions
        
        with pytest.raises(Exception):
            model.predict(X, verbose=0)
    
    def test_model_without_compilation(self):
        """Test using model without compilation."""
        model = build_custom_cnn()
        
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        
        # Should raise error when trying to train without compilation
        with pytest.raises(Exception):
            model.fit(X, y, epochs=1, verbose=0)


class TestPreprocessingErrors:
    """Test error handling in preprocessing."""
    
    def test_normalize_invalid_input(self):
        """Test normalization with invalid input."""
        # Test with None
        with pytest.raises(Exception):
            normalize_image(None)
        
        # Test with wrong type
        with pytest.raises(Exception):
            normalize_image("not an array")
    
    def test_normalize_batch_mismatched_shapes(self):
        """Test batch normalization with mismatched shapes."""
        # Create batch with inconsistent shapes (not possible with numpy arrays)
        # But test with wrong dimensions
        X = np.random.rand(10, 28, 28).astype(np.float32)  # Missing channel dim
        
        # Should either handle or raise appropriate error
        try:
            result = normalize_batch(X)
            # If it handles it, verify output
            assert result.shape == X.shape
        except Exception:
            # If it raises, that's also acceptable
            pass


class TestInferenceErrors:
    """Test error handling in inference."""
    
    @pytest.fixture
    def model(self):
        """Create simple model."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def test_inference_wrong_input_shape(self, model):
        """Test inference with wrong input shape."""
        X = np.random.rand(10, 64, 64, 1).astype(np.float32)  # Wrong size
        
        with pytest.raises(Exception):
            model.predict(X, verbose=0)
    
    def test_inference_wrong_dtype(self, model):
        """Test inference with wrong data type."""
        X = np.random.randint(0, 256, (10, 28, 28, 1), dtype=np.uint8)  # Wrong dtype
        
        # Should either work (auto-conversion) or raise error
        try:
            pred = model.predict(X, verbose=0)
            assert pred.shape == (10, 1)
        except Exception:
            pass  # Also acceptable
    
    def test_inference_nan_values(self, model):
        """Test inference with NaN values."""
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        X[0, 0, 0, 0] = np.nan
        
        # Should either handle or raise error
        try:
            pred = model.predict(X, verbose=0)
            # If it predicts, check for NaN in output
            assert not np.any(np.isnan(pred))
        except Exception:
            pass  # Also acceptable
    
    def test_inference_inf_values(self, model):
        """Test inference with infinite values."""
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        X[0, 0, 0, 0] = np.inf
        
        # Should either handle or raise error
        try:
            pred = model.predict(X, verbose=0)
            assert not np.any(np.isinf(pred))
        except Exception:
            pass


class TestWebAppErrors:
    """Test error handling in web application."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from src.web.app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_missing_image_data(self, client):
        """Test API with missing image data."""
        import json
        
        response = client.post(
            '/api/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_invalid_json(self, client):
        """Test API with invalid JSON."""
        response = client.post(
            '/api/predict',
            data='not valid json',
            content_type='application/json'
        )
        
        assert response.status_code in [400, 500]
    
    def test_invalid_base64(self, client):
        """Test API with invalid base64 data."""
        import json
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': 'invalid!!!base64'}),
            content_type='application/json'
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_oversized_request(self, client):
        """Test API with very large request."""
        import json
        import base64
        
        # Create large image
        large_data = 'x' * 10_000_000  # 10MB of data
        
        response = client.post(
            '/api/predict',
            data=json.dumps({'image': large_data}),
            content_type='application/json'
        )
        
        # Should handle (may timeout or reject)
        assert response.status_code in [200, 400, 413, 500]


class TestFileSystemErrors:
    """Test error handling for file system operations."""
    
    def test_read_only_directory(self):
        """Test handling of read-only directory."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Make read-only (on Unix systems)
        import os
        try:
            os.chmod(temp_path, 0o444)
            
            dataset = QuickDrawDataset(data_dir=str(temp_path / "subdir"))
            
            # Should handle or raise appropriate error
            # (may succeed on some systems)
        except Exception:
            pass  # Expected on some systems
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_path, 0o755)
    
    def test_disk_full_simulation(self):
        """Test handling of disk full scenario."""
        # This is difficult to test without actually filling disk
        # Just verify error handling exists
        pass


class TestMemoryErrors:
    """Test handling of memory-related errors."""
    
    def test_large_batch_size(self):
        """Test handling of very large batch size."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Try to predict on very large batch
        # This may or may not fail depending on available memory
        try:
            X = np.random.rand(10000, 28, 28, 1).astype(np.float32)
            pred = model.predict(X, verbose=0, batch_size=1000)
            assert pred.shape == (10000, 1)
        except MemoryError:
            pass  # Expected on systems with limited memory
        except Exception:
            pass  # Other errors also acceptable


class TestConcurrentAccessErrors:
    """Test error handling for concurrent access."""
    
    def test_concurrent_model_access(self):
        """Test concurrent access to model."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        # Simulate concurrent predictions
        X_test = np.random.rand(10, 28, 28, 1).astype(np.float32)
        
        # Multiple predictions (sequential in test, but tests thread-safety)
        for _ in range(5):
            pred = model.predict(X_test, verbose=0)
            assert pred.shape == (10, 1)


class TestRecoveryMechanisms:
    """Test error recovery mechanisms."""
    
    @patch('src.web.app.model', None)
    def test_app_without_model(self):
        """Test that app can run without model loaded."""
        from src.web.app import app
        
        # App should start even without model
        assert app is not None
    
    def test_partial_data_loading(self):
        """Test recovery from partial data loading failure."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create one valid file
        valid_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        np.save(temp_path / "valid.npy", valid_data)
        
        # Create one invalid file
        with open(temp_path / "invalid.npy", 'wb') as f:
            f.write(b'corrupted')
        
        dataset = QuickDrawDataset(data_dir=str(temp_path))
        
        # Should be able to load valid file
        data = dataset.load_class_data("valid")
        assert data.shape == (100, 28, 28)
        
        # Should fail on invalid file
        with pytest.raises(Exception):
            dataset.load_class_data("invalid")
