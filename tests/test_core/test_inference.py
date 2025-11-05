"""Unit tests for model inference."""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.core.inference import (
    load_model_and_mapping,
    predict_image,
    evaluate_model,
    predict_batch
)


class TestLoadModelAndMapping:
    """Test model and mapping loading."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for model and data."""
        model_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        yield Path(model_dir), Path(data_dir)
        # Cleanup handled by tempfile
    
    @patch('src.core.inference.keras.models.load_model')
    def test_load_model_and_mapping_success(self, mock_load_model, temp_dirs):
        """Test successful model and mapping loading."""
        model_dir, data_dir = temp_dirs
        
        # Create mock model
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Create class mapping
        class_mapping = {'negative': 0, 'positive': 1}
        with open(data_dir / "class_mapping.pkl", 'wb') as f:
            pickle.dump(class_mapping, f)
        
        model, idx_to_class = load_model_and_mapping(
            str(model_dir / "model.h5"),
            str(data_dir)
        )
        
        assert model == mock_model
        assert idx_to_class == {0: 'negative', 1: 'positive'}
    
    def test_load_model_and_mapping_missing_file(self, temp_dirs):
        """Test error when class mapping file is missing."""
        model_dir, data_dir = temp_dirs
        
        with patch('src.core.inference.keras.models.load_model'):
            with pytest.raises(FileNotFoundError):
                load_model_and_mapping(
                    str(model_dir / "model.h5"),
                    str(data_dir)
                )


class TestPredictImage:
    """Test single image prediction."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.8]]))
        return model
    
    @pytest.fixture
    def temp_image(self):
        """Create temporary test image."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img = Image.new('L', (28, 28), color=128)
        img.save(temp_file.name)
        yield temp_file.name
        Path(temp_file.name).unlink()
    
    def test_predict_image_positive(self, mock_model, temp_image):
        """Test prediction for positive class."""
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        class_name, confidence, probability = predict_image(
            mock_model,
            idx_to_class,
            temp_image,
            threshold=0.5
        )
        
        assert 'positive' in class_name
        assert confidence > 0.5
        assert 0.0 <= probability <= 1.0
    
    def test_predict_image_negative(self, temp_image):
        """Test prediction for negative class."""
        # Mock model with low probability
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.3]]))
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        class_name, confidence, probability = predict_image(
            model,
            idx_to_class,
            temp_image,
            threshold=0.5
        )
        
        assert 'negative' in class_name
        assert confidence > 0.5  # Confidence in negative class
        assert probability == 0.3
    
    def test_predict_image_custom_threshold(self, mock_model, temp_image):
        """Test prediction with custom threshold."""
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        # With high threshold, 0.8 probability should still be positive
        class_name, confidence, probability = predict_image(
            mock_model,
            idx_to_class,
            temp_image,
            threshold=0.7
        )
        
        assert 'positive' in class_name
    
    def test_predict_image_preprocessing(self, mock_model, temp_image):
        """Test that image is preprocessed correctly."""
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        predict_image(mock_model, idx_to_class, temp_image)
        
        # Check model was called with correct shape
        call_args = mock_model.predict.call_args
        input_array = call_args[0][0]
        
        assert input_array.shape == (1, 28, 28, 1)
        assert input_array.dtype == np.float32
        assert input_array.min() >= 0.0
        assert input_array.max() <= 1.0
    
    def test_predict_image_file_not_found(self, mock_model):
        """Test error handling for missing image file."""
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        with pytest.raises(FileNotFoundError):
            predict_image(mock_model, idx_to_class, "nonexistent.png")


class TestEvaluateModel:
    """Test model evaluation."""
    
    @pytest.fixture
    def temp_model_and_data(self):
        """Create temporary model and test data."""
        model_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        
        # Create test data
        X_test = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 100)
        
        np.save(Path(data_dir) / "X_test.npy", X_test)
        np.save(Path(data_dir) / "y_test.npy", y_test)
        
        # Create class mapping
        class_mapping = {'negative': 0, 'positive': 1}
        with open(Path(data_dir) / "class_mapping.pkl", 'wb') as f:
            pickle.dump(class_mapping, f)
        
        yield Path(model_dir), Path(data_dir), X_test, y_test
    
    @patch('src.core.inference.keras.models.load_model')
    @patch('src.core.inference.plt.savefig')
    def test_evaluate_model_runs(self, mock_savefig, mock_load_model, temp_model_and_data):
        """Test that evaluation runs without errors."""
        model_dir, data_dir, X_test, y_test = temp_model_and_data
        
        # Create mock model
        mock_model = Mock()
        mock_model.evaluate = Mock(return_value=(0.5, 0.85, 0.90))
        mock_model.predict = Mock(return_value=np.random.rand(len(X_test), 1))
        mock_load_model.return_value = mock_model
        
        # Should not raise error
        evaluate_model(str(model_dir / "model.h5"), str(data_dir))
        
        # Verify model methods were called
        assert mock_model.evaluate.called
        assert mock_model.predict.called
    
    @patch('src.core.inference.keras.models.load_model')
    def test_evaluate_model_missing_data(self, mock_load_model):
        """Test error when test data is missing."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        with pytest.raises(FileNotFoundError):
            evaluate_model("model.h5", "nonexistent_dir")


class TestPredictBatch:
    """Test batch prediction."""
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create directory with test images."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create test images
        for i in range(3):
            img = Image.new('L', (28, 28), color=128)
            img.save(temp_path / f"test_{i}.png")
        
        yield temp_path
    
    @pytest.fixture
    def mock_model_and_mapping(self):
        """Create mock model and mapping."""
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.7]]))
        idx_to_class = {0: 'negative', 1: 'positive'}
        return model, idx_to_class
    
    @patch('src.core.inference.load_model_and_mapping')
    def test_predict_batch_success(self, mock_load, temp_image_dir, mock_model_and_mapping):
        """Test batch prediction on directory."""
        mock_load.return_value = mock_model_and_mapping
        
        results = predict_batch("model.h5", str(temp_image_dir))
        
        assert len(results) == 3
        for result in results:
            assert 'file' in result
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    @patch('src.core.inference.load_model_and_mapping')
    def test_predict_batch_empty_directory(self, mock_load, mock_model_and_mapping):
        """Test batch prediction on empty directory."""
        mock_load.return_value = mock_model_and_mapping
        temp_dir = tempfile.mkdtemp()
        
        results = predict_batch("model.h5", temp_dir)
        
        assert len(results) == 0
    
    @patch('src.core.inference.load_model_and_mapping')
    @patch('src.core.inference.predict_image')
    def test_predict_batch_handles_errors(self, mock_predict, mock_load, 
                                         temp_image_dir, mock_model_and_mapping):
        """Test that batch prediction handles individual errors."""
        mock_load.return_value = mock_model_and_mapping
        mock_predict.side_effect = Exception("Processing error")
        
        # Should not raise, but handle errors gracefully
        results = predict_batch("model.h5", str(temp_image_dir))
        
        # Results should be empty due to errors
        assert len(results) == 0
