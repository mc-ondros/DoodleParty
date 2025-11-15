"""Integration tests for model inference pipeline."""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch
try:
    import keras
except ImportError:
    from tensorflow import keras

from src.core.models import build_custom_cnn
from src.core.inference import predict_image, evaluate_model, predict_batch
from src.data.augmentation import normalize_batch


class TestInferenceWithRealModel:
    """Test inference with actual trained model."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and test data."""
        # Build and train model
        model = build_custom_cnn()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Train on synthetic data
        X_train = np.random.rand(200, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 200)
        X_test = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 50)
        
        model.fit(X_train, y_train, epochs=2, verbose=0)
        
        # Save model and data
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        model_path = temp_path / "model.h5"
        model.save(model_path)
        
        data_dir = temp_path / "data"
        data_dir.mkdir()
        np.save(data_dir / "X_test.npy", X_test)
        np.save(data_dir / "y_test.npy", y_test)
        
        class_mapping = {'negative': 0, 'positive': 1}
        with open(data_dir / "class_mapping.pkl", 'wb') as f:
            pickle.dump(class_mapping, f)
        
        return model_path, data_dir, X_test, y_test
    
    def test_single_image_inference(self, trained_model_and_data):
        """Test inference on single image."""
        model_path, data_dir, X_test, y_test = trained_model_and_data
        
        # Create test image
        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img = Image.fromarray((X_test[0, :, :, 0] * 255).astype(np.uint8), 'L')
        img.save(temp_img.name)
        
        # Load model and predict
        model = keras.models.load_model(model_path)
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        class_name, confidence, probability = predict_image(
            model, idx_to_class, temp_img.name
        )
        
        # Verify prediction
        assert class_name in ['positive (in-distribution)', 'negative (out-of-distribution)']
        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= probability <= 1.0
        
        # Cleanup
        Path(temp_img.name).unlink()
    
    def test_batch_inference(self, trained_model_and_data):
        """Test inference on batch of images."""
        model_path, data_dir, X_test, y_test = trained_model_and_data
        
        # Create directory with test images
        img_dir = tempfile.mkdtemp()
        img_path = Path(img_dir)
        
        for i in range(5):
            img = Image.fromarray((X_test[i, :, :, 0] * 255).astype(np.uint8), 'L')
            img.save(img_path / f"test_{i}.png")
        
        # Batch predict
        results = predict_batch(str(model_path), str(img_path), str(data_dir))
        
        # Verify results
        assert len(results) == 5
        for result in results:
            assert 'file' in result
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    @patch('src.core.inference.plt.savefig')
    def test_model_evaluation(self, mock_savefig, trained_model_and_data):
        """Test full model evaluation."""
        model_path, data_dir, X_test, y_test = trained_model_and_data
        
        # Run evaluation
        evaluate_model(str(model_path), str(data_dir))
        
        # Verify confusion matrix was saved
        assert mock_savefig.called


class TestInferenceConsistency:
    """Test inference consistency across different scenarios."""
    
    @pytest.fixture
    def model(self):
        """Create simple trained model."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def test_same_input_same_output(self, model):
        """Test that same input produces same output."""
        X = np.random.rand(1, 28, 28, 1).astype(np.float32)
        
        pred1 = model.predict(X, verbose=0)
        pred2 = model.predict(X, verbose=0)
        
        assert np.allclose(pred1, pred2)
    
    def test_batch_vs_single_inference(self, model):
        """Test that batch and single inference produce same results."""
        X = np.random.rand(5, 28, 28, 1).astype(np.float32)
        
        # Batch inference
        batch_preds = model.predict(X, verbose=0)
        
        # Single inference
        single_preds = []
        for i in range(5):
            pred = model.predict(X[i:i+1], verbose=0)
            single_preds.append(pred[0])
        single_preds = np.array(single_preds)
        
        # Should be very close
        assert np.allclose(batch_preds, single_preds, rtol=1e-5)
    
    def test_preprocessing_consistency(self, model):
        """Test that preprocessing is consistent."""
        # Create image
        img_array = np.random.rand(28, 28).astype(np.float32)
        
        # Apply preprocessing twice
        norm1 = normalize_batch(img_array.reshape(1, 28, 28, 1))
        norm2 = normalize_batch(img_array.reshape(1, 28, 28, 1))
        
        # Should be identical
        assert np.allclose(norm1, norm2)


class TestInferencePerformance:
    """Test inference performance characteristics."""
    
    @pytest.fixture
    def model(self):
        """Create model for performance testing."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def test_single_inference_speed(self, model):
        """Test that single inference completes quickly."""
        import time
        
        X = np.random.rand(1, 28, 28, 1).astype(np.float32)
        
        start = time.time()
        model.predict(X, verbose=0)
        duration = time.time() - start
        
        # Should complete in reasonable time (< 1 second for CPU)
        assert duration < 1.0
    
    def test_batch_inference_efficiency(self, model):
        """Test that batch inference is more efficient than individual."""
        import time
        
        batch_size = 10
        X = np.random.rand(batch_size, 28, 28, 1).astype(np.float32)
        
        # Batch inference
        start = time.time()
        model.predict(X, verbose=0)
        batch_time = time.time() - start
        
        # Individual inference
        start = time.time()
        for i in range(batch_size):
            model.predict(X[i:i+1], verbose=0)
        individual_time = time.time() - start
        
        # Batch should be faster (or at least not much slower)
        # Allow some tolerance for overhead
        assert batch_time < individual_time * 1.5


class TestInferenceEdgeCases:
    """Test edge cases in inference."""
    
    @pytest.fixture
    def model(self):
        """Create model for edge case testing."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def test_all_black_image(self, model):
        """Test inference on all-black image."""
        X = np.zeros((1, 28, 28, 1), dtype=np.float32)
        
        pred = model.predict(X, verbose=0)
        
        assert pred.shape == (1, 1)
        assert 0.0 <= pred[0][0] <= 1.0
    
    def test_all_white_image(self, model):
        """Test inference on all-white image."""
        X = np.ones((1, 28, 28, 1), dtype=np.float32)
        
        pred = model.predict(X, verbose=0)
        
        assert pred.shape == (1, 1)
        assert 0.0 <= pred[0][0] <= 1.0
    
    def test_random_noise_image(self, model):
        """Test inference on random noise."""
        X = np.random.rand(1, 28, 28, 1).astype(np.float32)
        
        pred = model.predict(X, verbose=0)
        
        assert pred.shape == (1, 1)
        assert 0.0 <= pred[0][0] <= 1.0
    
    def test_extreme_values(self, model):
        """Test inference with extreme pixel values."""
        # Values at boundaries
        X = np.random.choice([0.0, 1.0], size=(1, 28, 28, 1)).astype(np.float32)
        
        pred = model.predict(X, verbose=0)
        
        assert pred.shape == (1, 1)
        assert 0.0 <= pred[0][0] <= 1.0
    
    def test_single_pixel_drawing(self, model):
        """Test inference on image with single pixel set."""
        X = np.zeros((1, 28, 28, 1), dtype=np.float32)
        X[0, 14, 14, 0] = 1.0  # Single pixel in center
        
        pred = model.predict(X, verbose=0)
        
        assert pred.shape == (1, 1)
        assert 0.0 <= pred[0][0] <= 1.0


class TestThresholdBehavior:
    """Test threshold application in inference."""
    
    @pytest.fixture
    def model_and_image(self):
        """Create model and test image."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        # Create test image
        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img = Image.new('L', (28, 28), color=128)
        img.save(temp_img.name)
        
        return model, temp_img.name
    
    def test_different_thresholds(self, model_and_image):
        """Test prediction with different thresholds."""
        model, img_path = model_and_image
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        # Get raw probability
        _, _, probability = predict_image(model, idx_to_class, img_path, threshold=0.5)
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            class_name, confidence, prob = predict_image(
                model, idx_to_class, img_path, threshold=threshold
            )
            results.append((threshold, class_name, probability))
        
        # Probability should be same regardless of threshold
        for _, _, prob in results:
            assert np.isclose(prob, probability)
        
        # Cleanup
        Path(img_path).unlink()
    
    def test_threshold_boundary(self, model_and_image):
        """Test behavior at threshold boundary."""
        model, img_path = model_and_image
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        # Get probability
        _, _, probability = predict_image(model, idx_to_class, img_path, threshold=0.5)
        
        # Test with threshold just below and above probability
        if probability > 0.01 and probability < 0.99:
            below_class, _, _ = predict_image(
                model, idx_to_class, img_path, threshold=probability - 0.01
            )
            above_class, _, _ = predict_image(
                model, idx_to_class, img_path, threshold=probability + 0.01
            )
            
            # Should give different classifications
            assert 'positive' in below_class
            assert 'negative' in above_class
        
        # Cleanup
        Path(img_path).unlink()
