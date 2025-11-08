"""Unit tests for data augmentation and preprocessing."""

import pytest
import numpy as np
from collections.abc import Iterator
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from src.data.augmentation import (
    normalize_image,
    normalize_batch,
    get_augmentation_generator,
    prepare_test_data
)


class TestNormalization:
    """Test normalization functions."""
    
    def test_normalize_image_basic(self):
        """Test basic image normalization."""
        # Create test image with known values
        img = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        
        normalized = normalize_image(img)
        
        # Should be in [0, 1] range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.shape == img.shape
    
    def test_normalize_image_uniform(self):
        """Test normalization with uniform image (no variation)."""
        # Uniform image should remain unchanged
        img = np.ones((28, 28), dtype=np.float32) * 0.5
        
        normalized = normalize_image(img)
        
        # Should return original for low std
        assert np.allclose(normalized, img)
    
    def test_normalize_image_3d(self):
        """Test normalization with 3D image (H, W, C)."""
        img = np.random.rand(28, 28, 1).astype(np.float32)
        
        normalized = normalize_image(img)
        
        assert normalized.shape == img.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_image_removes_brightness_bias(self):
        """Test that normalization removes brightness differences."""
        # Create two images with same pattern but different brightness
        pattern = np.random.rand(28, 28).astype(np.float32)
        bright_img = pattern * 0.8 + 0.2  # Bright version
        dark_img = pattern * 0.3  # Dark version
        
        bright_norm = normalize_image(bright_img)
        dark_norm = normalize_image(dark_img)
        
        # After normalization, they should be similar
        # (not identical due to rescaling, but correlation should be high)
        correlation = np.corrcoef(bright_norm.flatten(), dark_norm.flatten())[0, 1]
        assert correlation > 0.95
    
    def test_normalize_batch_shape(self):
        """Test batch normalization preserves shape."""
        batch = np.random.rand(32, 28, 28, 1).astype(np.float32)
        
        normalized = normalize_batch(batch)
        
        assert normalized.shape == batch.shape
        assert normalized.dtype == np.float32
    
    def test_normalize_batch_per_image(self):
        """Test that batch normalization is applied per-image."""
        # Create batch with different brightness levels
        batch = np.zeros((4, 28, 28, 1), dtype=np.float32)
        batch[0] = 0.2  # Dark
        batch[1] = 0.5  # Medium
        batch[2] = 0.8  # Bright
        batch[3] = np.random.rand(28, 28, 1)  # Random
        
        normalized = normalize_batch(batch)
        
        # Each image should be normalized independently
        for i in range(4):
            assert normalized[i].min() >= 0.0
            assert normalized[i].max() <= 1.0
    
    def test_normalize_batch_empty(self):
        """Test batch normalization with empty batch."""
        batch = np.zeros((0, 28, 28, 1), dtype=np.float32)
        
        normalized = normalize_batch(batch)
        
        assert normalized.shape == batch.shape


class TestAugmentation:
    """Test augmentation generator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_get_augmentation_generator_returns_tuple(self, sample_data):
        """Test that generator returns expected tuple."""
        X, y = sample_data
        
        result = get_augmentation_generator(X, y, batch_size=32)
        
        assert len(result) == 3
        X_norm, generator, augmentation = result
        assert isinstance(generator, Iterator)  # Check it's an iterator
        assert isinstance(augmentation, ImageDataGenerator)
    
    def test_get_augmentation_generator_normalizes_data(self, sample_data):
        """Test that generator normalizes input data."""
        X, y = sample_data
        
        X_norm, generator, _ = get_augmentation_generator(X, y)
        
        # Data should be normalized
        assert X_norm.shape == X.shape
        assert X_norm.dtype == np.float32
    
    def test_get_augmentation_generator_batch_size(self, sample_data):
        """Test generator produces correct batch size."""
        X, y = sample_data
        batch_size = 16
        
        _, generator, _ = get_augmentation_generator(X, y, batch_size=batch_size)
        
        # Get one batch
        X_batch, y_batch = next(generator)
        
        assert X_batch.shape[0] == batch_size
        assert y_batch.shape[0] == batch_size
    
    def test_get_augmentation_generator_augments_data(self, sample_data):
        """Test that generator applies augmentation."""
        X, y = sample_data
        
        X_norm, generator, _ = get_augmentation_generator(
            X, y,
            rotation_range=15,
            width_shift=0.1,
            height_shift=0.1,
            zoom_range=0.15
        )
        
        # Get multiple batches and check they're different (augmented)
        batch1, _ = next(generator)
        batch2, _ = next(generator)
        
        # Batches should be different due to augmentation
        assert not np.allclose(batch1, batch2)
    
    def test_get_augmentation_generator_custom_params(self, sample_data):
        """Test generator with custom augmentation parameters."""
        X, y = sample_data
        
        X_norm, generator, augmentation = get_augmentation_generator(
            X, y,
            batch_size=8,
            rotation_range=30,
            width_shift=0.2,
            height_shift=0.2,
            zoom_range=0.25
        )
        
        # Verify augmentation parameters
        assert augmentation.rotation_range == 30
        assert augmentation.width_shift_range == 0.2
        assert augmentation.height_shift_range == 0.2
        assert augmentation.zoom_range == [0.75, 1.25]
    
    def test_prepare_test_data(self):
        """Test test data preparation."""
        X_test = np.random.rand(50, 28, 28, 1).astype(np.float32)
        
        X_test_prepared = prepare_test_data(X_test)
        
        assert X_test_prepared.shape == X_test.shape
        assert X_test_prepared.dtype == np.float32
        # Should be normalized
        assert X_test_prepared.min() >= 0.0
        assert X_test_prepared.max() <= 1.0
    
    def test_prepare_test_data_consistency(self):
        """Test that test data preparation is deterministic."""
        X_test = np.random.rand(50, 28, 28, 1).astype(np.float32)
        
        prepared1 = prepare_test_data(X_test)
        prepared2 = prepare_test_data(X_test)
        
        # Should be identical (no randomness in test preparation)
        assert np.allclose(prepared1, prepared2)
