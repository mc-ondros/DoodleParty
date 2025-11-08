"""Integration tests for end-to-end workflow."""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch
import tensorflow as tf
from tensorflow import keras

from src.data.loaders import QuickDrawDataset
from src.data.augmentation import normalize_batch, get_augmentation_generator
from src.core.models import build_custom_cnn


class TestDataPipelineIntegration:
    """Test complete data pipeline from download to preprocessing."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        raw_dir = tempfile.mkdtemp()
        processed_dir = tempfile.mkdtemp()
        yield Path(raw_dir), Path(processed_dir)
    
    def test_full_data_pipeline(self, temp_dirs):
        """Test complete data pipeline: load -> preprocess -> augment."""
        raw_dir, processed_dir = temp_dirs
        
        # Step 1: Create mock data
        dataset = QuickDrawDataset(data_dir=str(raw_dir))
        test_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        np.save(raw_dir / "airplane.npy", test_data)
        
        # Step 2: Prepare dataset
        (X_train, y_train), (X_test, y_test), mapping = dataset.prepare_dataset(
            classes=["airplane"],
            output_dir=str(processed_dir),
            max_samples_per_class=100,
            test_split=0.2
        )
        
        # Step 3: Apply augmentation
        X_train_norm, generator, _ = get_augmentation_generator(X_train, y_train, batch_size=32)
        
        # Verify pipeline output
        assert X_train_norm.shape[0] > 0
        assert X_train_norm.dtype == np.float32
        assert X_train_norm.min() >= 0.0
        assert X_train_norm.max() <= 1.0
        
        # Verify generator works
        X_batch, y_batch = next(generator)
        assert X_batch.shape == (32, 28, 28, 1)
        assert y_batch.shape == (32,)


class TestModelTrainingIntegration:
    """Test model training workflow."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        X_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)
        return X_train, y_train, X_val, y_val
    
    def test_model_compile_and_train(self, sample_training_data):
        """Test that model can be compiled and trained."""
        X_train, y_train, X_val, y_val = sample_training_data
        
        # Build model
        model = build_custom_cnn()
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for 1 epoch
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Verify training completed
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_save_and_load(self, sample_training_data):
        """Test model saving and loading."""
        X_train, y_train, X_val, y_val = sample_training_data
        
        # Build and train model
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Save model
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        model.save(temp_file.name)
        
        # Load model (use keras from top-level import)
        import keras as standalone_keras
        loaded_model = standalone_keras.models.load_model(temp_file.name)
        
        # Verify loaded model works
        predictions = loaded_model.predict(X_val, verbose=0)
        assert predictions.shape == (len(X_val), 1)
        
        # Cleanup
        Path(temp_file.name).unlink()


class TestInferencePipelineIntegration:
    """Test complete inference pipeline."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a simple model."""
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train on dummy data
        X = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=1, verbose=0)
        
        return model
    
    def test_preprocessing_to_prediction(self, trained_model):
        """Test complete flow from raw image to prediction."""
        # Create raw image
        raw_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        
        # Preprocess
        normalized = raw_image.astype(np.float32) / 255.0
        normalized = normalize_batch(normalized.reshape(1, 28, 28, 1))
        
        # Predict
        prediction = trained_model.predict(normalized, verbose=0)
        
        # Verify output
        assert prediction.shape == (1, 1)
        assert 0.0 <= prediction[0][0] <= 1.0
    
    def test_batch_inference(self, trained_model):
        """Test batch inference on multiple images."""
        # Create batch of images
        batch_size = 10
        images = np.random.rand(batch_size, 28, 28, 1).astype(np.float32)
        
        # Normalize
        images_norm = normalize_batch(images)
        
        # Predict
        predictions = trained_model.predict(images_norm, verbose=0)
        
        # Verify
        assert predictions.shape == (batch_size, 1)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)


class TestDataAugmentationIntegration:
    """Test data augmentation in training context."""
    
    def test_augmentation_increases_diversity(self):
        """Test that augmentation creates diverse samples."""
        # Create simple pattern
        base_image = np.zeros((28, 28, 1), dtype=np.float32)
        base_image[10:18, 10:18, 0] = 1.0  # White square
        
        X = np.repeat(base_image[np.newaxis, :, :, :], 100, axis=0)
        y = np.ones(100)
        
        # Apply augmentation
        _, generator, _ = get_augmentation_generator(
            X, y,
            batch_size=10,
            rotation_range=30,
            width_shift=0.2,
            height_shift=0.2,
            zoom_range=0.2
        )
        
        # Get multiple batches
        batch1, _ = next(generator)
        batch2, _ = next(generator)
        
        # Batches should be different due to augmentation
        assert not np.allclose(batch1, batch2)
        
        # Individual images in batch should be different
        assert not np.allclose(batch1[0], batch1[1])


class TestModelEvaluationIntegration:
    """Test model evaluation workflow."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        model = build_custom_cnn()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        # Train briefly
        X_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Test data
        X_test = np.random.rand(50, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 50)
        
        return model, X_test, y_test
    
    def test_model_evaluation(self, model_and_data):
        """Test model evaluation returns expected metrics."""
        model, X_test, y_test = model_and_data
        
        # Evaluate
        results = model.evaluate(X_test, y_test, verbose=0)
        
        # Should return [loss, accuracy, auc]
        assert len(results) == 3
        assert all(isinstance(r, (float, np.floating)) for r in results)
    
    def test_prediction_and_thresholding(self, model_and_data):
        """Test prediction with threshold application."""
        model, X_test, y_test = model_and_data
        
        # Get probabilities
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        
        # Apply threshold
        threshold = 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Verify
        assert y_pred.shape == y_test.shape
        assert set(y_pred) <= {0, 1}
        assert len(y_pred_proba) == len(y_test)


class TestFullWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_complete_workflow(self):
        """Test full workflow: data -> train -> evaluate -> predict."""
        # Step 1: Create data
        X_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 20)
        
        # Step 2: Normalize
        X_train_norm = normalize_batch(X_train)
        X_test_norm = normalize_batch(X_test)
        
        # Step 3: Build model
        model = build_custom_cnn()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Step 4: Train
        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_test_norm, y_test),
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        # Step 5: Evaluate
        loss, accuracy = model.evaluate(X_test_norm, y_test, verbose=0)
        
        # Step 6: Predict
        predictions = model.predict(X_test_norm, verbose=0)
        
        # Verify complete workflow
        assert len(history.history['loss']) == 2
        assert isinstance(loss, (float, np.floating))
        assert isinstance(accuracy, (float, np.floating))
        assert predictions.shape == (len(X_test), 1)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)
    
    def test_workflow_with_augmentation(self):
        """Test workflow with data augmentation."""
        # Create data
        X_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)
        
        # Setup augmentation
        X_train_norm, train_generator, _ = get_augmentation_generator(
            X_train, y_train, batch_size=32
        )
        X_val_norm = normalize_batch(X_val)
        
        # Build and train model
        model = build_custom_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with generator
        history = model.fit(
            train_generator,
            steps_per_epoch=3,
            validation_data=(X_val_norm, y_val),
            epochs=1,
            verbose=0
        )
        
        # Verify training completed
        assert 'loss' in history.history
        assert 'val_loss' in history.history
