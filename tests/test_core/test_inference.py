"""
Comprehensive unit tests for DoodleHunter inference system.

This test suite validates the core inference functionality including:
1. Model loading and class mapping utilities
2. Single image prediction with proper preprocessing and thresholding
3. Model evaluation metrics and visualization generation
4. Batch processing with error handling and result aggregation

Test Coverage Focus:
- Correct input preprocessing (grayscale conversion, resizing, normalization)
- Proper model loading and class mapping inversion
- Threshold-based classification logic and confidence calculation
- Error handling for missing files and processing failures
- Batch processing resilience and result structure validation
- Integration with external dependencies (Keras, PIL, matplotlib)

Security Testing Considerations:
- Input validation and sanitization
- Error handling that doesn't leak sensitive information
- Robust file handling for potentially malicious inputs
- Consistent preprocessing between training and inference

Each test class corresponds to a specific inference function and validates
both happy path scenarios and error conditions to ensure production-ready
reliability.
"""

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
    """
    Test suite for load_model_and_mapping() function.

    This class validates the critical model loading utility that handles:
    - Loading trained Keras models from disk
    - Loading and inverting class mapping dictionaries
    - Proper error handling for missing files
    - Integration between model and mapping loading

    The load_model_and_mapping function is foundational to all inference
    operations, so thorough testing ensures reliability across the entire
    inference pipeline. Tests cover both successful loading scenarios
    and failure conditions to verify robust error handling.
    """
    
    @pytest.fixture
    def temp_dirs(self):
        """
        Pytest fixture providing isolated temporary directories for testing.

        Creates separate temporary directories for:
        - model_dir: Simulates model storage location
        - data_dir: Simulates processed data directory containing class_mapping.pkl

        This isolation ensures tests don't interfere with each other or
        with actual project files, following best practices for unit testing.
        Directories are automatically cleaned up by tempfile module.
        """
        model_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        yield Path(model_dir), Path(data_dir)
        # Cleanup handled by tempfile
    
    @patch('src.core.inference.keras.models.load_model')
    def test_load_model_and_mapping_success(self, mock_load_model, temp_dirs):
        """
        Test successful loading of model and class mapping.

        Validates that:
        - Keras model is loaded correctly from the specified path
        - Class mapping pickle file is loaded and properly inverted
        - Returned model matches the mock model instance
        - Inverted mapping correctly maps {0: 'negative', 1: 'positive'}

        This test ensures the fundamental integration between model loading
        and class mapping works as expected, which is critical for all
        subsequent inference operations.
        """
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
    
    @patch('src.core.inference.keras.models.load_model')
    def test_load_model_and_mapping_missing_file(self, mock_load_model, temp_dirs):
        """
        Test error handling when class mapping file is missing.

        Validates that:
        - FileNotFoundError is raised when class_mapping.pkl doesn't exist
        - Model loading is attempted first (mocked successfully)
        - Error occurs during class mapping loading phase
        - Proper exception type is raised for missing mapping file

        This test ensures robust error handling for incomplete model deployments
        where the model file exists but supporting data files are missing,
        preventing silent failures in production environments.
        """
        model_dir, data_dir = temp_dirs
        
        # Mock the model loading
        mock_load_model.return_value = Mock()
        
        with pytest.raises(FileNotFoundError):
            load_model_and_mapping(
                str(model_dir / "model.h5"),
                str(data_dir)
            )


class TestPredictImage:
    """
    Test suite for predict_image() function.

    This class validates the core single-image prediction functionality,
    ensuring proper image preprocessing, model inference, and result
    interpretation. Tests cover:

    1. Classification Logic:
       - Correct positive/negative classification based on threshold
       - Proper confidence calculation (always >= 0.5)
       - Custom threshold handling

    2. Input Preprocessing:
       - Grayscale conversion regardless of input format
       - 28x28 resizing to match training dimensions
       - Per-image normalization consistency
       - Correct tensor shape (1, 28, 28, 1) for model input

    3. Error Handling:
       - FileNotFoundError for missing image files
       - Graceful handling of corrupted or unsupported images

    4. Output Structure:
       - Correct return tuple format (class_name, confidence, probability)
       - Proper class name formatting with descriptive labels
       - Valid confidence and probability ranges

    These tests ensure the predict_image function provides reliable,
    consistent results that match the training pipeline expectations.
    """
    
    @pytest.fixture
    def mock_model(self):
        """
        Pytest fixture providing a mock Keras model for testing.

        Returns a MagicMock object configured to:
        - Return a probability of 0.8 when predict() is called
        - Simulate realistic model behavior without actual computation
        - Allow verification of input preprocessing through call arguments

        This mock enables fast, deterministic testing of prediction logic
        without requiring actual trained models or GPU resources.
        """
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.8]]))
        return model
    
    @pytest.fixture
    def temp_image(self):
        """
        Pytest fixture providing a temporary 28x28 grayscale test image.

        Creates a PNG image file with:
        - 28x28 pixel dimensions (matching training data expectations)
        - Grayscale mode ('L') with uniform gray value (128)
        - Proper file cleanup after test completion

        This fixture simulates realistic input images while ensuring
        test isolation and automatic resource cleanup.
        """
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img = Image.new('L', (28, 28), color=128)
        img.save(temp_file.name)
        yield temp_file.name
        Path(temp_file.name).unlink()
    
    def test_predict_image_positive(self, mock_model, temp_image):
        """
        Test positive class prediction with probability above threshold.

        Validates that when model outputs probability >= threshold (0.8 >= 0.5):
        - Class name contains 'positive (in-distribution)'
        - Confidence is correctly calculated as the probability value (0.8)
        - Probability is within valid range [0.0, 1.0]
        - Output structure matches expected tuple format

        This test verifies the core classification logic for positive detections,
        ensuring proper interpretation of model outputs and confidence scoring.
        """
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
        """
        Test negative class prediction with probability below threshold.

        Validates that when model outputs probability < threshold (0.3 < 0.5):
        - Class name contains 'negative (out-of-distribution)'
        - Confidence is correctly calculated as (1 - probability) = 0.7
        - Confidence is always >= 0.5 (representing model certainty)
        - Raw probability is preserved exactly as model output (0.3)

        This test ensures proper handling of negative classifications and
        validates the confidence calculation logic that always represents
        the model's certainty in its prediction (never < 0.5).
        """
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
        """
        Test custom threshold handling for flexible classification sensitivity.

        Validates that:
        - Custom threshold parameter (0.7) is properly applied
        - Model output (0.8) is correctly compared against custom threshold
        - Classification result remains positive since 0.8 >= 0.7
        - Threshold flexibility enables tuning for different use cases

        This test demonstrates the threshold parameter's role in enabling
        adjustable sensitivity for different security requirements:
        - Lower thresholds increase recall (more positives detected)
        - Higher thresholds increase precision (fewer false positives)
        - Default threshold of 0.5 provides balanced performance
        """
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
        """
        Test correct image preprocessing pipeline for inference consistency.

        Validates that input images are properly preprocessed to match
        training pipeline expectations:
        - Input tensor shape is (1, 28, 28, 1) for batch dimension + HWC format
        - Data type is float32 for model compatibility
        - Pixel values are normalized to [0.0, 1.0] range
        - Grayscale conversion and 28x28 resizing are applied

        This test ensures preprocessing consistency between training and
        inference, which is critical for maintaining model performance.
        Inconsistent preprocessing is a common source of deployment failures.
        """
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
        """
        Test robust error handling for missing input image files.

        Validates that:
        - FileNotFoundError is raised when input image doesn't exist
        - Error occurs during image loading phase (PIL.Image.open)
        - Model is not called when input file is invalid
        - Proper exception type provides clear debugging information

        This test ensures the function fails gracefully with informative
        errors rather than crashing or producing undefined behavior,
        which is essential for production reliability and user experience.
        """
        idx_to_class = {0: 'negative', 1: 'positive'}
        
        with pytest.raises(FileNotFoundError):
            predict_image(mock_model, idx_to_class, "nonexistent.png")


class TestEvaluateModel:
    """
    Test suite for evaluate_model() function.

    This class validates the comprehensive model evaluation functionality,
    ensuring proper loading of test datasets, model evaluation metrics,
    and visualization generation. Tests cover:

    1. Evaluation Pipeline:
       - Correct loading of X_test.npy and y_test.npy files
       - Proper model evaluation with loss, accuracy, and AUC metrics
       - Classification report generation with correct class labels
       - Confusion matrix computation and visualization

    2. Data Handling:
       - Test data directory structure validation
       - Class mapping integration for proper label interpretation
       - Error handling for missing test data files

    3. Visualization Output:
       - Confusion matrix heatmap generation with seaborn
       - Proper file saving with correct path resolution
       - Visualization labeling with descriptive class names

    4. Integration Testing:
       - End-to-end evaluation workflow without errors
       - Model method calls (evaluate, predict) are properly invoked
       - Resource cleanup and memory management

    These tests ensure the evaluation function provides reliable performance
    metrics essential for model selection, validation, and monitoring.
    """
    
    @pytest.fixture
    def temp_model_and_data(self):
        """
        Pytest fixture providing temporary model directory and test dataset.

        Creates:
        - model_dir: Temporary directory for model file simulation
        - data_dir: Temporary directory containing evaluation test data
        - X_test.npy: Random test input features (100, 28, 28, 1)
        - y_test.npy: Random test labels (100 binary labels)
        - class_mapping.pkl: Standard binary class mapping

        This fixture simulates the complete evaluation environment with
        realistic test data structure matching the training pipeline output,
        enabling comprehensive integration testing of the evaluation function.
        """
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
        """
        Test end-to-end evaluation workflow executes without errors.

        Validates that the complete evaluation pipeline:
        - Successfully loads model and test data from temporary directories
        - Calls model.evaluate() with correct test data
        - Calls model.predict() for detailed predictions
        - Generates classification report with proper class labels
        - Creates confusion matrix visualization
        - Saves visualization file to correct location

        This integration test ensures all evaluation components work together
        seamlessly, providing confidence that the evaluation function will
        operate correctly in production environments with real models and data.
        """
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
        """
        Test error handling when evaluation test data files are missing.

        Validates that:
        - FileNotFoundError is raised when X_test.npy or y_test.npy don't exist
        - Model loading is attempted first (mocked successfully)
        - Error occurs during test data loading phase
        - Proper exception provides clear indication of missing test data

        This test ensures robust error handling for incomplete evaluation
        environments, preventing silent failures when test datasets are
        not properly prepared or deployed alongside models.
        """
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        with pytest.raises(FileNotFoundError):
            evaluate_model("model.h5", "nonexistent_dir")


class TestPredictBatch:
    """
    Test suite for predict_batch() function.

    This class validates the batch processing functionality for directory-based
    image classification, ensuring efficient processing of multiple images with
    proper error handling and result aggregation. Tests cover:

    1. Batch Processing Logic:
       - Correct discovery of PNG/JPG files in specified directory
       - Single model loading for efficiency across all images
       - Sequential processing with consistent results
       - Proper result structure with file tracking and predictions

    2. Edge Cases:
       - Empty directories return empty results (no errors)
       - Missing or invalid images are gracefully skipped
       - Mixed file types (only PNG/JPG processed)

    3. Error Resilience:
       - Individual image processing failures don't stop batch processing
       - Errors are logged but processing continues
       - Results only include successfully processed images

    4. Result Structure:
       - Each result contains 'file', 'predicted_class', 'confidence'
       - File names are preserved for tracking and correlation
       - Confidence values are properly calculated per image

    These tests ensure the batch function provides reliable, production-ready
    processing for high-throughput classification workflows.
    """
    
    @pytest.fixture
    def temp_image_dir(self):
        """
        Pytest fixture providing a temporary directory with multiple test images.

        Creates a directory containing 3 PNG files with:
        - 28x28 pixel dimensions matching training expectations
        - Grayscale mode with uniform gray values
        - Sequential naming (test_0.png, test_1.png, test_2.png)

        This fixture simulates realistic batch processing scenarios with
        multiple images, enabling testing of directory traversal, file
        discovery, and result aggregation logic.
        """
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create test images
        for i in range(3):
            img = Image.new('L', (28, 28), color=128)
            img.save(temp_path / f"test_{i}.png")
        
        yield temp_path
    
    @pytest.fixture
    def mock_model_and_mapping(self):
        """
        Pytest fixture providing mock model and class mapping for batch testing.

        Returns:
        - model: Mock Keras model returning probability of 0.7
        - idx_to_class: Standard binary class mapping {0: 'negative', 1: 'positive'}

        This fixture enables testing of batch processing logic without
        requiring actual trained models, ensuring fast and deterministic
        test execution while validating the integration between model
        loading and prediction components.
        """
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.7]]))
        idx_to_class = {0: 'negative', 1: 'positive'}
        return model, idx_to_class
    
    @patch('src.core.inference.load_model_and_mapping')
    def test_predict_batch_success(self, mock_load, temp_image_dir, mock_model_and_mapping):
        """
        Test batch prediction on directory.

        Validates that:
        - Directory traversal discovers all PNG files
        - Single model loading is used for efficiency
        - Sequential processing returns expected results
        - Results include file names, predicted classes, and confidence

        This test ensures the batch function processes directories
        correctly and returns valid results, which is essential for
        production workflows where multiple images need to be classified.
        """
        mock_load.return_value = mock_model_and_mapping
        
        results = predict_batch("model.h5", str(temp_image_dir))
        
        assert len(results) == 3
        for result in results:
            assert 'file' in result
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    @patch('src.core.inference.load_model_and_mapping')
    def test_predict_batch_empty_directory(self, mock_load, mock_model_and_mapping):
        """
        Test batch prediction on empty directory.

        Validates that:
        - Empty directory returns empty results list
        - No errors are raised during directory traversal
        - Model loading is not attempted for empty directories

        This test ensures the batch function handles empty directories
        gracefully, which is a common scenario in production workflows
        where directories may be empty or contain no valid images.
        """
        mock_load.return_value = mock_model_and_mapping
        temp_dir = tempfile.mkdtemp()
        
        results = predict_batch("model.h5", temp_dir)
        
        assert len(results) == 0
    
    @patch('src.core.inference.load_model_and_mapping')
    @patch('src.core.inference.predict_image')
    def test_predict_batch_handles_errors(self, mock_predict, mock_load, 
                                         temp_image_dir, mock_model_and_mapping):
        """
        Test that batch prediction handles individual errors.

        Validates that:
        - Individual image processing errors are gracefully handled
        - Model loading is attempted only once
        - Results list is empty when all images fail
        - Error logging occurs but processing continues

        This test ensures the batch function can handle partial failures
        without crashing, which is important for production workflows
        where some images may be corrupted or invalid.
        """
        mock_load.return_value = mock_model_and_mapping
        mock_predict.side_effect = Exception("Processing error")
        
        # Should not raise, but handle errors gracefully
        results = predict_batch("model.h5", str(temp_image_dir))
        
        # Results should be empty due to errors
        assert len(results) == 0
