"""
Tests for patch extraction and region-based detection.

Tests cover:
- Patch extraction with various configurations
- Adaptive patch selection
- Aggregation strategies
- SlidingWindowDetector functionality
- Edge cases and error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.core.patch_extraction import (
    extract_patches,
    normalize_patch,
    select_adaptive_patches,
    aggregate_predictions,
    AggregationStrategy,
    PatchInfo,
    DetectionResult,
    SlidingWindowDetector
)


class TestPatchExtraction:
    """Tests for patch extraction functions."""
    
    def test_extract_patches_basic(self):
        """Test basic patch extraction without overlap."""
        # Create 256x256 image
        image = np.random.rand(256, 256)
        
        # Extract 128x128 patches with no overlap
        patches = extract_patches(image, patch_size=(128, 128), stride=(128, 128))
        
        # Should get 4 patches (2x2 grid)
        assert len(patches) == 4
        
        # Check patch properties
        for patch in patches:
            assert patch.patch.shape == (128, 128, 1)
            assert patch.width == 128
            assert patch.height == 128
            assert 0 <= patch.content_ratio <= 1
    
    def test_extract_patches_with_overlap(self):
        """Test patch extraction with overlapping windows."""
        image = np.random.rand(256, 256)
        
        # Extract patches with 50% overlap (stride=64)
        patches = extract_patches(image, patch_size=(128, 128), stride=(64, 64))
        
        # With stride=64, we get more patches due to overlap
        # (256-128)/64 + 1 = 3 patches per dimension
        assert len(patches) == 9  # 3x3 grid
    
    def test_extract_patches_with_channels(self):
        """Test patch extraction from image with channel dimension."""
        image = np.random.rand(256, 256, 1)
        
        patches = extract_patches(image, patch_size=(128, 128))
        
        assert len(patches) == 4
        assert all(p.patch.shape == (128, 128, 1) for p in patches)
    
    def test_extract_patches_content_ratio(self):
        """Test content ratio calculation."""
        # Create image with known content
        image = np.zeros((128, 128))
        image[32:96, 32:96] = 0.5  # 50% of pixels have content
        
        patches = extract_patches(image, patch_size=(128, 128), normalize=False)
        
        assert len(patches) == 1
        # Content ratio should be approximately 0.25 (64x64 / 128x128)
        assert 0.20 < patches[0].content_ratio < 0.30
    
    def test_normalize_patch(self):
        """Test patch normalization."""
        patch = np.array([[0, 50], [100, 255]], dtype=np.float32)
        
        normalized = normalize_patch(patch)
        
        # Should be normalized to [0, 1]
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
    
    def test_normalize_patch_uniform(self):
        """Test normalization of uniform patch."""
        patch = np.ones((10, 10)) * 128
        
        normalized = normalize_patch(patch)
        
        # Uniform patch should become all zeros
        assert np.all(normalized == 0)


class TestAdaptivePatchSelection:
    """Tests for adaptive patch selection."""
    
    def test_select_patches_by_content(self):
        """Test filtering patches by content ratio."""
        # Create patches with varying content ratios
        patches = [
            PatchInfo(np.random.rand(128, 128, 1), 0, 0, 128, 128, 0.01, 0),  # Low content
            PatchInfo(np.random.rand(128, 128, 1), 128, 0, 128, 128, 0.50, 1),  # High content
            PatchInfo(np.random.rand(128, 128, 1), 0, 128, 128, 128, 0.30, 2),  # Medium content
            PatchInfo(np.random.rand(128, 128, 1), 128, 128, 128, 128, 0.02, 3),  # Low content
        ]
        
        # Select patches with min_content_ratio=0.05
        selected = select_adaptive_patches(patches, min_content_ratio=0.05)
        
        # Should filter out patches 0 and 3
        assert len(selected) == 2
        assert selected[0].content_ratio == 0.50  # Sorted by content (descending)
        assert selected[1].content_ratio == 0.30
    
    def test_select_patches_max_limit(self):
        """Test limiting maximum number of patches."""
        patches = [
            PatchInfo(np.random.rand(128, 128, 1), 0, 0, 128, 128, 0.1 * i, i)
            for i in range(1, 11)
        ]
        
        # Select max 5 patches
        selected = select_adaptive_patches(patches, min_content_ratio=0.05, max_patches=5)
        
        assert len(selected) == 5
        # Should be top 5 by content ratio
        assert all(selected[i].content_ratio >= selected[i+1].content_ratio 
                   for i in range(len(selected)-1))
    
    def test_select_patches_empty_list(self):
        """Test with empty patch list."""
        selected = select_adaptive_patches([], min_content_ratio=0.05)
        assert len(selected) == 0


class TestAggregationStrategies:
    """Tests for prediction aggregation strategies."""
    
    def test_aggregation_max(self):
        """Test MAX aggregation strategy."""
        predictions = [
            {'confidence': 0.3, 'is_positive': False},
            {'confidence': 0.8, 'is_positive': True},
            {'confidence': 0.5, 'is_positive': True},
        ]
        
        confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=AggregationStrategy.MAX,
            threshold=0.5
        )
        
        assert confidence == 0.8
        assert is_positive is True
    
    def test_aggregation_mean(self):
        """Test MEAN aggregation strategy."""
        predictions = [
            {'confidence': 0.2, 'is_positive': False},
            {'confidence': 0.4, 'is_positive': False},
            {'confidence': 0.6, 'is_positive': True},
        ]
        
        confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=AggregationStrategy.MEAN,
            threshold=0.5
        )
        
        assert confidence == 0.4
        assert is_positive is False
    
    def test_aggregation_voting(self):
        """Test VOTING aggregation strategy."""
        predictions = [
            {'confidence': 0.3, 'is_positive': False},
            {'confidence': 0.6, 'is_positive': True},
            {'confidence': 0.7, 'is_positive': True},
        ]
        
        confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=AggregationStrategy.VOTING,
            threshold=0.5
        )
        
        # 2 out of 3 are positive, vote_ratio = 0.667
        assert confidence > 0.6
        assert is_positive is True
    
    def test_aggregation_any_positive(self):
        """Test ANY_POSITIVE aggregation strategy."""
        predictions = [
            {'confidence': 0.3, 'is_positive': False},
            {'confidence': 0.4, 'is_positive': False},
            {'confidence': 0.6, 'is_positive': True},
        ]
        
        confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=AggregationStrategy.ANY_POSITIVE,
            threshold=0.5
        )
        
        assert confidence == 1.0
        assert is_positive is True
    
    def test_aggregation_weighted_mean(self):
        """Test WEIGHTED_MEAN aggregation strategy."""
        predictions = [
            {'confidence': 0.2, 'is_positive': False, 'content_ratio': 0.1},
            {'confidence': 0.8, 'is_positive': True, 'content_ratio': 0.5},
        ]
        
        confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=AggregationStrategy.WEIGHTED_MEAN,
            threshold=0.5
        )
        
        # Weighted average: (0.2*0.1 + 0.8*0.5) / (0.1 + 0.5) = 0.7
        assert 0.65 < confidence < 0.75
        assert is_positive is True
    
    def test_aggregation_empty_predictions(self):
        """Test aggregation with empty predictions."""
        confidence, is_positive = aggregate_predictions(
            [],
            strategy=AggregationStrategy.MAX,
            threshold=0.5
        )
        
        assert confidence == 0.0
        assert is_positive is False


class TestSlidingWindowDetector:
    """Tests for SlidingWindowDetector class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        # Mock predict to return random confidences
        model.predict = Mock(side_effect=lambda x, verbose: 
            np.random.rand(x.shape[0], 1))
        return model
    
    def test_detector_initialization(self, mock_model):
        """Test detector initialization with various parameters."""
        detector = SlidingWindowDetector(
            model=mock_model,
            patch_size=(128, 128),
            stride=(64, 64),
            min_content_ratio=0.05,
            max_patches=16,
            early_stopping=True,
            aggregation_strategy=AggregationStrategy.MAX
        )
        
        assert detector.patch_size == (128, 128)
        assert detector.stride == (64, 64)
        assert detector.min_content_ratio == 0.05
        assert detector.max_patches == 16
        assert detector.early_stopping is True
    
    def test_detect_basic(self, mock_model):
        """Test basic detection on an image."""
        # Create test image
        image = np.random.rand(256, 256)
        
        # Mock model to return low confidence (negative)
        mock_model.predict = Mock(return_value=np.array([[0.2]]))
        
        detector = SlidingWindowDetector(
            model=mock_model,
            patch_size=(128, 128),
            stride=(128, 128),
            early_stopping=False
        )
        
        result = detector.detect(image)
        
        assert isinstance(result, DetectionResult)
        assert result.is_positive is False
        assert 0 <= result.confidence <= 1
        assert result.num_patches_analyzed > 0
    
    def test_detect_batch(self, mock_model):
        """Test batch detection (all patches in one forward pass)."""
        image = np.random.rand(256, 256)
        
        # Mock model for batch inference
        def batch_predict(x, verbose):
            return np.random.rand(x.shape[0], 1)
        mock_model.predict = Mock(side_effect=batch_predict)
        
        detector = SlidingWindowDetector(
            model=mock_model,
            patch_size=(128, 128),
            stride=(128, 128),
            early_stopping=False
        )
        
        result = detector.detect_batch(image)
        
        assert isinstance(result, DetectionResult)
        assert result.num_patches_analyzed > 0
        assert len(result.patch_predictions) > 0
    
    def test_early_stopping(self, mock_model):
        """Test early stopping on high confidence detection."""
        image = np.random.rand(256, 256)
        
        # Mock model to return high confidence on first patch
        mock_model.predict = Mock(return_value=np.array([[0.95]]))
        
        detector = SlidingWindowDetector(
            model=mock_model,
            patch_size=(128, 128),
            stride=(128, 128),
            early_stopping=True,
            early_stop_threshold=0.9
        )
        
        result = detector.detect(image)
        
        assert result.early_stopped is True
        # Should stop after first positive detection
        assert result.num_patches_analyzed == 1
    
    def test_empty_image_handling(self, mock_model):
        """Test handling of empty (all white/black) images."""
        # Create empty image (all zeros)
        image = np.zeros((256, 256))
        
        detector = SlidingWindowDetector(
            model=mock_model,
            patch_size=(128, 128),
            min_content_ratio=0.05
        )
        
        result = detector.detect(image)
        
        # No patches should be selected (no content)
        assert result.num_patches_analyzed == 0
        assert result.is_positive is False
        assert result.confidence == 0.0
    
    def test_different_aggregation_strategies(self, mock_model):
        """Test detector with different aggregation strategies."""
        image = np.random.rand(256, 256)
        
        strategies = [
            AggregationStrategy.MAX,
            AggregationStrategy.MEAN,
            AggregationStrategy.VOTING,
            AggregationStrategy.ANY_POSITIVE
        ]
        
        for strategy in strategies:
            # Mock model to return varied confidences
            mock_model.predict = Mock(side_effect=lambda x, verbose: 
                np.array([[0.3], [0.6], [0.4], [0.7]])[:x.shape[0]])
            
            detector = SlidingWindowDetector(
                model=mock_model,
                patch_size=(128, 128),
                stride=(128, 128),
                aggregation_strategy=strategy,
                early_stopping=False
            )
            
            result = detector.detect_batch(image)
            
            assert isinstance(result, DetectionResult)
            assert result.aggregation_strategy == strategy.value


class TestIntegration:
    """Integration tests for complete detection pipeline."""
    
    def test_complete_detection_pipeline(self):
        """Test complete detection pipeline from image to result."""
        # Create mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.75]]))
        
        # Create test image with some content
        image = np.zeros((256, 256))
        image[64:192, 64:192] = np.random.rand(128, 128)  # Center has content
        
        # Create detector
        detector = SlidingWindowDetector(
            model=model,
            patch_size=(128, 128),
            stride=(128, 128),
            min_content_ratio=0.05,
            max_patches=16,
            early_stopping=False,
            aggregation_strategy=AggregationStrategy.MAX,
            classification_threshold=0.5
        )
        
        # Run detection
        result = detector.detect_batch(image)
        
        # Verify result structure
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_positive, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.patch_predictions, list)
        assert isinstance(result.num_patches_analyzed, int)
        assert isinstance(result.early_stopped, bool)
        
        # Verify predictions
        assert result.num_patches_analyzed > 0
        for pred in result.patch_predictions:
            assert 'patch_index' in pred
            assert 'x' in pred
            assert 'y' in pred
            assert 'confidence' in pred
            assert 'is_positive' in pred
            assert 'content_ratio' in pred
    
    def test_content_dilution_detection(self):
        """Test detection of diluted inappropriate content."""
        # Simulate content dilution attack:
        # Small inappropriate content (high confidence) mixed with lots of innocent content
        
        model = Mock()
        
        # First patch: high confidence (inappropriate)
        # Other patches: low confidence (innocent)
        def variable_predict(x, verbose):
            if x.shape[0] == 1:
                return np.array([[0.95]])  # Single patch inference
            else:
                # Batch inference: first patch high, others low
                return np.vstack([np.array([[0.95]])] + 
                                [np.array([[0.1]]) for _ in range(x.shape[0]-1)])
        
        model.predict = Mock(side_effect=variable_predict)
        
        # Create image with small region of content
        image = np.zeros((512, 512))
        image[0:128, 0:128] = np.random.rand(128, 128) * 0.8  # Small suspicious region
        
        detector = SlidingWindowDetector(
            model=model,
            patch_size=(128, 128),
            stride=(128, 128),
            min_content_ratio=0.05,
            aggregation_strategy=AggregationStrategy.MAX,  # Most aggressive
            classification_threshold=0.5
        )
        
        result = detector.detect_batch(image)
        
        # Should detect as positive due to MAX aggregation
        # (takes maximum confidence from all patches)
        assert result.is_positive is True
        assert result.confidence >= 0.9
        
        # Verify multiple patches were analyzed
        assert result.num_patches_analyzed > 1


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
