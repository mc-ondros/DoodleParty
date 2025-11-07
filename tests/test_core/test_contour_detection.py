"""
Unit tests for hierarchical contour detection.

Tests the ability to detect offensive content nested inside benign shapes
using RETR_TREE mode. This addresses the limitation where RETR_EXTERNAL
only finds outer boundaries and misses nested content.

Test scenarios:
- Offensive content inside a circle (containment attack)
- Multiple nested levels (circle -> square -> offensive)
- Mixed content (some nested, some not)
- Edge cases (empty canvas, single shape, no nesting)

Related:
- src/core/contour_detection.py (ContourDetector implementation)
- .documentation/roadmap.md (Phase 3.1 requirements)
"""

import unittest
import numpy as np
import cv2
from typing import Tuple

from src.core.contour_detection import (
    ContourDetector,
    ContourRetrievalMode,
    detect_contours,
    ContourInfo
)


class TestHierarchicalContourDetection(unittest.TestCase):
    """Test hierarchical contour detection with RETR_TREE mode."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model that classifies based on simple heuristics
        # For testing, we'll use a simple rule: small shapes are offensive
        self.mock_model = MockClassifier()

    def create_circle(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: int = 255
    ) -> np.ndarray:
        """Draw a circle on the image."""
        cv2.circle(image, center, radius, color, -1)
        return image

    def create_rectangle(
        self,
        image: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        color: int = 255
    ) -> np.ndarray:
        """Draw a rectangle on the image."""
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        return image

    def test_detect_contours_external_mode(self):
        """Test that RETR_EXTERNAL only finds outer boundaries."""
        # Create image with nested shapes (circle containing a small rectangle)
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Draw outer circle
        self.create_circle(image, (100, 100), 80, 255)
        
        # Draw inner rectangle (nested inside circle)
        # First, invert to create hole, then draw rectangle
        image_with_hole = image.copy()
        self.create_circle(image_with_hole, (100, 100), 60, 0)
        self.create_rectangle(image_with_hole, (80, 80), (120, 120), 255)

        # Detect contours with EXTERNAL mode
        contours, hierarchy = detect_contours(image_with_hole, ContourRetrievalMode.EXTERNAL)

        # RETR_EXTERNAL should only find the outer circle
        # The inner rectangle should not be detected as a separate contour
        self.assertGreater(len(contours), 0, "Should detect at least the outer boundary")
        
        # Check that no nested contours are detected
        nested_count = sum(1 for c in contours if c.hierarchy_level > 0)
        self.assertEqual(nested_count, 0, "RETR_EXTERNAL should not detect nested contours")

    def test_detect_contours_tree_mode(self):
        """Test that RETR_TREE finds nested contours."""
        # Create image with nested shapes
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Draw outer circle
        self.create_circle(image, (100, 100), 80, 255)
        
        # Create hole and inner shape
        self.create_circle(image, (100, 100), 60, 0)
        self.create_rectangle(image, (80, 80), (120, 120), 255)

        # Detect contours with TREE mode
        contours, hierarchy = detect_contours(image, ContourRetrievalMode.TREE)

        # RETR_TREE should find both outer and inner contours
        self.assertGreater(len(contours), 1, "Should detect multiple contours including nested")
        
        # Check that we have nested contours
        nested_count = sum(1 for c in contours if c.hierarchy_level > 0)
        self.assertGreater(nested_count, 0, "RETR_TREE should detect nested contours")

    def test_hierarchy_levels(self):
        """Test that hierarchy levels are correctly calculated."""
        # Create image with multiple nesting levels
        image = np.zeros((300, 300), dtype=np.uint8)
        
        # Level 0: Outer circle
        self.create_circle(image, (150, 150), 140, 255)
        
        # Level 1: Middle circle (nested in outer)
        self.create_circle(image, (150, 150), 100, 0)
        self.create_circle(image, (150, 150), 90, 255)
        
        # Level 2: Inner circle (nested in middle)
        self.create_circle(image, (150, 150), 50, 0)
        self.create_circle(image, (150, 150), 40, 255)

        # Detect contours with TREE mode
        contours, hierarchy = detect_contours(image, ContourRetrievalMode.TREE)

        # Should have contours at different hierarchy levels
        levels = [c.hierarchy_level for c in contours]
        max_level = max(levels) if levels else 0
        
        self.assertGreater(max_level, 0, "Should have nested contours at different levels")

    def test_parent_child_relationships(self):
        """Test that parent-child relationships are correctly identified."""
        # Create simple nested structure
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Outer shape
        self.create_circle(image, (100, 100), 80, 255)
        
        # Inner shape (child)
        self.create_circle(image, (100, 100), 60, 0)
        self.create_rectangle(image, (80, 80), (120, 120), 255)

        # Detect contours
        contours, hierarchy = detect_contours(image, ContourRetrievalMode.TREE)

        # Find contours with parents
        children = [c for c in contours if c.parent_id is not None]
        
        self.assertGreater(len(children), 0, "Should have child contours with parent references")

    def test_contour_detector_external_vs_tree(self):
        """Test ContourDetector behavior with EXTERNAL vs TREE modes."""
        # Create test image with nested offensive content
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Large benign circle
        self.create_circle(image, (100, 100), 80, 255)
        
        # Small offensive shape inside (simulated)
        self.create_circle(image, (100, 100), 60, 0)
        self.create_rectangle(image, (90, 90), (110, 110), 255)

        # Test with EXTERNAL mode
        detector_external = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.EXTERNAL,
            min_contour_area=50,
            is_tflite=False
        )
        result_external = detector_external.detect(image)

        # Test with TREE mode
        detector_tree = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=50,
            is_tflite=False
        )
        result_tree = detector_tree.detect(image)

        # TREE mode should analyze more contours than EXTERNAL
        self.assertGreaterEqual(
            result_tree.num_contours_analyzed,
            result_external.num_contours_analyzed,
            "TREE mode should analyze at least as many contours as EXTERNAL"
        )

    def test_detect_hierarchical_method(self):
        """Test the detect_hierarchical method for nested content detection."""
        # Create image with offensive content nested in benign shape
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Outer benign circle
        self.create_circle(image, (100, 100), 80, 255)
        
        # Inner offensive content
        self.create_circle(image, (100, 100), 60, 0)
        self.create_rectangle(image, (85, 85), (115, 115), 255)

        # Create detector with TREE mode
        detector = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=50,
            is_tflite=False
        )

        # Run hierarchical detection
        result = detector.detect_hierarchical(image)

        # Should detect the nested content
        self.assertIsNotNone(result)
        self.assertGreater(result.num_contours_analyzed, 0)

    def test_hierarchical_detection_requires_tree_mode(self):
        """Test that detect_hierarchical raises error with EXTERNAL mode."""
        image = np.zeros((200, 200), dtype=np.uint8)
        self.create_circle(image, (100, 100), 50, 255)

        # Create detector with EXTERNAL mode
        detector = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.EXTERNAL,
            is_tflite=False
        )

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            detector.detect_hierarchical(image)
        
        self.assertIn("requires retrieval_mode=TREE", str(context.exception))

    def test_empty_canvas(self):
        """Test detection on empty canvas."""
        image = np.zeros((200, 200), dtype=np.uint8)

        detector = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.TREE,
            is_tflite=False
        )

        result = detector.detect(image)

        self.assertFalse(result.is_positive)
        self.assertEqual(result.num_contours_analyzed, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_single_shape_no_nesting(self):
        """Test detection with single shape (no nesting)."""
        image = np.zeros((200, 200), dtype=np.uint8)
        self.create_circle(image, (100, 100), 50, 255)

        detector = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=100,
            is_tflite=False
        )

        result = detector.detect(image)

        # Should detect the single shape
        self.assertGreater(result.num_contours_analyzed, 0)
        # All contours should be at level 0 (no nesting)
        for contour in result.contour_predictions:
            self.assertEqual(contour.hierarchy_level, 0)

    def test_min_contour_area_filtering(self):
        """Test that small contours are filtered out."""
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Large shape
        self.create_circle(image, (100, 100), 80, 255)
        
        # Very small shape (should be filtered)
        self.create_rectangle(image, (10, 10), (15, 15), 255)

        detector = ContourDetector(
            model=self.mock_model,
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=500,  # High threshold
            is_tflite=False
        )

        result = detector.detect(image)

        # Should only detect large shape
        for contour in result.contour_predictions:
            self.assertGreaterEqual(contour.area, 500)

    def test_early_stopping(self):
        """Test early stopping on high-confidence detection."""
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Create multiple shapes
        self.create_circle(image, (50, 50), 30, 255)
        self.create_circle(image, (150, 150), 30, 255)

        # Create detector with early stopping enabled
        detector = ContourDetector(
            model=MockHighConfidenceClassifier(),  # Always returns high confidence
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=50,
            early_stopping=True,
            early_stop_threshold=0.9,
            is_tflite=False
        )

        result = detector.detect(image)

        # Should stop early
        self.assertTrue(result.early_stopped)
        # Should analyze fewer contours than total available
        contours_total, _ = detect_contours(image, ContourRetrievalMode.TREE)
        filtered_total = len([c for c in contours_total if c.area >= 50])
        self.assertLessEqual(result.num_contours_analyzed, filtered_total)


class MockClassifier:
    """Mock classifier for testing."""
    
    def predict(self, image_batch, verbose=0):
        """
        Mock prediction based on simple heuristic.
        
        Small shapes (area < 2000 pixels) are classified as offensive.
        """
        # Calculate approximate area from image
        img = image_batch[0]
        white_pixels = np.sum(img > 0.5)
        
        # Small shapes are "offensive"
        if white_pixels < 2000:
            confidence = 0.8
        else:
            confidence = 0.2
        
        return np.array([[confidence]])


class MockHighConfidenceClassifier:
    """Mock classifier that always returns high confidence."""
    
    def predict(self, image_batch, verbose=0):
        """Always return high confidence positive."""
        return np.array([[0.95]])


if __name__ == '__main__':
    unittest.main()
