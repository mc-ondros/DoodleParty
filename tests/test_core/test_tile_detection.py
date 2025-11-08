"""
Unit tests for tile-based detection system.

Tests the TileDetector class including:
- Grid dimension calculation for various canvas sizes
- Tile coordinate mapping
- Dirty tile tracking
- Tile caching
- Non-square canvas support
- Edge case handling

Related:
- src/core/tile_detection.py (TileDetector implementation)
- .documentation/roadmap.md (Phase 3.2 requirements)
"""

import unittest
import numpy as np
from typing import List, Tuple

from src.core.tile_detection import (
    TileDetector,
    TileGrid,
    TileCoordinate,
    TileSize,
    TileDetectionResult
)


class TestTileGrid(unittest.TestCase):
    """Test TileGrid coordinate mapping and grid calculations."""
    
    def test_square_canvas_divisible(self):
        """Test grid calculation for square canvas with divisible dimensions."""
        grid = TileGrid(canvas_width=512, canvas_height=512, tile_size=64)
        
        self.assertEqual(grid.grid_cols, 8)
        self.assertEqual(grid.grid_rows, 8)
        self.assertEqual(grid.total_tiles, 64)
    
    def test_square_canvas_non_divisible_clip(self):
        """Test grid calculation with non-divisible dimensions (clip mode)."""
        grid = TileGrid(canvas_width=500, canvas_height=500, tile_size=64, padding_mode='clip')
        
        # 500 // 64 = 7 (clips 52 pixels)
        self.assertEqual(grid.grid_cols, 7)
        self.assertEqual(grid.grid_rows, 7)
        self.assertEqual(grid.total_tiles, 49)
    
    def test_square_canvas_non_divisible_pad(self):
        """Test grid calculation with non-divisible dimensions (pad mode)."""
        grid = TileGrid(canvas_width=500, canvas_height=500, tile_size=64, padding_mode='pad')
        
        # ceil(500 / 64) = 8 (pads to include partial tiles)
        self.assertEqual(grid.grid_cols, 8)
        self.assertEqual(grid.grid_rows, 8)
        self.assertEqual(grid.total_tiles, 64)
    
    def test_non_square_canvas(self):
        """Test grid calculation for non-square canvas."""
        grid = TileGrid(canvas_width=1024, canvas_height=768, tile_size=64)
        
        self.assertEqual(grid.grid_cols, 16)  # 1024 // 64
        self.assertEqual(grid.grid_rows, 12)  # 768 // 64
        self.assertEqual(grid.total_tiles, 192)
    
    def test_canvas_to_tile_conversion(self):
        """Test converting canvas coordinates to tile coordinates."""
        grid = TileGrid(canvas_width=512, canvas_height=512, tile_size=64)
        
        # Top-left corner
        coord = grid.canvas_to_tile(0, 0)
        self.assertEqual(coord.row, 0)
        self.assertEqual(coord.col, 0)
        
        # Center
        coord = grid.canvas_to_tile(256, 256)
        self.assertEqual(coord.row, 4)
        self.assertEqual(coord.col, 4)
        
        # Bottom-right
        coord = grid.canvas_to_tile(511, 511)
        self.assertEqual(coord.row, 7)
        self.assertEqual(coord.col, 7)
    
    def test_get_tile_bbox(self):
        """Test getting bounding box for a tile."""
        grid = TileGrid(canvas_width=512, canvas_height=512, tile_size=64)
        
        # First tile
        bbox = grid.get_tile_bbox(TileCoordinate(row=0, col=0))
        self.assertEqual(bbox, (0, 0, 64, 64))
        
        # Middle tile
        bbox = grid.get_tile_bbox(TileCoordinate(row=4, col=4))
        self.assertEqual(bbox, (256, 256, 64, 64))
        
        # Last tile
        bbox = grid.get_tile_bbox(TileCoordinate(row=7, col=7))
        self.assertEqual(bbox, (448, 448, 64, 64))
    
    def test_get_tile_bbox_edge_partial(self):
        """Test bounding box for partial edge tiles."""
        grid = TileGrid(canvas_width=500, canvas_height=500, tile_size=64, padding_mode='pad')
        
        # Last column tile (partial width)
        bbox = grid.get_tile_bbox(TileCoordinate(row=0, col=7))
        self.assertEqual(bbox[0], 448)  # x
        self.assertEqual(bbox[1], 0)    # y
        self.assertEqual(bbox[2], 52)   # width (500 - 448)
        self.assertEqual(bbox[3], 64)   # height
    
    def test_get_tiles_in_bbox(self):
        """Test getting all tiles intersecting a bounding box."""
        grid = TileGrid(canvas_width=512, canvas_height=512, tile_size=64)
        
        # Small bbox within single tile
        tiles = grid.get_tiles_in_bbox(10, 10, 20, 20)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0], TileCoordinate(row=0, col=0))
        
        # Bbox spanning 4 tiles (2x2)
        tiles = grid.get_tiles_in_bbox(60, 60, 10, 10)
        self.assertEqual(len(tiles), 4)
        
        # Bbox spanning entire canvas
        tiles = grid.get_tiles_in_bbox(0, 0, 512, 512)
        self.assertEqual(len(tiles), 64)
    
    def test_get_all_tiles(self):
        """Test getting all tiles in grid."""
        grid = TileGrid(canvas_width=256, canvas_height=256, tile_size=64)
        
        tiles = grid.get_all_tiles()
        self.assertEqual(len(tiles), 16)  # 4x4 grid
        
        # Check first and last tiles
        self.assertEqual(tiles[0], TileCoordinate(row=0, col=0))
        self.assertEqual(tiles[-1], TileCoordinate(row=3, col=3))


class TestTileDetector(unittest.TestCase):
    """Test TileDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockClassifier()
    
    def test_initialization(self):
        """Test TileDetector initialization."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=512,
            tile_size=64,
            is_tflite=False
        )
        
        self.assertEqual(detector.grid.grid_cols, 8)
        self.assertEqual(detector.grid.grid_rows, 8)
        self.assertEqual(len(detector.dirty_tiles), 64)  # All tiles initially dirty
    
    def test_mark_dirty_tiles_single_point(self):
        """Test marking tiles dirty from a single point."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=512,
            tile_size=64,
            is_tflite=False
        )
        
        # Clear dirty tiles
        detector.dirty_tiles.clear()
        
        # Mark single point
        detector.mark_dirty_tiles([(100, 100)])
        
        # Should mark exactly one tile
        self.assertEqual(len(detector.dirty_tiles), 1)
        self.assertIn(TileCoordinate(row=1, col=1), detector.dirty_tiles)
    
    def test_mark_dirty_tiles_stroke(self):
        """Test marking tiles dirty from a stroke."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=512,
            tile_size=64,
            is_tflite=False
        )
        
        detector.dirty_tiles.clear()
        
        # Stroke spanning multiple tiles
        stroke = [(i, i) for i in range(0, 200, 10)]
        detector.mark_dirty_tiles(stroke)
        
        # Should mark multiple tiles along diagonal
        self.assertGreater(len(detector.dirty_tiles), 1)
    
    def test_mark_all_dirty(self):
        """Test marking all tiles as dirty."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=256,
            canvas_height=256,
            tile_size=64,
            is_tflite=False
        )
        
        detector.dirty_tiles.clear()
        detector.mark_all_dirty()
        
        self.assertEqual(len(detector.dirty_tiles), 16)  # 4x4 grid
    
    def test_tile_extraction(self):
        """Test extracting a tile from an image."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=256,
            canvas_height=256,
            tile_size=64,
            is_tflite=False
        )
        
        # Create test image
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Extract tile
        tile = detector.extract_tile(image, TileCoordinate(row=0, col=0), target_size=(28, 28))
        
        # Check shape
        self.assertEqual(tile.shape, (28, 28, 1))
        
        # Check normalization
        self.assertGreaterEqual(tile.min(), 0.0)
        self.assertLessEqual(tile.max(), 1.0)
    
    def test_tile_extraction_edge_partial(self):
        """Test extracting a partial edge tile."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=200,
            canvas_height=200,
            tile_size=64,
            padding_mode='pad',
            is_tflite=False
        )
        
        # Create test image
        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        # Extract edge tile (should be padded)
        tile = detector.extract_tile(image, TileCoordinate(row=3, col=3), target_size=(28, 28))
        
        # Should still be correct shape after padding
        self.assertEqual(tile.shape, (28, 28, 1))
    
    def test_detect_full_analysis(self):
        """Test full tile detection (all tiles)."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=256,
            canvas_height=256,
            tile_size=64,
            is_tflite=False,
            enable_caching=False
        )
        
        # Create test image
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Run detection
        result = detector.detect(image, force_full_analysis=True)
        
        # Check result
        self.assertIsInstance(result, TileDetectionResult)
        self.assertEqual(result.num_tiles_analyzed, 16)  # 4x4 grid
        self.assertEqual(len(result.tile_predictions), 16)
        self.assertEqual(result.grid_dimensions, (4, 4))
        self.assertEqual(result.tile_size, 64)
    
    def test_detect_incremental_with_caching(self):
        """Test incremental detection with caching."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=256,
            canvas_height=256,
            tile_size=64,
            is_tflite=False,
            enable_caching=True
        )
        
        # Create test image
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # First detection (all tiles dirty)
        result1 = detector.detect(image)
        self.assertEqual(result1.num_tiles_analyzed, 16)
        self.assertEqual(result1.num_tiles_cached, 0)
        
        # Mark only one tile dirty
        detector.mark_dirty_tiles([(10, 10)])
        
        # Second detection (only one tile dirty, rest cached)
        result2 = detector.detect(image)
        self.assertEqual(result2.num_tiles_analyzed, 1)
        # Note: num_tiles_cached counts tiles in this detection, not total cache
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=256,
            canvas_height=256,
            tile_size=64,
            is_tflite=False,
            enable_caching=True
        )
        
        # Create test image and run detection
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        detector.detect(image)
        
        # Cache should have entries
        self.assertGreater(len(detector.tile_cache), 0)
        
        # Clear cache
        detector.clear_cache()
        
        # Cache should be empty and all tiles dirty
        self.assertEqual(len(detector.tile_cache), 0)
        self.assertEqual(len(detector.dirty_tiles), 16)
    
    def test_non_square_canvas_detection(self):
        """Test detection on non-square canvas."""
        detector = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=768,
            tile_size=64,
            is_tflite=False
        )
        
        # Create non-square image
        image = np.random.randint(0, 255, (768, 512), dtype=np.uint8)
        
        # Run detection
        result = detector.detect(image, force_full_analysis=True)
        
        # Check dimensions
        self.assertEqual(result.canvas_dimensions, (512, 768))
        self.assertEqual(result.grid_dimensions, (12, 8))  # 768/64 x 512/64
        self.assertEqual(result.num_tiles_analyzed, 96)
    
    def test_different_tile_sizes(self):
        """Test detection with different tile sizes."""
        image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # Test with 32x32 tiles
        detector_32 = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=512,
            tile_size=32,
            is_tflite=False
        )
        result_32 = detector_32.detect(image, force_full_analysis=True)
        self.assertEqual(result_32.grid_dimensions, (16, 16))
        self.assertEqual(result_32.num_tiles_analyzed, 256)
        
        # Test with 128x128 tiles
        detector_128 = TileDetector(
            model=self.mock_model,
            canvas_width=512,
            canvas_height=512,
            tile_size=128,
            is_tflite=False
        )
        result_128 = detector_128.detect(image, force_full_analysis=True)
        self.assertEqual(result_128.grid_dimensions, (4, 4))
        self.assertEqual(result_128.num_tiles_analyzed, 16)


class MockClassifier:
    """Mock classifier for testing."""
    
    def predict(self, image_batch, verbose=0):
        """
        Mock prediction based on tile brightness.
        
        Brighter tiles get higher confidence.
        """
        img = image_batch[0]
        brightness = np.mean(img)
        
        # Map brightness to confidence
        confidence = min(brightness * 2, 1.0)
        
        return np.array([[confidence]])


if __name__ == '__main__':
    unittest.main()
