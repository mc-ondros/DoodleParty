"""
Tests for content removal system.

Verifies localization, removal strategies, and undo functionality.
"""

import pytest
import numpy as np
from src.core.content_removal import (
    ContentRemover,
    RemovalStrategy,
    FlaggedRegion,
    LocalizationResult
)


class TestContentRemover:
    """Test suite for ContentRemover class."""
    
    def test_initialization(self):
        """Test ContentRemover initialization."""
        remover = ContentRemover(blur_kernel_size=25)
        assert remover.blur_kernel_size == 25
        assert len(remover.undo_stack) == 0
        assert remover.max_undo_depth == 10
    
    def test_blur_kernel_size_odd(self):
        """Test that blur kernel size is always odd."""
        remover = ContentRemover(blur_kernel_size=24)
        assert remover.blur_kernel_size == 25  # Should be adjusted to 25
    
    def test_highlight_regions(self):
        """Test highlighting flagged regions."""
        remover = ContentRemover()
        
        # Create test image
        image = np.ones((512, 512), dtype=np.uint8) * 200
        
        # Create flagged regions
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            ),
            FlaggedRegion(
                bounding_box=(200, 200, 60, 60),
                confidence=0.92,
                region_type='contour',
                region_id=1
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.92,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Highlight regions
        highlighted = remover.highlight_regions(image, localization)
        
        # Check output shape
        assert highlighted.shape == (512, 512, 3)
        
        # Check that highlighted regions differ from original
        assert not np.array_equal(highlighted[100:150, 100:150], image[100:150, 100:150])
    
    def test_apply_blur(self):
        """Test blur removal strategy."""
        remover = ContentRemover(blur_kernel_size=25)
        
        # Create test image with pattern in the region to be blurred
        image = np.ones((512, 512), dtype=np.uint8) * 200
        
        # Create a checkerboard pattern in the region
        for i in range(100, 150, 10):
            for j in range(100, 150, 10):
                if (i + j) % 20 == 0:
                    image[i:i+10, j:j+10] = 50
                else:
                    image[i:i+10, j:j+10] = 250
        
        # Create flagged region
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.85,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Store original standard deviation
        original_std = image[100:150, 100:150].std()
        
        # Apply blur
        result = remover.remove_content(image, localization, RemovalStrategy.BLUR)
        
        assert result.success
        assert result.regions_removed == 1
        assert result.strategy_used == 'blur'
        assert result.modified_image is not None
        
        # Check that region was blurred (values should be more uniform)
        blurred_std = result.modified_image[100:150, 100:150].std()
        assert blurred_std < original_std
        assert original_std > 0  # Ensure we had variation to begin with
    
    def test_apply_erase(self):
        """Test erase removal strategy."""
        remover = ContentRemover()
        
        # Create test image
        image = np.ones((512, 512), dtype=np.uint8) * 200
        image[100:150, 100:150] = 50  # Dark region
        
        # Create flagged region
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.85,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Apply erase
        result = remover.remove_content(image, localization, RemovalStrategy.ERASE)
        
        assert result.success
        assert result.regions_removed == 1
        assert result.strategy_used == 'erase'
        
        # Check that region was filled with background color
        erased_region = result.modified_image[100:150, 100:150]
        assert np.all(erased_region == 243)
    
    def test_apply_placeholder(self):
        """Test placeholder removal strategy."""
        remover = ContentRemover()
        
        # Create test image
        image = np.ones((512, 512), dtype=np.uint8) * 200
        
        # Create flagged region
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.85,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Apply placeholder
        result = remover.remove_content(image, localization, RemovalStrategy.PLACEHOLDER)
        
        assert result.success
        assert result.regions_removed == 1
        assert result.strategy_used == 'placeholder'
        assert result.modified_image.shape == (512, 512, 3)  # Should be RGB
    
    def test_undo_functionality(self):
        """Test undo stack functionality."""
        remover = ContentRemover()
        
        # Create test images
        image1 = np.ones((512, 512), dtype=np.uint8) * 100
        image2 = np.ones((512, 512), dtype=np.uint8) * 150
        
        # Create flagged region
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.85,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Apply removal (should save to undo stack)
        result1 = remover.remove_content(image1, localization, RemovalStrategy.BLUR)
        assert result1.can_undo
        assert remover.can_undo()
        
        # Apply another removal
        result2 = remover.remove_content(image2, localization, RemovalStrategy.BLUR)
        assert result2.can_undo
        
        # Undo
        undone = remover.undo()
        assert undone is not None
        assert np.array_equal(undone, image2)
        
        # Undo again
        undone2 = remover.undo()
        assert undone2 is not None
        assert np.array_equal(undone2, image1)
        
        # No more undo available
        assert not remover.can_undo()
        assert remover.undo() is None
    
    def test_undo_stack_limit(self):
        """Test that undo stack respects max depth."""
        remover = ContentRemover()
        remover.max_undo_depth = 3
        
        # Create test images
        images = [np.ones((512, 512), dtype=np.uint8) * i for i in range(10)]
        
        # Create flagged region
        regions = [
            FlaggedRegion(
                bounding_box=(100, 100, 50, 50),
                confidence=0.85,
                region_type='tile',
                region_id=0
            )
        ]
        
        localization = LocalizationResult(
            flagged_regions=regions,
            overall_confidence=0.85,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Apply multiple removals
        for img in images:
            remover.remove_content(img, localization, RemovalStrategy.BLUR)
        
        # Check stack size
        assert len(remover.undo_stack) == 3
    
    def test_no_flagged_regions(self):
        """Test removal with no flagged regions."""
        remover = ContentRemover()
        
        image = np.ones((512, 512), dtype=np.uint8) * 200
        
        # Empty localization
        localization = LocalizationResult(
            flagged_regions=[],
            overall_confidence=0.0,
            detection_method='tile',
            canvas_dimensions=(512, 512)
        )
        
        # Apply removal
        result = remover.remove_content(image, localization, RemovalStrategy.BLUR)
        
        assert result.success
        assert result.regions_removed == 0
        assert result.error == "No flagged regions to remove"
        assert not result.can_undo
    
    def test_localize_from_tiles(self):
        """Test creating localization from tile results."""
        remover = ContentRemover()
        
        # Mock tile result
        from src.core.tile_detection import TileDetectionResult, TileInfo, TileCoordinate
        
        tile_predictions = [
            TileInfo(
                coordinate=TileCoordinate(row=0, col=0),
                bounding_box=(0, 0, 64, 64),
                confidence=0.85,
                is_positive=True
            ),
            TileInfo(
                coordinate=TileCoordinate(row=1, col=1),
                bounding_box=(64, 64, 64, 64),
                confidence=0.92,
                is_positive=True
            ),
            TileInfo(
                coordinate=TileCoordinate(row=2, col=2),
                bounding_box=(128, 128, 64, 64),
                confidence=0.3,
                is_positive=False
            )
        ]
        
        tile_result = TileDetectionResult(
            is_positive=True,
            confidence=0.92,
            tile_predictions=tile_predictions,
            num_tiles_analyzed=3,
            num_tiles_cached=0,
            grid_dimensions=(8, 8),
            tile_size=64,
            canvas_dimensions=(512, 512)
        )
        
        # Create localization
        localization = remover.localize_from_tiles(tile_result, threshold=0.5)
        
        assert len(localization.flagged_regions) == 2  # Only positive tiles
        assert localization.overall_confidence == 0.92
        assert localization.detection_method == 'tile'
        assert localization.canvas_dimensions == (512, 512)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
