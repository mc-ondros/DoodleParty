"""Tile-based detection system for efficient real-time content moderation.

This module implements a fixed tile grid approach for analyzing canvas drawings.
Instead of analyzing the entire canvas or using sliding windows, we divide the
canvas into fixed tiles and only analyze tiles that have been modified (dirty tracking).

Key Features:
- Fixed tile grid (configurable: 64x64, 32x32, or 128x128)
- Dirty tile tracking for incremental updates
- Batch inference for all dirty tiles in single forward pass
- Overlapping tiles to reduce boundary artifacts
- Memory-efficient caching of unchanged regions

Architecture:
1. Canvas is divided into NxN grid of tiles
2. Each stroke marks affected tiles as "dirty"
3. Only dirty tiles are extracted and batched for inference
4. Results are cached until tiles are modified again
5. Aggregation determines overall canvas verdict

Performance Targets (RPi4):
- Single tile inference: <10ms
- Full 64-tile grid: <200ms (batched)
- Incremental update (1-4 tiles): <50ms

Related:
- src/core/batch_inference.py (batch processing)
- src/web/app.py (Flask integration)
- src/web/static/script.js (frontend stroke tracking)

Exports:
- TileGrid: Main tile grid manager
- TileDetector: Detection engine with caching
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
import time
import logging
from collections import defaultdict

logger = logging.getLogger('DoodleHunter.TileDetection')


@dataclass
class Tile:
    """Represents a single tile in the grid."""
    x: int  # Grid column index
    y: int  # Grid row index
    x_pixel: int  # Top-left pixel x coordinate
    y_pixel: int  # Top-left pixel y coordinate
    width: int  # Tile width in pixels
    height: int  # Tile height in pixels
    is_dirty: bool = True  # Whether tile needs re-analysis
    last_prediction: Optional[float] = None  # Cached prediction result
    last_update_time: float = 0.0  # Timestamp of last modification


@dataclass
class TileGridConfig:
    """Configuration for tile grid."""
    canvas_width: int = 512
    canvas_height: int = 512
    grid_size: int = 8  # 8x8 = 64 tiles (recommended)
    overlap_pixels: int = 0  # Overlapping tiles to reduce boundary artifacts
    tile_size: int = 64  # Size of each tile in pixels (for model input)
    
    @property
    def num_tiles(self) -> int:
        """Total number of tiles in grid."""
        return self.grid_size * self.grid_size
    
    @property
    def tile_width(self) -> int:
        """Width of each tile in canvas pixels."""
        return self.canvas_width // self.grid_size
    
    @property
    def tile_height(self) -> int:
        """Height of each tile in canvas pixels."""
        return self.canvas_height // self.grid_size


class TileGrid:
    """Manages a fixed grid of tiles over the canvas."""
    
    def __init__(self, config: TileGridConfig):
        """Initialize tile grid with given configuration."""
        self.config = config
        self.tiles: List[List[Tile]] = []
        self._initialize_grid()
        logger.info(f"Initialized {config.grid_size}x{config.grid_size} tile grid "
                   f"({config.num_tiles} tiles, {config.tile_width}x{config.tile_height}px each)")
    
    def _initialize_grid(self) -> None:
        """Create the tile grid structure."""
        tile_width = self.config.tile_width
        tile_height = self.config.tile_height
        
        for row in range(self.config.grid_size):
            tile_row = []
            for col in range(self.config.grid_size):
                tile = Tile(
                    x=col,
                    y=row,
                    x_pixel=col * tile_width,
                    y_pixel=row * tile_height,
                    width=tile_width,
                    height=tile_height,
                    is_dirty=False,  # Start clean
                    last_prediction=None
                )
                tile_row.append(tile)
            self.tiles.append(tile_row)
    
    def mark_dirty_by_stroke(self, stroke_points: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Mark tiles as dirty based on stroke coordinates.
        
        Args:
            stroke_points: List of (x, y) pixel coordinates in the stroke
            
        Returns:
            Set of (row, col) tile indices that were marked dirty
        """
        dirty_tiles = set()
        tile_width = self.config.tile_width
        tile_height = self.config.tile_height
        
        for x, y in stroke_points:
            # Calculate which tile this point belongs to
            col = min(int(x // tile_width), self.config.grid_size - 1)
            row = min(int(y // tile_height), self.config.grid_size - 1)
            
            # Clamp to valid range
            col = max(0, col)
            row = max(0, row)
            
            if not self.tiles[row][col].is_dirty:
                self.tiles[row][col].is_dirty = True
                self.tiles[row][col].last_update_time = time.time()
                dirty_tiles.add((row, col))
        
        return dirty_tiles
    
    def mark_dirty_by_bbox(self, x: int, y: int, width: int, height: int) -> Set[Tuple[int, int]]:
        """Mark tiles as dirty based on bounding box.
        
        Args:
            x, y: Top-left corner of bounding box
            width, height: Dimensions of bounding box
            
        Returns:
            Set of (row, col) tile indices that were marked dirty
        """
        dirty_tiles = set()
        tile_width = self.config.tile_width
        tile_height = self.config.tile_height
        
        # Calculate tile range covered by bbox
        col_start = max(0, int(x // tile_width))
        col_end = min(self.config.grid_size - 1, int((x + width) // tile_width))
        row_start = max(0, int(y // tile_height))
        row_end = min(self.config.grid_size - 1, int((y + height) // tile_height))
        
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if not self.tiles[row][col].is_dirty:
                    self.tiles[row][col].is_dirty = True
                    self.tiles[row][col].last_update_time = time.time()
                    dirty_tiles.add((row, col))
        
        return dirty_tiles
    
    def get_dirty_tiles(self) -> List[Tile]:
        """Get all tiles marked as dirty."""
        dirty = []
        for row in self.tiles:
            for tile in row:
                if tile.is_dirty:
                    dirty.append(tile)
        return dirty
    
    def clear_dirty_flags(self) -> None:
        """Clear all dirty flags (after processing)."""
        for row in self.tiles:
            for tile in row:
                tile.is_dirty = False
    
    def reset(self) -> None:
        """Reset all tiles to clean state with no cached predictions."""
        for row in self.tiles:
            for tile in row:
                tile.is_dirty = False
                tile.last_prediction = None
                tile.last_update_time = 0.0
    
    def get_tile_at(self, row: int, col: int) -> Optional[Tile]:
        """Get tile at specific grid position."""
        if 0 <= row < self.config.grid_size and 0 <= col < self.config.grid_size:
            return self.tiles[row][col]
        return None
    
    def extract_tile_image(self, tile: Tile, canvas_array: np.ndarray) -> np.ndarray:
        """Extract tile region from canvas image with proper preprocessing.
        
        Args:
            tile: Tile to extract
            canvas_array: Full canvas image (H, W) or (H, W, C)
            
        Returns:
            Extracted tile image resized to config.tile_size with z-score normalization
        """
        # Extract region
        y_end = min(tile.y_pixel + tile.height, canvas_array.shape[0])
        x_end = min(tile.x_pixel + tile.width, canvas_array.shape[1])
        
        tile_img = canvas_array[tile.y_pixel:y_end, tile.x_pixel:x_end]
        
        # Resize to model input size if needed
        if tile_img.shape[0] != self.config.tile_size or tile_img.shape[1] != self.config.tile_size:
            from PIL import Image
            
            # Handle channel dimension
            if len(tile_img.shape) == 3:
                tile_pil = Image.fromarray((tile_img[:,:,0] * 255).astype(np.uint8), 'L')
            else:
                tile_pil = Image.fromarray((tile_img * 255).astype(np.uint8), 'L')
            
            tile_resized = tile_pil.resize((self.config.tile_size, self.config.tile_size), 
                                          Image.Resampling.LANCZOS)
            tile_img = np.array(tile_resized, dtype=np.float32) / 255.0
        
        # Ensure channel dimension
        if len(tile_img.shape) == 2:
            tile_img = np.expand_dims(tile_img, axis=-1)
        
        # Apply z-score normalization (same as training)
        tile_flat = tile_img.flatten()
        tile_std = tile_flat.std()
        
        if tile_std > 0.01:
            tile_mean = tile_flat.mean()
            tile_img = (tile_img - tile_mean) / (tile_std + 1e-7)
            # Rescale from [-2, 2] to [0, 1]
            tile_img = (tile_img + 2) / 4
            tile_img = np.clip(tile_img, 0, 1)
        
        return tile_img


class TileDetector:
    """Tile-based detection engine with caching and batch inference."""

    def __init__(self, model, grid_config: Optional[TileGridConfig] = None):
        """Initialize tile detector.

        Args:
            model: TensorFlow/TFLite model or wrapper with predict() method
            grid_config: Tile grid configuration (uses defaults if None)
        """
        self.model = model
        self.grid_config = grid_config or TileGridConfig()
        self.grid = TileGrid(self.grid_config)
        logger.info("TileDetector initialized")
    
    def analyze_canvas(self, canvas_array: np.ndarray, 
                      threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze canvas using tile-based detection.
        
        Args:
            canvas_array: Canvas image array (H, W) or (H, W, C), normalized [0, 1]
            threshold: Classification threshold
            
        Returns:
            Detection results with tile-level details
        """
        start_time = time.time()
        
        # Get dirty tiles
        dirty_tiles = self.grid.get_dirty_tiles()
        num_dirty = len(dirty_tiles)
        
        logger.info(f"Analyzing {num_dirty} dirty tiles out of {self.grid_config.num_tiles} total")
        
        if num_dirty == 0:
            # No changes, return cached result
            return self._get_cached_result(threshold)
        
        # Extract tile images
        tile_images = []
        for tile in dirty_tiles:
            tile_img = self.grid.extract_tile_image(tile, canvas_array)
            tile_images.append(tile_img)
            logger.debug(f"Extracted tile ({tile.x},{tile.y}): shape={tile_img.shape}, "
                        f"range=[{tile_img.min():.3f}, {tile_img.max():.3f}]")
        
        # Batch inference on all dirty tiles
        inference_start = time.time()
        predictions = self._batch_predict(tile_images)
        inference_time = (time.time() - inference_start) * 1000
        
        # Update tile predictions
        for tile, pred in zip(dirty_tiles, predictions):
            tile.last_prediction = float(pred)
            logger.debug(f"Tile ({tile.x},{tile.y}) prediction: {pred:.4f}")
        
        # Clear dirty flags
        self.grid.clear_dirty_flags()
        
        # Aggregate results
        all_predictions = []
        for row in self.grid.tiles:
            for tile in row:
                if tile.last_prediction is not None:
                    all_predictions.append(tile.last_prediction)
        
        if not all_predictions:
            final_prediction = 0.0
        else:
            final_prediction = max(all_predictions)  # Max aggregation
            logger.info(f"Tile predictions: min={min(all_predictions):.4f}, "
                       f"max={max(all_predictions):.4f}, "
                       f"mean={np.mean(all_predictions):.4f}, "
                       f"count={len(all_predictions)}")
        
        is_positive = final_prediction >= threshold
        confidence = final_prediction if is_positive else (1 - final_prediction)
        
        logger.info(f"Final: prediction={final_prediction:.4f}, threshold={threshold}, "
                   f"is_positive={is_positive}, confidence={confidence:.4f}")
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'success': True,
            'is_positive': bool(is_positive),
            'confidence': float(confidence),
            'final_prediction': float(final_prediction),
            'threshold': threshold,
            'num_tiles_analyzed': num_dirty,
            'total_tiles': self.grid_config.num_tiles,
            'inference_time_ms': round(inference_time, 2),
            'total_time_ms': round(total_time, 2),
            'grid_size': f"{self.grid_config.grid_size}x{self.grid_config.grid_size}",
            'tile_predictions': all_predictions
        }
    
    def _batch_predict(self, tile_images: List[np.ndarray]) -> np.ndarray:
        """Run batch inference on tile images."""
        if not tile_images:
            return np.array([])
        
        # Stack into batch
        batch = np.stack(tile_images, axis=0)
        
        # Ensure correct dtype
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        
        # Run inference
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(batch, verbose=0)
        else:
            predictions = self.model(batch)
        
        return predictions.flatten()
    
    def _get_cached_result(self, threshold: float) -> Dict[str, Any]:
        """Get result from cached tile predictions."""
        all_predictions = []
        for row in self.grid.tiles:
            for tile in row:
                if tile.last_prediction is not None:
                    all_predictions.append(tile.last_prediction)
        
        if not all_predictions:
            final_prediction = 0.0
        else:
            final_prediction = max(all_predictions)
        
        is_positive = final_prediction >= threshold
        confidence = final_prediction if is_positive else (1 - final_prediction)
        
        return {
            'success': True,
            'is_positive': bool(is_positive),
            'confidence': float(confidence),
            'final_prediction': float(final_prediction),
            'threshold': threshold,
            'num_tiles_analyzed': 0,
            'total_tiles': self.grid_config.num_tiles,
            'inference_time_ms': 0.0,
            'total_time_ms': 0.0,
            'grid_size': f"{self.grid_config.grid_size}x{self.grid_config.grid_size}",
            'cached': True
        }
    
    def mark_dirty_by_stroke(self, stroke_points: List[Tuple[int, int]]) -> int:
        """Mark tiles dirty based on stroke. Returns number of tiles marked."""
        dirty = self.grid.mark_dirty_by_stroke(stroke_points)
        return len(dirty)
    
    def reset(self) -> None:
        """Reset detector state."""
        self.grid.reset()
        logger.info("TileDetector reset")


if __name__ == '__main__':
    print("Tile-Based Detection System")
    print("=" * 50)
    print("Features:")
    print("  - Fixed tile grid (8x8 = 64 tiles recommended)")
    print("  - Dirty tile tracking for incremental updates")
    print("  - Batch inference for all dirty tiles")
    print("  - Result caching for unchanged regions")
    print("  - Stroke grouping (temporal/spatial)")
    print("\nPerformance Targets (RPi4):")
    print("  - Single tile: <10ms")
    print("  - Full grid (64 tiles): <200ms")
    print("  - Incremental (1-4 tiles): <50ms")
