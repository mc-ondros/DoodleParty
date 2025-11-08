"""
Tile-based detection system for DoodleHunter.

This module implements tile-based detection by dividing the canvas into a fixed
grid and analyzing each tile independently. This approach is robust against
content dilution attacks where offensive content is mixed with innocent shapes.

Key Features:
- Flexible canvas dimensions (not hardcoded to square)
- Dynamic grid calculation based on tile size
- Dirty tile tracking for incremental updates
- Tile caching to avoid redundant inference
- Configurable tile sizes (32x32, 64x64, 128x128)
- Batch inference optimization

Why tile-based detection:
- Prevents content dilution by analyzing all regions independently
- Enables incremental updates (only re-analyze changed tiles)
- Predictable performance (fixed number of tiles)
- Better coverage than contour-based for distributed content

Related:
- src/web/app.py (Flask API endpoints)
- src/core/inference.py (model inference pipeline)
- src/core/contour_detection.py (alternative detection method)

Exports:
- TileDetector: Main class for tile-based detection
- TileGrid: Grid configuration and coordinate mapping
- TileDetectionResult: Result dataclass
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class TileSize(Enum):
    """Predefined tile sizes with performance characteristics."""
    SMALL = 32   # High precision, more tiles, slower
    MEDIUM = 64  # Recommended balance
    LARGE = 128  # Low budget, fewer tiles, faster


@dataclass
class TileCoordinate:
    """Coordinate of a tile in the grid."""
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


@dataclass
class TileInfo:
    """Information about a single tile."""
    coordinate: TileCoordinate
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height) in canvas coords
    confidence: Optional[float] = None
    is_positive: Optional[bool] = None
    is_cached: bool = False


@dataclass
class TileDetectionResult:
    """Result of tile-based detection."""
    is_positive: bool
    confidence: float
    tile_predictions: List[TileInfo]
    num_tiles_analyzed: int
    num_tiles_cached: int
    grid_dimensions: Tuple[int, int]  # (rows, cols)
    tile_size: int
    canvas_dimensions: Tuple[int, int]  # (width, height)


class TileGrid:
    """
    Grid configuration and coordinate mapping for tile-based detection.
    
    Handles flexible canvas dimensions and provides utilities for:
    - Converting between canvas coordinates and tile indices
    - Calculating grid dimensions
    - Handling non-divisible dimensions
    """
    
    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        tile_size: int,
        padding_mode: str = 'clip'
    ):
        """
        Initialize tile grid.
        
        Args:
            canvas_width: Canvas width in pixels
            canvas_height: Canvas height in pixels
            tile_size: Size of each tile (tiles are square)
            padding_mode: How to handle non-divisible dimensions
                         'clip' - ignore partial tiles at edges
                         'pad' - include partial tiles (pad with zeros)
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.tile_size = tile_size
        self.padding_mode = padding_mode
        
        # Calculate grid dimensions
        if padding_mode == 'clip':
            self.grid_cols = canvas_width // tile_size
            self.grid_rows = canvas_height // tile_size
        else:  # pad
            self.grid_cols = (canvas_width + tile_size - 1) // tile_size
            self.grid_rows = (canvas_height + tile_size - 1) // tile_size
        
        self.total_tiles = self.grid_rows * self.grid_cols
    
    def get_tile_bbox(self, coord: TileCoordinate) -> Tuple[int, int, int, int]:
        """
        Get bounding box for a tile in canvas coordinates.
        
        Args:
            coord: Tile coordinate
        
        Returns:
            Tuple of (x, y, width, height)
        """
        x = coord.col * self.tile_size
        y = coord.row * self.tile_size
        
        # Handle edge tiles that may be smaller
        width = min(self.tile_size, self.canvas_width - x)
        height = min(self.tile_size, self.canvas_height - y)
        
        return (x, y, width, height)
    
    def canvas_to_tile(self, x: int, y: int) -> TileCoordinate:
        """
        Convert canvas coordinates to tile coordinate.
        
        Args:
            x: X coordinate in canvas
            y: Y coordinate in canvas
        
        Returns:
            TileCoordinate
        """
        col = min(x // self.tile_size, self.grid_cols - 1)
        row = min(y // self.tile_size, self.grid_rows - 1)
        return TileCoordinate(row=row, col=col)
    
    def get_tiles_in_bbox(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> List[TileCoordinate]:
        """
        Get all tiles that intersect with a bounding box.
        
        Args:
            x: X coordinate of bbox top-left
            y: Y coordinate of bbox top-left
            width: Bbox width
            height: Bbox height
        
        Returns:
            List of TileCoordinate objects
        """
        # Get tile coordinates for bbox corners
        top_left = self.canvas_to_tile(x, y)
        bottom_right = self.canvas_to_tile(
            min(x + width - 1, self.canvas_width - 1),
            min(y + height - 1, self.canvas_height - 1)
        )
        
        # Generate all tiles in the range
        tiles = []
        for row in range(top_left.row, bottom_right.row + 1):
            for col in range(top_left.col, bottom_right.col + 1):
                tiles.append(TileCoordinate(row=row, col=col))
        
        return tiles
    
    def get_all_tiles(self) -> List[TileCoordinate]:
        """Get all tile coordinates in the grid."""
        tiles = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                tiles.append(TileCoordinate(row=row, col=col))
        return tiles


class TileDetector:
    """
    Tile-based detector for drawing classification.
    
    Divides canvas into fixed-size tiles and analyzes each independently.
    Supports dirty tile tracking for incremental updates and caching for
    performance optimization.
    
    Example:
        detector = TileDetector(
            model=model,
            canvas_width=512,
            canvas_height=512,
            tile_size=64
        )
        
        # Initial detection
        result = detector.detect(image)
        
        # Mark tiles dirty after new strokes
        detector.mark_dirty_tiles(stroke_points)
        
        # Incremental update (only analyzes dirty tiles)
        result = detector.detect(image)
    """
    
    def __init__(
        self,
        model=None,
        tflite_interpreter=None,
        is_tflite: bool = False,
        canvas_width: int = 512,
        canvas_height: int = 512,
        tile_size: int = 64,
        classification_threshold: float = 0.5,
        enable_caching: bool = True,
        padding_mode: str = 'clip'
    ):
        """
        Initialize tile detector.
        
        Args:
            model: Keras model for inference (or None if using TFLite)
            tflite_interpreter: TFLite interpreter (or None if using Keras)
            is_tflite: Whether using TFLite model
            canvas_width: Canvas width in pixels
            canvas_height: Canvas height in pixels
            tile_size: Size of each tile (32, 64, or 128 recommended)
            classification_threshold: Threshold for binary classification
            enable_caching: Whether to cache tile predictions
            padding_mode: How to handle non-divisible dimensions ('clip' or 'pad')
        """
        self.model = model
        self.tflite_interpreter = tflite_interpreter
        self.is_tflite = is_tflite
        self.classification_threshold = classification_threshold
        self.enable_caching = enable_caching
        
        # Initialize grid
        self.grid = TileGrid(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            tile_size=tile_size,
            padding_mode=padding_mode
        )
        
        # Dirty tile tracking
        self.dirty_tiles: Set[TileCoordinate] = set(self.grid.get_all_tiles())
        
        # Tile cache: coordinate -> (confidence, is_positive)
        self.tile_cache: Dict[TileCoordinate, Tuple[float, bool]] = {}
    
    def mark_dirty_tiles(self, stroke_points: List[Tuple[int, int]]):
        """
        Mark tiles as dirty based on stroke points.
        
        Args:
            stroke_points: List of (x, y) coordinates in the stroke
        """
        if not stroke_points:
            return
        
        # Calculate bounding box of stroke
        xs = [p[0] for p in stroke_points]
        ys = [p[1] for p in stroke_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Get all tiles intersecting the stroke bbox
        affected_tiles = self.grid.get_tiles_in_bbox(
            x_min, y_min,
            x_max - x_min + 1,
            y_max - y_min + 1
        )
        
        # Mark as dirty
        self.dirty_tiles.update(affected_tiles)
    
    def mark_all_dirty(self):
        """Mark all tiles as dirty (forces full re-analysis)."""
        self.dirty_tiles = set(self.grid.get_all_tiles())
    
    def clear_cache(self):
        """Clear the tile cache."""
        self.tile_cache.clear()
        self.mark_all_dirty()
    
    def reset(self):
        """Reset detector state (clear cache and mark all dirty)."""
        self.clear_cache()
    
    def extract_tile(
        self,
        image: np.ndarray,
        coord: TileCoordinate,
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Extract and preprocess a tile from the image.
        
        Args:
            image: Source image (H, W) or (H, W, C)
            coord: Tile coordinate
            target_size: Size to resize tile to (for model input)
        
        Returns:
            Preprocessed tile as numpy array (target_size[0], target_size[1], 1)
        """
        # Get tile bounding box
        x, y, w, h = self.grid.get_tile_bbox(coord)
        
        # Extract tile
        if len(image.shape) == 3:
            tile = image[y:y+h, x:x+w, 0]
        else:
            tile = image[y:y+h, x:x+w]
        
        # Validate tile is not empty before resizing
        if tile.size == 0 or h == 0 or w == 0:
            # Return empty tile filled with zeros
            tile = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
            tile = np.expand_dims(tile, axis=-1)
            return tile
        
        # Pad if necessary (for edge tiles)
        if w < self.grid.tile_size or h < self.grid.tile_size:
            padded = np.zeros((self.grid.tile_size, self.grid.tile_size), dtype=tile.dtype)
            padded[:h, :w] = tile
            tile = padded
        
        # Resize to target size (only if tile is valid)
        if tile.shape[0] > 0 and tile.shape[1] > 0:
            tile = cv2.resize(tile, target_size, interpolation=cv2.INTER_AREA)
        else:
            # Fallback to zeros if still invalid
            tile = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        
        # Normalize to [0, 1]
        tile = tile.astype(np.float32) / 255.0
        
        # Add channel dimension
        tile = np.expand_dims(tile, axis=-1)
        
        return tile
    
    def predict_tile(self, tile: np.ndarray) -> float:
        """
        Run inference on a single tile.
        
        Args:
            tile: Preprocessed tile (28, 28, 1)
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Add batch dimension
        tile_batch = np.expand_dims(tile, axis=0)
        
        # Run inference based on model type
        if self.is_tflite and self.tflite_interpreter is not None:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            
            input_array = tile_batch.astype(np.float32)
            self.tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
            self.tflite_interpreter.invoke()
            
            output_tensor = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            confidence = float(output_tensor[0][0])
        else:
            confidence = self.model.predict(tile_batch, verbose=0)[0][0]
            confidence = float(confidence)
        
        return confidence
    
    def detect(
        self,
        image: np.ndarray,
        force_full_analysis: bool = False
    ) -> TileDetectionResult:
        """
        Perform tile-based detection on an image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            force_full_analysis: If True, analyze all tiles (ignore cache)
        
        Returns:
            TileDetectionResult with tile-level and overall predictions
        """
        # Determine which tiles to analyze
        if force_full_analysis:
            tiles_to_analyze = self.grid.get_all_tiles()
        else:
            tiles_to_analyze = list(self.dirty_tiles)
        
        # Analyze tiles
        tile_predictions = []
        num_cached = 0
        
        for coord in tiles_to_analyze:
            # Check cache if enabled
            if self.enable_caching and not force_full_analysis and coord in self.tile_cache:
                confidence, is_positive = self.tile_cache[coord]
                is_cached = True
                num_cached += 1
            else:
                # Extract and predict
                tile = self.extract_tile(image, coord)
                confidence = self.predict_tile(tile)
                is_positive = confidence >= self.classification_threshold
                
                # Update cache
                if self.enable_caching:
                    self.tile_cache[coord] = (confidence, is_positive)
                
                is_cached = False
            
            # Create TileInfo
            bbox = self.grid.get_tile_bbox(coord)
            tile_info = TileInfo(
                coordinate=coord,
                bounding_box=bbox,
                confidence=confidence,
                is_positive=is_positive,
                is_cached=is_cached
            )
            tile_predictions.append(tile_info)
        
        # Clear dirty flags for analyzed tiles
        if not force_full_analysis:
            self.dirty_tiles.clear()
        
        # Aggregate results
        if tile_predictions:
            max_confidence = max(t.confidence for t in tile_predictions if t.confidence is not None)
            is_positive = any(t.is_positive for t in tile_predictions if t.is_positive is not None)
        else:
            max_confidence = 0.0
            is_positive = False
        
        return TileDetectionResult(
            is_positive=is_positive,
            confidence=float(max_confidence),
            tile_predictions=tile_predictions,
            num_tiles_analyzed=len(tiles_to_analyze),
            num_tiles_cached=num_cached,
            grid_dimensions=(self.grid.grid_rows, self.grid.grid_cols),
            tile_size=self.grid.tile_size,
            canvas_dimensions=(self.grid.canvas_width, self.grid.canvas_height)
        )
