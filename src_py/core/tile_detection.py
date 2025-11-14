"""
Tile-based grid detection for image processing.
"""

import numpy as np
from typing import List, Tuple
import cv2


class TileDetector:
    """Detect and process image tiles."""
    
    @staticmethod
    def create_tiles(image: np.ndarray, tile_size: int = 32) -> List[np.ndarray]:
        """
        Split image into tiles.
        
        Args:
            image: Input image
            tile_size: Size of each tile
        
        Returns:
            List of tile arrays
        """
        tiles = []
        height, width = image.shape[:2]
        
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
        
        return tiles
    
    @staticmethod
    def reconstruct_from_tiles(tiles: List[np.ndarray], image_shape: Tuple[int, int], tile_size: int = 32) -> np.ndarray:
        """
        Reconstruct image from tiles.
        
        Args:
            tiles: List of tiles
            image_shape: Original image shape
            tile_size: Size of each tile
        
        Returns:
            Reconstructed image
        """
        height, width = image_shape
        image = np.zeros((height, width), dtype=tiles[0].dtype)
        
        tile_idx = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if tile_idx < len(tiles):
                    tile = tiles[tile_idx]
                    image[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
                    tile_idx += 1
        
        return image
    
    @staticmethod
    def get_tile_features(tile: np.ndarray) -> dict:
        """
        Extract features from a single tile.
        
        Args:
            tile: Tile array
        
        Returns:
            Dictionary of tile features
        """
        return {
            'mean': np.mean(tile),
            'std': np.std(tile),
            'min': np.min(tile),
            'max': np.max(tile),
            'sum': np.sum(tile),
            'coverage': np.count_nonzero(tile) / tile.size if tile.size > 0 else 0
        }
