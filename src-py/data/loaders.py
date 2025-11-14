"""
Dataset loading for QuickDraw and other sources.
"""

import numpy as np
import os
from typing import Tuple, List


class QuickDrawLoader:
    """Load QuickDraw dataset."""
    
    @staticmethod
    def load_quickdraw_npy(data_dir: str, categories: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load QuickDraw data from .npy files.
        
        Args:
            data_dir: Directory containing .npy files
            categories: List of category names to load
        
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for idx, category in enumerate(categories):
            file_path = os.path.join(data_dir, f'{category}.npy')
            if os.path.exists(file_path):
                data = np.load(file_path)
                # Normalize to 0-1 range
                data = data.astype(np.float32) / 255.0
                
                images.extend(data)
                labels.extend([idx] * len(data))
        
        return np.array(images), np.array(labels)
    
    @staticmethod
    def load_quickdraw_split(
        data_dir: str,
        categories: List[str],
        train_split: float = 0.8
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and split QuickDraw data.
        
        Args:
            data_dir: Directory containing .npy files
            categories: List of category names to load
            train_split: Fraction for training (0-1)
        
        Returns:
            Tuple of ((train_images, train_labels), (val_images, val_labels))
        """
        images, labels = QuickDrawLoader.load_quickdraw_npy(data_dir, categories)
        
        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        # Split
        split_idx = int(len(images) * train_split)
        
        train_images = images[:split_idx]
        train_labels = labels[:split_idx]
        val_images = images[split_idx:]
        val_labels = labels[split_idx:]
        
        return (train_images, train_labels), (val_images, val_labels)
    
    @staticmethod
    def load_batch(data_dir: str, category: str, batch_size: int = 32):
        """
        Generator to load data in batches.
        
        Args:
            data_dir: Directory containing .npy files
            category: Category name to load
            batch_size: Batch size
        
        Yields:
            Batches of data
        """
        file_path = os.path.join(data_dir, f'{category}.npy')
        if os.path.exists(file_path):
            data = np.load(file_path).astype(np.float32) / 255.0
            
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
