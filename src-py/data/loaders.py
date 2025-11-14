"""
Dataset loading for QuickDraw and other sources.
"""

import numpy as np
import os
from typing import Tuple, List


class QuickDrawLoader:
    """Load QuickDraw dataset."""
    
    @staticmethod
    def load_quickdraw_npy(data_dir: str, categories: List[str], target_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load QuickDraw data from .npy files.
        
        Args:
            data_dir: Directory containing .npy files
            categories: List of category names to load
            target_size: Target image size (default 128x128)
        
        Returns:
            Tuple of (images, labels)
        """
        from PIL import Image
        
        images = []
        labels = []
        
        for idx, category in enumerate(categories):
            file_path = os.path.join(data_dir, f'{category}.npy')
            if os.path.exists(file_path):
                data = np.load(file_path)
                
                # Determine original size from flattened array
                # Google QuickDraw: 784 = 28x28
                # Quickdraw-appendix: 16384 = 128x128
                pixels_per_image = data.shape[1] if len(data.shape) > 1 else data.size
                
                if pixels_per_image == 784:  # 28x28
                    original_size = 28
                elif pixels_per_image == 16384:  # 128x128
                    original_size = 128
                else:
                    # Try to infer square size
                    original_size = int(np.sqrt(pixels_per_image))
                    if original_size * original_size != pixels_per_image:
                        raise ValueError(f"Cannot determine image size for {category}.npy with {pixels_per_image} pixels")
                
                # Reshape to square images
                data = data.reshape(-1, original_size, original_size)
                
                # Resize if needed
                if original_size != target_size:
                    print(f"  Resizing {category} from {original_size}x{original_size} to {target_size}x{target_size}...")
                    resized_data = []
                    for img in data:
                        pil_img = Image.fromarray(img.astype(np.uint8))
                        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        resized_data.append(np.array(pil_img))
                    data = np.array(resized_data)
                
                # Flatten back for storage
                data = data.reshape(-1, target_size * target_size)
                
                # Normalize to 0-1 range
                data = data.astype(np.float32) / 255.0
                
                images.extend(data)
                labels.extend([idx] * len(data))
                
                print(f"  Loaded {len(data)} images from {category}.npy")
        
        return np.array(images), np.array(labels)
    
    @staticmethod
    def load_quickdraw_split(
        data_dir: str,
        categories: List[str],
        train_split: float = 0.8,
        target_size: int = 128
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and split QuickDraw data.
        
        Args:
            data_dir: Directory containing .npy files
            categories: List of category names to load
            train_split: Fraction for training (0-1)
            target_size: Target image size (default 128x128)
        
        Returns:
            Tuple of ((train_images, train_labels), (val_images, val_labels))
        """
        images, labels = QuickDrawLoader.load_quickdraw_npy(data_dir, categories, target_size)
        
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
