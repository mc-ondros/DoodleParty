"""
Dataset loading for QuickDraw and other sources.
"""

import numpy as np
import os
import cv2
from pathlib import Path
from typing import Tuple, List, Optional


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


class QuickDrawAppendixLoader:
    """Load QuickDraw Appendix dataset (128x128 images) and downscale to 28x28."""

    @staticmethod
    def _prepare_npy_images(data: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """Ensure npy batches are shaped/normalized as 28x28 inputs."""
        if data.size == 0:
            return data.reshape(0, target_size[0], target_size[1], 1)

        data = data.astype(np.float32)

        if data.ndim == 2:
            pixels = data.shape[1]
            side = int(np.sqrt(pixels))
            if side * side != pixels:
                raise ValueError(f"Cannot reshape flattened sample of length {pixels} into a square image")
            data = data.reshape(-1, side, side)
        elif data.ndim == 3:
            pass
        elif data.ndim == 4:
            if data.shape[-1] != 1:
                data = np.mean(data, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported npy data shape: {data.shape}")

        if data.ndim == 3:
            data = np.expand_dims(data, axis=-1)

        current_h, current_w = data.shape[1:3]
        if (current_h, current_w) != target_size:
            resized = np.empty((data.shape[0], target_size[0], target_size[1], 1), dtype=np.float32)
            for idx in range(data.shape[0]):
                resized[idx, ..., 0] = cv2.resize(data[idx, ..., 0], target_size, interpolation=cv2.INTER_AREA)
            data = resized

        max_val = data.max()
        if max_val > 1.0:
            data = data / 255.0

        return data
    
    @staticmethod
    def load_appendix_images(
        data_dir: str,
        category: str,
        max_samples: Optional[int] = None,
        target_size: Tuple[int, int] = (28, 28)
    ) -> np.ndarray:
        """
        Load QuickDraw Appendix images (128x128) and downscale to 28x28.
        
        Args:
            data_dir: Directory containing image files
            category: Category subdirectory name
            max_samples: Maximum number of samples to load (None for all)
            target_size: Target size to resize to (default 28x28)
        
        Returns:
            Array of shape (N, 28, 28, 1) normalized to [0, 1]
        """
        category_dir = Path(data_dir) / category
        
        if not category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")
        
        # Find all image files (png, jpg, jpeg)
        image_files = sorted(list(category_dir.glob('*.png')) + 
                           list(category_dir.glob('*.jpg')) + 
                           list(category_dir.glob('*.jpeg')))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        images = []
        for img_path in image_files:
            try:
                # Load as grayscale
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Downscale from 128x128 to 28x28 using INTER_AREA (best for downscaling)
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Normalize to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                images.append(img_normalized)
                
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found in {category_dir}")
        
        # Stack and add channel dimension
        images_array = np.array(images)
        if images_array.ndim == 3:
            images_array = np.expand_dims(images_array, axis=-1)
        
        return images_array
    
    @staticmethod
    def load_appendix_category(
        data_dir: str,
        category: str,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load QuickDraw Appendix category with labels.
        
        Args:
            data_dir: Directory containing category subdirectories
            category: Category name
            max_samples: Maximum samples to load
        
        Returns:
            Tuple of (images, labels) where labels are all 1.0 for positive class
        """
        images = QuickDrawAppendixLoader.load_appendix_images(
            data_dir, category, max_samples
        )
        
        # Create labels (assuming appendix is all positive examples)
        labels = np.ones(len(images), dtype=np.float32)
        
        return images, labels
    
    @staticmethod
    def load_mixed_dataset(
        npy_data_dir: str,
        appendix_data_dir: str,
        positive_category: str = "penis",
        negative_categories: Optional[List[str]] = None,
        max_npy_samples: int = 10000,
        max_appendix_samples: int = 5000,
        train_split: float = 0.8
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load mixed dataset from both QuickDraw .npy (28x28) and Appendix (128x128→28x28).
        
        Args:
            npy_data_dir: Directory with .npy files
            appendix_data_dir: Directory with appendix image subdirectories
            positive_category: Positive class name
            negative_categories: List of negative class names
            max_npy_samples: Max samples per category from .npy
            max_appendix_samples: Max samples from appendix
            train_split: Train/validation split ratio
        
        Returns:
            ((train_images, train_labels), (val_images, val_labels))
        """
        if negative_categories is None:
            negative_categories = [
                "circle", "line", "square", "triangle", "star",
                "rectangle", "diamond", "heart", "cloud", "moon"
            ]
        
        all_images = []
        all_labels = []
        
        # Load positive samples from .npy
        npy_path = os.path.join(npy_data_dir, f"{positive_category}.npy")
        if os.path.exists(npy_path):
            data = np.load(npy_path)
            data = data[:max_npy_samples]
            data = QuickDrawAppendixLoader._prepare_npy_images(data, target_size=(28, 28))
            all_images.append(data)
            all_labels.append(np.ones(len(data), dtype=np.float32))
        
        # Load positive samples from appendix
        try:
            appendix_images, appendix_labels = QuickDrawAppendixLoader.load_appendix_category(
                appendix_data_dir, positive_category, max_appendix_samples
            )
            all_images.append(appendix_images)
            all_labels.append(appendix_labels)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load appendix data: {e}")
        
        # Load negative samples from .npy
        for category in negative_categories:
            npy_path = os.path.join(npy_data_dir, f"{category}.npy")
            if os.path.exists(npy_path):
                data = np.load(npy_path)
                data = data[:max_npy_samples]
                data = QuickDrawAppendixLoader._prepare_npy_images(data, target_size=(28, 28))
                all_images.append(data)
                all_labels.append(np.zeros(len(data), dtype=np.float32))
        
        # Combine all data
        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = all_images[indices]
        all_labels = all_labels[indices]
        
        # Split train/val
        split_idx = int(len(all_images) * train_split)
        train_images = all_images[:split_idx]
        train_labels = all_labels[:split_idx]
        val_images = all_images[split_idx:]
        val_labels = all_labels[split_idx:]
        
        return (train_images, train_labels), (val_images, val_labels)
