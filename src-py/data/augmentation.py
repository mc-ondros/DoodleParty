"""
Data augmentation pipeline for training.
"""

import numpy as np
import cv2
from typing import Callable


class DataAugmentation:
    """Data augmentation utilities."""
    
    @staticmethod
    def random_rotation(image: np.ndarray, angle_range: int = 15) -> np.ndarray:
        """
        Apply random rotation to image.
        
        Args:
            image: Input image
            angle_range: Maximum rotation angle
        
        Returns:
            Rotated image
        """
        angle = np.random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def random_shift(image: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
        """
        Apply random shift to image.
        
        Args:
            image: Input image
            shift_range: Maximum shift as fraction of image size
        
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        shift_h = int(h * np.random.uniform(-shift_range, shift_range))
        shift_w = int(w * np.random.uniform(-shift_range, shift_range))
        
        M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def random_scale(image: np.ndarray, scale_range: float = 0.2) -> np.ndarray:
        """
        Apply random scaling to image.
        
        Args:
            image: Input image
            scale_range: Maximum scale change
        
        Returns:
            Scaled image
        """
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad or crop to original size
        if new_h > h or new_w > w:
            resized = resized[:h, :w]
        else:
            padded = np.zeros_like(image)
            padded[:new_h, :new_w] = resized
            resized = padded
        
        return resized
    
    @staticmethod
    def random_noise(image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add random noise to image.
        
        Args:
            image: Input image
            noise_level: Noise level
        
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_level, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def augment_batch(
        batch: np.ndarray,
        augmentation_functions: list = None,
        p: float = 0.5
    ) -> np.ndarray:
        """
        Apply augmentations to a batch of images.
        
        Args:
            batch: Batch of images
            augmentation_functions: List of augmentation functions
            p: Probability of applying each augmentation
        
        Returns:
            Augmented batch
        """
        if augmentation_functions is None:
            augmentation_functions = [
                DataAugmentation.random_rotation,
                DataAugmentation.random_shift,
                DataAugmentation.random_scale,
                DataAugmentation.random_noise
            ]
        
        augmented = batch.copy()
        
        for func in augmentation_functions:
            if np.random.random() < p:
                for i in range(len(augmented)):
                    augmented[i] = func(augmented[i])
        
        return augmented
