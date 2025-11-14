"""
Single and batch prediction inference.
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple


class InferenceEngine:
    """Engine for model inference and predictions."""
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to the trained model
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = None
    
    def set_class_names(self, class_names: List[str]):
        """Set class names for predictions."""
        self.class_names = class_names
    
    def predict_single(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict class for a single image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with predictions
        """
        # Ensure correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)[0]
        
        result = {}
        if self.class_names:
            for i, prob in enumerate(predictions):
                result[self.class_names[i]] = float(prob)
        else:
            result = {f'class_{i}': float(prob) for i, prob in enumerate(predictions)}
        
        return result
    
    def predict_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """
        Predict classes for multiple images.
        
        Args:
            images: Batch of images as numpy array
        
        Returns:
            List of prediction dictionaries
        """
        predictions = self.model.predict(images, verbose=0)
        
        results = []
        for pred in predictions:
            result = {}
            if self.class_names:
                for i, prob in enumerate(pred):
                    result[self.class_names[i]] = float(prob)
            else:
                result = {f'class_{i}': float(prob) for i, prob in enumerate(pred)}
            results.append(result)
        
        return results
    
    def predict_top_k(
        self,
        image: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for an image.
        
        Args:
            image: Input image
            k: Number of top predictions
        
        Returns:
            List of (class_name, probability) tuples
        """
        predictions = self.predict_single(image)
        top_k = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k
