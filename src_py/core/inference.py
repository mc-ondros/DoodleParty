"""
Inference Module - Neural Network Predictions

Provides both single and batch prediction capabilities for trained models.
Handles image preprocessing, model loading, and result formatting.

Key Components:
- InferenceEngine: Main class that loads models and performs predictions
- Supported prediction types: single image, batch, top-k results

Examples:
    >>> engine = InferenceEngine("path/to/model.h5")
    >>> result = engine.predict_single(image_array)
    >>> print(result)  # {'circle': 0.85, 'square': 0.15}

Related Modules:
- src-py/core/training.py (model training)
- src-py/data/loaders.py (data loading utilities)
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, TypedDict

# Type aliases for clearer type hints
PredictionDict = Dict[str, float]
BatchPredictions = List[PredictionDict]
TopKPredictions = List[Tuple[str, float]]


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
    
    def predict_batch(self, images: np.ndarray) -> BatchPredictions:
        """
        Predict class probabilities for a batch of images.
        
        Args:
            images: Batch of images as numpy array with shape (N, H, W, C)
                    Where N is number of images
        
        Returns:
            List of dictionaries mapping class names to probabilities
            
        Example:
            >>> batch = np.array([image1, image2, image3])
            >>> results = engine.predict_batch(batch)
            >>> print(results[0])
            {'circle': 0.95, 'square': 0.05}
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
    ) -> TopKPredictions:
        """
        Get top-k highest probability predictions for an image.
        
        Args:
            image: Input image as numpy array
            k: Number of top predictions to return (default: 5)
        
        Returns:
            Ordered list of (class_name, probability) tuples,
            sorted descending by probability
            
        Example:
            >>> top3 = engine.predict_top_k(image_array, k=3)
            >>> print(top3)
            [('circle', 0.92), ('triangle', 0.07), ('square', 0.01)]
        """
        predictions = self.predict_single(image)
        top_k = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k
