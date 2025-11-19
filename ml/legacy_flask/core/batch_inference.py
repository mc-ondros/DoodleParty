"""Optimized batch inference for multiple patches.

Provides efficient batch processing for analyzing multiple image patches
in a single forward pass, significantly improving throughput for region-based
detection.

Key Features:
- Batch patch extraction
- Parallel preprocessing
- Single forward pass for all patches
- Result aggregation strategies
- Memory-efficient processing

Related:
- src/core/patch_extraction.py (patch extraction)
- src/core/inference.py (single image inference)
- src/core/models.py (model architectures)

Exports:
- BatchInferenceEngine: Main batch processing class
- preprocess_batch: Batch preprocessing utilities
- aggregate_batch_results: Result aggregation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class BatchInferenceResult:
    """Results from batch inference."""
    predictions: np.ndarray
    confidences: np.ndarray
    processing_time_ms: float
    num_patches: int
    batch_size: int


class BatchInferenceEngine:
    """Efficient batch inference engine for processing multiple patches."""
    
    def __init__(self, model, batch_size: int = 16, input_shape: Tuple[int, int, int] = (128, 128, 1)):
        """Initialize batch inference engine."""
        self.model = model
        self.batch_size = batch_size
        self.input_shape = input_shape
    
    def preprocess_batch(self, patches: List[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of patches for inference."""
        batch = np.stack(patches, axis=0)
        
        if len(batch.shape) == 3:
            batch = np.expand_dims(batch, axis=-1)
        
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        
        return batch
    
    def predict_batch(self, patches: List[np.ndarray]) -> BatchInferenceResult:
        """Run inference on a batch of patches."""
        import time
        start_time = time.time()
        
        batch = self.preprocess_batch(patches)
        num_patches = len(patches)
        
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(batch, verbose=0)
        else:
            predictions = self.model(batch)
        
        predictions = predictions.flatten()
        confidences = np.abs(predictions - 0.5) * 2
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchInferenceResult(
            predictions=predictions,
            confidences=confidences,
            processing_time_ms=processing_time_ms,
            num_patches=num_patches,
            batch_size=self.batch_size
        )
    
    def predict_batches(self, patches: List[np.ndarray], show_progress: bool = False) -> BatchInferenceResult:
        """Process patches in multiple batches if needed."""
        import time
        start_time = time.time()
        
        all_predictions = []
        all_confidences = []
        
        for i in range(0, len(patches), self.batch_size):
            batch_patches = patches[i:i + self.batch_size]
            result = self.predict_batch(batch_patches)
            
            all_predictions.extend(result.predictions)
            all_confidences.extend(result.confidences)
            
            if show_progress:
                print(f"Processed {min(i + self.batch_size, len(patches))}/{len(patches)} patches")
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchInferenceResult(
            predictions=np.array(all_predictions),
            confidences=np.array(all_confidences),
            processing_time_ms=processing_time_ms,
            num_patches=len(patches),
            batch_size=self.batch_size
        )


def aggregate_batch_results(results: BatchInferenceResult, strategy: str = 'max', threshold: float = 0.5) -> Dict:
    """Aggregate batch inference results using specified strategy."""
    predictions = results.predictions
    
    if strategy == 'max':
        final_prediction = np.max(predictions)
    elif strategy == 'mean':
        final_prediction = np.mean(predictions)
    elif strategy == 'voting':
        votes = (predictions >= threshold).sum()
        final_prediction = votes / len(predictions)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    is_positive = final_prediction >= threshold
    confidence = final_prediction if is_positive else (1 - final_prediction)
    
    return {
        'is_positive': bool(is_positive),
        'confidence': float(confidence),
        'final_prediction': float(final_prediction),
        'num_patches': results.num_patches,
        'processing_time_ms': results.processing_time_ms,
        'strategy': strategy
    }


if __name__ == '__main__':
    print('Batch Inference Engine')
    print('Use with SlidingWindowDetector for region-based detection.')
