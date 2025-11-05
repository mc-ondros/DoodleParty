"""
Patch extraction and sliding window detection for robustness improvements.

This module implements region-based detection to prevent content dilution attacks
where inappropriate content is mixed with innocent content. By analyzing multiple
patches of the canvas independently, we can detect suspicious content even when
diluted across the image.

Key Features:
- Sliding window patch extraction
- Adaptive patch selection (skip empty regions)
- Early stopping on first positive detection
- Batch inference support
- Multiple aggregation strategies

Related:
- src/core/inference.py (model inference)
- src/core/models.py (model architectures)

Exports:
- extract_patches, select_adaptive_patches, aggregate_predictions
- SlidingWindowDetector
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class AggregationStrategy(Enum):
    """Strategies for aggregating patch predictions."""
    MAX = 'max'  # Maximum confidence (most aggressive)
    MEAN = 'mean'  # Average confidence (balanced)
    WEIGHTED_MEAN = 'weighted_mean'  # Weighted by patch content
    VOTING = 'voting'  # Binary voting with threshold
    ANY_POSITIVE = 'any_positive'  # Flag if any patch is positive


@dataclass
class PatchInfo:
    """Information about an extracted patch."""
    patch: np.ndarray  # The image patch (H, W, C)
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int  # Patch width
    height: int  # Patch height
    content_ratio: float  # Ratio of non-empty pixels (0-1)
    index: int  # Patch index in the grid


@dataclass
class DetectionResult:
    """Result of region-based detection."""
    is_positive: bool  # Overall classification result
    confidence: float  # Overall confidence score
    patch_predictions: List[Dict]  # Individual patch predictions
    num_patches_analyzed: int  # Number of patches actually analyzed
    early_stopped: bool  # Whether early stopping was triggered
    aggregation_strategy: str  # Strategy used for aggregation


def extract_patches(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (128, 128),
    stride: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> List[PatchInfo]:
    """
    Extract patches from an image using sliding window approach.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        patch_size: Size of each patch (height, width)
        stride: Step size for sliding window. If None, uses patch_size (no overlap)
        normalize: Whether to normalize each patch independently
    
    Returns:
        List of PatchInfo objects containing patches and metadata
    """
    # Ensure image has channel dimension
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    height, width, channels = image.shape
    patch_h, patch_w = patch_size
    
    # Default stride is patch size (no overlap)
    if stride is None:
        stride = patch_size
    stride_h, stride_w = stride
    
    patches = []
    patch_idx = 0
    
    # Extract patches with sliding window
    for y in range(0, height - patch_h + 1, stride_h):
        for x in range(0, width - patch_w + 1, stride_w):
            # Extract patch
            patch = image[y:y + patch_h, x:x + patch_w, :]
            
            # Calculate content ratio (non-empty pixels)
            # Assuming empty regions are near 0 (black) or 1 (white) after normalization
            content_mask = (patch > 0.1) & (patch < 0.9)
            content_ratio = np.mean(content_mask)
            
            # Normalize patch if requested
            if normalize:
                patch = normalize_patch(patch)
            
            patches.append(PatchInfo(
                patch=patch,
                x=x,
                y=y,
                width=patch_w,
                height=patch_h,
                content_ratio=float(content_ratio),
                index=patch_idx
            ))
            patch_idx += 1
    
    return patches


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalize a patch to [0, 1] range.
    
    Args:
        patch: Input patch array
    
    Returns:
        Normalized patch
    """
    patch_min = np.min(patch)
    patch_max = np.max(patch)
    
    # Avoid division by zero
    if patch_max - patch_min < 1e-6:
        return np.zeros_like(patch)
    
    return (patch - patch_min) / (patch_max - patch_min)


def select_adaptive_patches(
    patches: List[PatchInfo],
    min_content_ratio: float = 0.05,
    max_patches: Optional[int] = None
) -> List[PatchInfo]:
    """
    Adaptively select patches based on content, skipping empty regions.
    
    This prevents wasting computation on empty/white regions and focuses
    on areas with actual drawing content.
    
    Args:
        patches: List of all extracted patches
        min_content_ratio: Minimum content ratio to consider (0-1)
        max_patches: Maximum number of patches to return (None for all)
    
    Returns:
        Filtered list of patches sorted by content ratio (descending)
    """
    # Filter patches with sufficient content
    content_patches = [
        p for p in patches 
        if p.content_ratio >= min_content_ratio
    ]
    
    # Sort by content ratio (descending) - analyze most dense regions first
    content_patches.sort(key=lambda p: p.content_ratio, reverse=True)
    
    # Limit number of patches if specified
    if max_patches is not None and len(content_patches) > max_patches:
        content_patches = content_patches[:max_patches]
    
    return content_patches


def aggregate_predictions(
    predictions: List[Dict],
    strategy: AggregationStrategy = AggregationStrategy.MAX,
    threshold: float = 0.5
) -> Tuple[float, bool]:
    """
    Aggregate predictions from multiple patches.
    
    Args:
        predictions: List of prediction dicts with 'confidence' and 'is_positive'
        strategy: Aggregation strategy to use
        threshold: Classification threshold (for voting and final decision)
    
    Returns:
        Tuple of (aggregated_confidence, is_positive)
    """
    if not predictions:
        return 0.0, False
    
    confidences = [p['confidence'] for p in predictions]
    
    if strategy == AggregationStrategy.MAX:
        # Most aggressive: take maximum confidence
        agg_confidence = max(confidences)
    
    elif strategy == AggregationStrategy.MEAN:
        # Balanced: average all confidences
        agg_confidence = np.mean(confidences)
    
    elif strategy == AggregationStrategy.WEIGHTED_MEAN:
        # Weight by content ratio
        weights = [p.get('content_ratio', 1.0) for p in predictions]
        agg_confidence = np.average(confidences, weights=weights)
    
    elif strategy == AggregationStrategy.VOTING:
        # Binary voting: majority wins
        positive_votes = sum(1 for p in predictions if p['is_positive'])
        vote_ratio = positive_votes / len(predictions)
        agg_confidence = vote_ratio
    
    elif strategy == AggregationStrategy.ANY_POSITIVE:
        # Flag if ANY patch is positive (most aggressive)
        agg_confidence = 1.0 if any(p['is_positive'] for p in predictions) else 0.0
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    is_positive = agg_confidence >= threshold
    return float(agg_confidence), is_positive


class SlidingWindowDetector:
    """
    Region-based detector using sliding window approach.
    
    This class provides a complete pipeline for:
    1. Extracting patches from a canvas
    2. Adaptively selecting patches with content
    3. Running batch inference on patches
    4. Aggregating predictions with early stopping
    5. Preventing content dilution attacks
    
    Example:
        detector = SlidingWindowDetector(
            model=model,
            patch_size=(128, 128),
            stride=(64, 64),
            min_content_ratio=0.05,
            early_stopping=True,
            aggregation_strategy=AggregationStrategy.MAX
        )
        
        result = detector.detect(canvas_image)
        print(f"Positive: {result.is_positive}, Confidence: {result.confidence}")
    """
    
    def __init__(
        self,
        model,
        patch_size: Tuple[int, int] = (128, 128),
        stride: Optional[Tuple[int, int]] = None,
        min_content_ratio: float = 0.05,
        max_patches: Optional[int] = 16,
        early_stopping: bool = True,
        early_stop_threshold: float = 0.9,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.MAX,
        classification_threshold: float = 0.5
    ):
        """
        Initialize sliding window detector.
        
        Args:
            model: Trained model for inference
            patch_size: Size of patches to extract
            stride: Sliding window stride (None = no overlap)
            min_content_ratio: Minimum content to analyze a patch
            max_patches: Maximum number of patches to analyze
            early_stopping: Whether to stop on first positive detection
            early_stop_threshold: Confidence threshold for early stopping
            aggregation_strategy: Strategy for combining patch predictions
            classification_threshold: Threshold for final classification
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.min_content_ratio = min_content_ratio
        self.max_patches = max_patches
        self.early_stopping = early_stopping
        self.early_stop_threshold = early_stop_threshold
        self.aggregation_strategy = aggregation_strategy
        self.classification_threshold = classification_threshold
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Perform region-based detection on an image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
        
        Returns:
            DetectionResult with classification and detailed patch info
        """
        # Step 1: Extract all patches
        all_patches = extract_patches(
            image,
            patch_size=self.patch_size,
            stride=self.stride,
            normalize=True
        )
        
        # Step 2: Adaptively select patches with content
        selected_patches = select_adaptive_patches(
            all_patches,
            min_content_ratio=self.min_content_ratio,
            max_patches=self.max_patches
        )
        
        if not selected_patches:
            # No patches with content - treat as negative
            return DetectionResult(
                is_positive=False,
                confidence=0.0,
                patch_predictions=[],
                num_patches_analyzed=0,
                early_stopped=False,
                aggregation_strategy=self.aggregation_strategy.value
            )
        
        # Step 3: Run inference with early stopping
        predictions = []
        early_stopped = False
        
        for i, patch_info in enumerate(selected_patches):
            # Prepare patch for inference
            patch_array = np.expand_dims(patch_info.patch, axis=0)
            
            # Run inference (single patch)
            confidence = self.model.predict(patch_array, verbose=0)[0][0]
            is_positive = confidence >= self.classification_threshold
            
            predictions.append({
                'patch_index': patch_info.index,
                'x': patch_info.x,
                'y': patch_info.y,
                'confidence': float(confidence),
                'is_positive': is_positive,
                'content_ratio': patch_info.content_ratio
            })
            
            # Early stopping: stop if high confidence positive detection
            if self.early_stopping and confidence >= self.early_stop_threshold:
                early_stopped = True
                break
        
        # Step 4: Aggregate predictions
        agg_confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=self.aggregation_strategy,
            threshold=self.classification_threshold
        )
        
        return DetectionResult(
            is_positive=is_positive,
            confidence=agg_confidence,
            patch_predictions=predictions,
            num_patches_analyzed=len(predictions),
            early_stopped=early_stopped,
            aggregation_strategy=self.aggregation_strategy.value
        )
    
    def detect_batch(self, image: np.ndarray) -> DetectionResult:
        """
        Perform region-based detection with batch inference.
        
        This is more efficient as all patches are processed in a single
        forward pass through the model.
        
        Args:
            image: Input image (H, W) or (H, W, C)
        
        Returns:
            DetectionResult with classification and detailed patch info
        """
        # Step 1: Extract all patches
        all_patches = extract_patches(
            image,
            patch_size=self.patch_size,
            stride=self.stride,
            normalize=True
        )
        
        # Step 2: Adaptively select patches with content
        selected_patches = select_adaptive_patches(
            all_patches,
            min_content_ratio=self.min_content_ratio,
            max_patches=self.max_patches
        )
        
        if not selected_patches:
            return DetectionResult(
                is_positive=False,
                confidence=0.0,
                patch_predictions=[],
                num_patches_analyzed=0,
                early_stopped=False,
                aggregation_strategy=self.aggregation_strategy.value
            )
        
        # Step 3: Batch inference (all patches in single forward pass)
        batch_array = np.array([p.patch for p in selected_patches])
        batch_confidences = self.model.predict(batch_array, verbose=0).flatten()
        
        # Step 4: Process results
        predictions = []
        early_stopped = False
        
        for i, (patch_info, confidence) in enumerate(zip(selected_patches, batch_confidences)):
            is_positive = confidence >= self.classification_threshold
            
            predictions.append({
                'patch_index': patch_info.index,
                'x': patch_info.x,
                'y': patch_info.y,
                'confidence': float(confidence),
                'is_positive': is_positive,
                'content_ratio': patch_info.content_ratio
            })
            
            # Check early stopping (but batch is already processed)
            if self.early_stopping and confidence >= self.early_stop_threshold:
                early_stopped = True
                # Note: In batch mode, all patches are already processed,
                # but we can still flag early stopping for reporting
        
        # Step 5: Aggregate predictions
        agg_confidence, is_positive = aggregate_predictions(
            predictions,
            strategy=self.aggregation_strategy,
            threshold=self.classification_threshold
        )
        
        return DetectionResult(
            is_positive=is_positive,
            confidence=agg_confidence,
            patch_predictions=predictions,
            num_patches_analyzed=len(predictions),
            early_stopped=early_stopped,
            aggregation_strategy=self.aggregation_strategy.value
        )


def visualize_detections(
    image: np.ndarray,
    detection_result: DetectionResult,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize patch-based detection results on the image.
    
    Args:
        image: Original image
        detection_result: Detection result with patch predictions
        save_path: Optional path to save visualization
    
    Returns:
        Visualization image with bounding boxes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    if len(image.shape) == 3 and image.shape[2] == 1:
        ax.imshow(image[:, :, 0], cmap='gray')
    else:
        ax.imshow(image, cmap='gray')
    
    # Draw bounding boxes for each patch
    for pred in detection_result.patch_predictions:
        x, y = pred['x'], pred['y']
        confidence = pred['confidence']
        is_positive = pred['is_positive']
        
        # Color based on prediction
        color = 'red' if is_positive else 'green'
        alpha = min(confidence, 0.8)
        
        # Draw rectangle
        rect = mpatches.Rectangle(
            (x, y), 128, 128,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Add confidence text
        ax.text(
            x + 5, y + 15,
            f'{confidence:.2f}',
            color=color,
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    ax.set_title(
        f'Detection: {"POSITIVE" if detection_result.is_positive else "NEGATIVE"} '
        f'(Confidence: {detection_result.confidence:.2f})\n'
        f'Patches analyzed: {detection_result.num_patches_analyzed}, '
        f'Early stopped: {detection_result.early_stopped}',
        fontsize=12,
        weight='bold'
    )
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Convert plot to numpy array
    fig.canvas.draw()
    vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return vis_array
