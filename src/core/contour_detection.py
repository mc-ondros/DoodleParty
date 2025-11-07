"""
Contour-based detection system for DoodleHunter.

This module implements contour-based detection using OpenCV to find and classify
individual shapes in drawings. It provides two modes:
1. RETR_EXTERNAL: Finds only outer boundaries (current limitation)
2. RETR_TREE: Full hierarchy including nested content (solution)

Why contour-based detection:
- Isolates individual shapes for independent classification
- More efficient than sliding window when shapes are well-separated
- Prevents content dilution attacks by analyzing each shape separately
- Enables hierarchical detection of nested content (offensive inside benign)

Related:
- src/web/app.py (Flask API endpoints)
- src/core/inference.py (model inference pipeline)
- src/core/patch_extraction.py (sliding window alternative)

Exports:
- ContourDetector: Main class for contour-based detection
- detect_contours: Function for simple contour detection
- ContourResult: Result dataclass
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.core.patch_extraction import DetectionResult


class ContourRetrievalMode(Enum):
    """OpenCV contour retrieval modes."""
    EXTERNAL = 'external'  # Only outer boundaries (current limitation)
    TREE = 'tree'  # Full hierarchy (handles nested content)


@dataclass
class ContourInfo:
    """Information about a detected contour."""
    contour: np.ndarray  # Contour points
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    area: float  # Contour area
    hierarchy_level: int  # Nesting level (0 = outer boundary)
    parent_id: Optional[int]  # Parent contour index (None for root)
    confidence: Optional[float]  # Classification confidence
    is_positive: Optional[bool]  # Classification result


@dataclass
class ContourDetectionResult:
    """Result of contour-based detection."""
    is_positive: bool  # Overall classification result
    confidence: float  # Overall confidence score
    contour_predictions: List[ContourInfo]  # Individual contour predictions
    num_contours_analyzed: int  # Number of contours analyzed
    retrieval_mode: str  # EXTERNAL or TREE
    early_stopped: bool  # Whether early stopping was triggered


def detect_contours(
    image: np.ndarray,
    mode: ContourRetrievalMode = ContourRetrievalMode.EXTERNAL
) -> Tuple[List[ContourInfo], np.ndarray]:
    """
    Detect contours in an image using OpenCV.

    Args:
        image: Input image (H, W) or (H, W, C), should be binary or grayscale
        mode: Contour retrieval mode (EXTERNAL or TREE)

    Returns:
        Tuple of (contour_list, hierarchy) where:
            - contour_list: List of ContourInfo objects
            - hierarchy: OpenCV hierarchy array

    Raises:
        ValueError: If image is invalid
    """
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image

    # Ensure image is binary
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    # Note: RETR_EXTERNAL only finds outer boundaries
    # RETR_TREE finds full hierarchy including nested contours
    retrieval_mode = cv2.RETR_EXTERNAL if mode == ContourRetrievalMode.EXTERNAL else cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(
        binary,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process contours into ContourInfo objects
    contour_list = []
    if hierarchy is not None and len(contours) > 0:
        # hierarchy[0] gives us the hierarchy array
        # Format: [next, previous, first_child, parent]
        for idx, contour in enumerate(contours):
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Get hierarchy information
            # hierarchy[0][idx] gives the hierarchy for this contour
            h_info = hierarchy[0][idx]

            # Determine nesting level and parent
            # We need to traverse up to count nesting levels
            level = 0
            parent = h_info[3]  # Parent index

            if mode == ContourRetrievalMode.TREE and parent >= 0:
                # Count nesting levels by traversing parent chain
                temp_parent = parent
                while temp_parent >= 0 and temp_parent < len(hierarchy[0]):
                    level += 1
                    temp_parent = hierarchy[0][temp_parent][3]
                    # Safety check to prevent infinite loops
                    if level > 100:  # Arbitrary limit
                        break

            contour_info = ContourInfo(
                contour=contour,
                bounding_box=(x, y, w, h),
                area=area,
                hierarchy_level=level,
                parent_id=parent if parent >= 0 else None,
                confidence=None,
                is_positive=None
            )
            contour_list.append(contour_info)

    return contour_list, hierarchy


def extract_contour_patches(
    image: np.ndarray,
    contour: ContourInfo,
    target_size: Tuple[int, int] = (128, 128),
    padding: int = 10
) -> np.ndarray:
    """
    Extract and normalize a patch from a specific contour.

    Args:
        image: Source image
        contour: ContourInfo object with bounding box
        target_size: Size to resize patch to
        padding: Additional padding around contour

    Returns:
        Normalized patch as numpy array (target_size, target_size, 1)
    """
    x, y, w, h = contour.bounding_box

    # Add padding, but don't go outside image boundaries
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    # Extract patch
    if len(image.shape) == 3:
        patch = image[y1:y2, x1:x2, 0]
    else:
        patch = image[y1:y2, x1:x2]

    # Resize to target size
    patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    patch = patch.astype(np.float32) / 255.0

    # Add channel dimension
    patch = np.expand_dims(patch, axis=-1)

    return patch


class ContourDetector:
    """
    Contour-based detector for shape classification.

    This class provides contour extraction, classification, and hierarchical
    analysis to detect inappropriate content even when nested inside benign shapes.

    Key Features:
    - RETR_EXTERNAL mode: Only analyzes outer boundaries (faster, may miss nested)
    - RETR_TREE mode: Full hierarchy analysis (handles nested content)
    - Early stopping on high-confidence detection
    - Configurable contour filtering (area, aspect ratio, etc.)

    Example:
        detector = ContourDetector(
            model=model,
            tflite_interpreter=interpreter,
            is_tflite=False,
            retrieval_mode=ContourRetrievalMode.TREE,
            min_contour_area=100,
            early_stopping=True
        )
        result = detector.detect(image)
    """

    def __init__(
        self,
        model=None,
        tflite_interpreter=None,
        is_tflite: bool = False,
        retrieval_mode: ContourRetrievalMode = ContourRetrievalMode.EXTERNAL,
        min_contour_area: int = 100,
        max_contours: int = 50,
        early_stopping: bool = True,
        early_stop_threshold: float = 0.9,
        classification_threshold: float = 0.5
    ):
        """
        Initialize contour detector.

        Args:
            model: Keras model for inference (or None if using TFLite)
            tflite_interpreter: TFLite interpreter for inference (or None)
            is_tflite: Whether using TFLite model
            retrieval_mode: Contour retrieval mode (EXTERNAL or TREE)
            min_contour_area: Minimum contour area to analyze
            max_contours: Maximum number of contours to analyze
            early_stopping: Whether to stop on first high-confidence detection
            early_stop_threshold: Confidence threshold for early stopping
            classification_threshold: Threshold for final classification
        """
        self.model = model
        self.tflite_interpreter = tflite_interpreter
        self.is_tflite = is_tflite
        self.retrieval_mode = retrieval_mode
        self.min_contour_area = min_contour_area
        self.max_contours = max_contours
        self.early_stopping = early_stopping
        self.early_stop_threshold = early_stop_threshold
        self.classification_threshold = classification_threshold

    def detect(
        self,
        image: np.ndarray,
        normalize_input: bool = False
    ) -> ContourDetectionResult:
        """
        Perform contour-based detection on an image.

        Args:
            image: Input image (H, W) or (H, W, C)
            normalize_input: Whether to normalize the image before contour detection

        Returns:
            ContourDetectionResult with classification and detailed contour info
        """
        # Normalize input if requested
        if normalize_input and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3:
            gray = image[:, :, 0]
        else:
            gray = image

        # Step 1: Detect contours
        contours, hierarchy = detect_contours(gray, self.retrieval_mode)

        # Step 2: Filter contours by area
        filtered_contours = [
            c for c in contours
            if c.area >= self.min_contour_area
        ]

        # Step 3: Limit number of contours
        if len(filtered_contours) > self.max_contours:
            # Sort by area (descending) and take top N
            filtered_contours.sort(key=lambda c: c.area, reverse=True)
            filtered_contours = filtered_contours[:self.max_contours]

        # Step 4: Classify each contour
        predictions = []
        early_stopped = False

        for contour in filtered_contours:
            # Extract patch for this contour
            patch = extract_contour_patches(image, contour)

            # Add batch dimension
            patch_batch = np.expand_dims(patch, axis=0)

            # Run inference based on model type
            if self.is_tflite and self.tflite_interpreter is not None:
                # Use TFLite inference
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()

                # Set input tensor
                input_array = patch_batch.astype(np.float32)
                self.tflite_interpreter.set_tensor(input_details[0]['index'], input_array)

                # Invoke interpreter
                self.tflite_interpreter.invoke()

                # Get output tensor
                output_tensor = self.tflite_interpreter.get_tensor(output_details[0]['index'])
                confidence = float(output_tensor[0][0])
            else:
                # Use Keras model inference
                confidence = self.model.predict(patch_batch, verbose=0)[0][0]
                confidence = float(confidence)

            is_positive = confidence >= self.classification_threshold

            # Update contour info
            contour.confidence = confidence
            contour.is_positive = is_positive
            predictions.append(contour)

            # Early stopping: stop if high confidence positive detection
            if self.early_stopping and confidence >= self.early_stop_threshold:
                early_stopped = True
                break

        # Step 5: Determine overall result
        # Use max confidence across all contours
        if predictions:
            max_confidence = max(p.confidence for p in predictions if p.confidence is not None)
            is_positive = any(p.is_positive for p in predictions if p.is_positive is not None)
        else:
            max_confidence = 0.0
            is_positive = False

        return ContourDetectionResult(
            is_positive=is_positive,
            confidence=float(max_confidence),
            contour_predictions=predictions,
            num_contours_analyzed=len(predictions),
            retrieval_mode=self.retrieval_mode.value,
            early_stopped=early_stopped
        )

    def detect_hierarchical(
        self,
        image: np.ndarray,
        normalize_input: bool = False
    ) -> ContourDetectionResult:
        """
        Perform hierarchical contour detection with nested content analysis.

        This method specifically handles the case where offensive content
        is nested inside benign shapes (e.g., offensive drawing inside a circle).

        Only available when retrieval_mode is TREE.

        Args:
            image: Input image (H, W) or (H, W, C)
            normalize_input: Whether to normalize the image before detection

        Returns:
            ContourDetectionResult with hierarchical analysis
        """
        if self.retrieval_mode != ContourRetrievalMode.TREE:
            raise ValueError(
                "Hierarchical detection requires retrieval_mode=TREE. "
                f"Current mode: {self.retrieval_mode.value}"
            )

        # Perform detection (already handles hierarchy)
        result = self.detect(image, normalize_input)

        # Step 6: Additional hierarchical analysis
        # Check for nested positive detections inside negative containers
        for idx, contour in enumerate(result.contour_predictions):
            if contour.confidence is None or contour.is_positive is None:
                continue

            # Find child contours by checking if their parent_id matches current index
            children = [
                c for c in result.contour_predictions
                if c.parent_id is not None and c.parent_id == idx
            ]

            # Check if any children are positive while parent is negative
            if children:
                has_positive_child = any(c.is_positive for c in children if c.is_positive is not None)
                parent_is_positive = contour.is_positive

                # If parent is negative but has positive child, this is a containment case
                if not parent_is_positive and has_positive_child:
                    # This is a security concern - inappropriate content hidden in benign shape
                    # We can optionally increase confidence or flag this
                    contour.confidence = max(contour.confidence or 0.0, 0.5)
                    contour.is_positive = True

        # Recalculate overall result after hierarchical analysis
        if result.contour_predictions:
            max_confidence = max(
                p.confidence for p in result.contour_predictions
                if p.confidence is not None
            )
            is_positive = any(
                p.is_positive for p in result.contour_predictions
                if p.is_positive is not None
            )
        else:
            max_confidence = 0.0
            is_positive = False

        result.confidence = float(max_confidence)
        result.is_positive = is_positive

        return result


def visualize_contours(
    image: np.ndarray,
    result: ContourDetectionResult,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize contour detection results on the image.

    Args:
        image: Original image
        result: Contour detection result
        save_path: Optional path to save visualization

    Returns:
        Visualization image with contour overlays
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    # Create visualization
    if len(image.shape) == 3 and image.shape[2] == 1:
        vis_img = image[:, :, 0]
    else:
        vis_img = image

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vis_img, cmap='gray')

    # Draw contours
    for contour in result.contour_predictions:
        if contour.confidence is None or contour.is_positive is None:
            continue

        # Color based on prediction
        color = 'red' if contour.is_positive else 'green'
        alpha = min(contour.confidence, 0.8)

        # Draw contour
        contour_draw = contour.contour
        if len(contour_draw.shape) == 3 and contour_draw.shape[1] == 1:
            contour_draw = contour_draw.reshape(-1, 2)

        ax.plot(contour_draw[:, 0], contour_draw[:, 1], color=color, linewidth=2, alpha=alpha)

        # Add bounding box
        x, y, w, h = contour.bounding_box
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            alpha=alpha
        )
        ax.add_patch(rect)

        # Add confidence text
        ax.text(
            x + 5, y + 15,
            f'{contour.confidence:.2f}',
            color=color,
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    ax.set_title(
        f'Contour Detection ({result.retrieval_mode.upper()})\n'
        f'Detection: {"POSITIVE" if result.is_positive else "NEGATIVE"} '
        f'(Confidence: {result.confidence:.2f})\n'
        f'Contours analyzed: {result.num_contours_analyzed}, '
        f'Early stopped: {result.early_stopped}',
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
