"""
Normalization and preprocessing utilities for shape-based detection.

Extracted from src/core/shape_detection.py to keep transformation logic isolated.

Public API:
- normalize_shape(...)
- preprocess_for_model(...)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

from .shape_types import PADDING_COLOR_BG

logger = logging.getLogger(__name__)


def normalize_shape(
    image: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    target_size: int,
    padding_color: int = PADDING_COLOR_BG,
) -> np.ndarray:
    """
    Extract shape region and normalize with padding.

    Rules:
    - Normalize to a square of `target_size`.
    - Add a generous margin around the shape to capture multi-part objects.
    - Preserve aspect ratio during resize.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    target_size = int(target_size)

    if (
        not isinstance(bounding_box, (tuple, list))
        or len(bounding_box) != 4
    ):
        raise ValueError("bounding_box must be a 4-tuple (x, y, w, h)")

    x, y, w, h = map(int, bounding_box)

    # Add minimal margin matching training data: ~8% of max(w, h), min 16px.
    # Training data uses ~20px padding on 256x256, which is ~8% margin.
    # This keeps drawings large enough for accurate model recognition.
    margin = max(int(0.08 * max(w, h)), 16)

    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(image.shape[1], x + w + margin)
    y_end = min(image.shape[0], y + h + margin)

    # If box collapses (e.g. near edges), fall back to full canvas.
    if (x_end - x_start) < max(8, w) or (y_end - y_start) < max(8, h):
        x_start, y_start = 0, 0
        x_end, y_end = image.shape[1], image.shape[0]

    shape_region = image[y_start:y_end, x_start:x_end].copy()

    # Handle empty / degenerate regions
    if (
        shape_region.size == 0
        or shape_region.shape[0] <= 1
        or shape_region.shape[1] <= 1
    ):
        return np.full(
            (target_size, target_size),
            padding_color,
            dtype=np.uint8,
        )

    region_h, region_w = shape_region.shape[:2]

    # Use 98% of canvas to maximize drawing size (leave 1% margin on each side)
    # Training data centers shapes with minimal padding
    inner_size = int(target_size * 0.98)
    scale = min(inner_size / float(region_w), inner_size / float(region_h))
    scale = max(scale, 1e-3)

    new_w = max(1, int(round(region_w * scale)))
    new_h = max(1, int(round(region_h * scale)))

    resized = cv2.resize(
        shape_region,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    normalized = np.full(
        (target_size, target_size),
        padding_color,
        dtype=np.uint8,
    )

    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    normalized[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return normalized


def preprocess_for_model(
    image: np.ndarray,
    target_size: Optional[int],
    skeletonize: bool = True,
) -> np.ndarray:
    """
    Preprocess normalized shape for model input.
    
    Args:
        image: Input image (grayscale or RGB)
        target_size: Target size for model input (e.g. 128)
        skeletonize: If True, extract thin skeleton from thick/filled shapes
                    to match QuickDraw line-drawing style (default: True)

    Returns:
        Array of shape (1, H, W, 1) with values in [0, 1].
    """
    if target_size is None:
        raise ValueError("target_size must be provided")

    target_h = target_w = int(target_size)

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if image.ndim == 3:
        # If RGB/BGR, convert to grayscale
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError("Unsupported channel configuration for model input")

    if image.shape[:2] != (target_h, target_w):
        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
    
    # CRITICAL FIX: QuickDraw model was trained on thin line drawings,
    # not thick/filled shapes. If the input has thick strokes, we extract
    # the skeleton (thin centerline) to match the training data style.
    # HOWEVER: Simple geometric shapes (circles, squares) should NOT be
    # skeletonized as they become false positives.
    if skeletonize:
        # Check if we have thick strokes (many bright pixels)
        bright_ratio = np.sum(image > 127) / image.size
        
        # If >5% of pixels are bright, likely a thick/filled shape
        # (lowered from 10% to account for padding added by normalize_shape)
        if bright_ratio > 0.05:
            # Binarize first
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Check if this is a simple geometric shape (circle, square, triangle)
            # by analyzing contours and their circularity/rectangularity
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            is_geometric = False
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    # Circularity: 4π * area / perimeter^2 (1.0 = perfect circle)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Check for rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bbox_area = w * h
                    extent = area / bbox_area if bbox_area > 0 else 0
                    aspect_ratio = max(w, h) / max(min(w, h), 1)
                    
                    # Simple geometric shapes have:
                    # - High circularity (>0.7 for circles/ovals)
                    # - High extent (>0.7 for rectangles/squares)  
                    # - Low vertex count AND high convexity (for triangles, pentagons)
                    if circularity > 0.70:  # Circle/oval
                        is_geometric = True
                        logger.debug(f"Detected circle/oval (circularity={circularity:.3f}), skipping skeletonization")
                    elif extent > 0.75 and aspect_ratio < 3.0:  # Rectangle/square
                        is_geometric = True
                        logger.debug(f"Detected rectangle/square (extent={extent:.3f}), skipping skeletonization")
                    elif len(largest_contour) > 0:
                        # Check for regular polygon (triangle, pentagon, hexagon)
                        # Must have both few vertices AND high extent/convexity
                        epsilon = 0.04 * perimeter
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        # Require high extent (>0.6) to avoid complex organic shapes
                        if len(approx) >= 3 and len(approx) <= 6 and extent > 0.60:
                            is_geometric = True
                            logger.debug(f"Detected regular polygon (vertices={len(approx)}, extent={extent:.3f}), skipping skeletonization")
            
            # Only skeletonize if NOT a simple geometric shape
            if not is_geometric:
                # Apply Zhang-Suen thinning to extract skeleton
                try:
                    skeleton = cv2.ximgproc.thinning(
                        binary, 
                        thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                    )
                    image = skeleton
                    logger.debug(f"Applied skeletonization (bright_ratio={bright_ratio:.3f})")
                except AttributeError:
                    # ximgproc not available, fall back to morphological thinning
                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(binary, kernel, iterations=2)
                    image = eroded
                    logger.debug(f"Applied erosion fallback (bright_ratio={bright_ratio:.3f})")

    # Convert to [0, 1] range
    img_array = image.astype(np.float32) / 255.0
    
    # Apply z-score normalization (same as training pipeline)
    # This removes brightness bias and ensures consistent distribution
    img_flat = img_array.flatten()
    if img_flat.std() > 0.01:  # Only normalize if sufficient variation
        # Standardize to zero mean, unit variance
        img_array = (img_array - img_flat.mean()) / (img_flat.std() + 1e-7)
        # Rescale from ~[-2, 2] to [0, 1] for model compatibility
        img_array = (img_array + 2) / 4
        img_array = np.clip(img_array, 0, 1)
    
    img_array = img_array.reshape(1, target_h, target_w, 1)
    return img_array
