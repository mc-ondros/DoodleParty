"""
Shared dataclasses and constants for shape-based detection.

This module centralizes core types so they can be reused across:
- shape_extraction
- shape_normalization
- shape_inference
- shape_grouping
- shape_heuristics
- shape_detection (facade / orchestrator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Centralized tuning constants and defaults.
DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.5
MIN_SHAPE_AREA_PX: int = 100
PADDING_COLOR_BG: int = 0  # Black background (matches QuickDraw training data)

# Grouping thresholds for generic nearby grouping
NEARBY_MERGE_DISTANCE_PX: float = 100.0
NEARBY_MERGE_IOU: float = 0.05

# Grouping thresholds for positive/near-positive merging
POSITIVE_MERGE_DISTANCE_PX: float = 80.0
POSITIVE_MERGE_IOU: float = 0.05

# Near-positive penis heuristic
NEAR_POSITIVE_MIN_SCORE: float = 0.45
NEAR_POSITIVE_MIN_COUNT: int = 3


@dataclass
class ShapeInfo:
    """
    Information about a detected shape.

    Attributes:
        contour: Original contour points or clustered stroke points.
        bounding_box: (x, y, width, height) on original canvas.
        normalized_image: Shape normalized to model input size.
        confidence: ML confidence score.
        is_positive: Whether shape is flagged as offensive.
        area: Contour/box area in pixels.
        shape_id: Unique identifier.
    """
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    normalized_image: np.ndarray
    confidence: float
    is_positive: bool
    area: int
    shape_id: int


@dataclass
class ShapeDetectionResult:
    """
    Result of shape-based detection.

    Attributes:
        is_positive: Overall verdict (after grouping / heuristics).
        confidence: Highest confidence among groups or shapes.
        shape_predictions: Individual shape results (raw).
        num_shapes_analyzed: Number of shapes evaluated.
        canvas_dimensions: (width, height) of original canvas.
        grouped_boxes: Merged group bounding boxes (x, y, w, h).
        grouped_scores: Aggregated confidence per group.
    """
    is_positive: bool
    confidence: float
    shape_predictions: List[ShapeInfo]
    num_shapes_analyzed: int
    canvas_dimensions: Tuple[int, int]
    grouped_boxes: List[Tuple[int, int, int, int]]
    grouped_scores: List[float]