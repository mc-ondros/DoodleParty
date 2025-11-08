"""
Shape-based detection orchestrator for DoodleHunter.

This module provides the ShapeDetector facade that coordinates:
- Shape extraction (stroke clustering & contour detection)
- Shape normalization (padding, resizing, preprocessing)
- ML inference (TFLite & Keras model evaluation)
- Shape grouping (proximity-based merging)
- Penis-specific heuristics (near-positive clustering)

Why shape detection:
- Analyzes complete objects rather than arbitrary tile fragments
- Normalizes shapes to model input size (e.g. 128x128)
- Handles non-square canvases properly
- Provides precise localization of offensive content

Architecture:
This is the orchestrator/facade. Core functionality is delegated to:
- src/core/shape_types.py (dataclasses & constants)
- src/core/shape_extraction.py (stroke & contour extraction)
- src/core/shape_normalization.py (padding, resizing, preprocessing)
- src/core/shape_inference.py (model inference)
- src/core/shape_grouping.py (proximity grouping & heuristics)

Related:
- src/core/contour_detection.py (alternative contour-based detector)
- src/core/tile_detection.py (tile-based detector)
- src/web/app.py (Flask API endpoints)

Exports:
- ShapeDetector: Main orchestrator class
- ShapeInfo: Re-exported from shape_types
- ShapeDetectionResult: Re-exported from shape_types
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Any, Dict, Iterable

from .shape_types import (
    ShapeInfo,
    ShapeDetectionResult,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MIN_SHAPE_AREA_PX,
    PADDING_COLOR_BG,
    NEARBY_MERGE_DISTANCE_PX,
    NEARBY_MERGE_IOU,
    POSITIVE_MERGE_DISTANCE_PX,
    POSITIVE_MERGE_IOU,
    NEAR_POSITIVE_MIN_SCORE,
    NEAR_POSITIVE_MIN_COUNT,
)
from .shape_extraction import (
    propose_shapes,
    extract_shapes_from_strokes,
    extract_shapes,
)
from .shape_normalization import normalize_shape, preprocess_for_model
from .shape_inference import detect_model_input_size, predict_shape_with_model
from .shape_grouping import (
    merge_nearby_shapes,
    merge_positive_shapes,
    apply_near_positive_heuristic,
)

logger = logging.getLogger(__name__)


def _ensure_debug_dir() -> str:
    """
    Ensure the global debug directory exists.

    Returns:
        Path to the debug directory as string.
    """
    debug_dir = "/tmp/doodlehunter_debug"
    try:
        os.makedirs(debug_dir, exist_ok=True)
    except Exception as exc:
        # Hard requirement is "must output images"; if creation fails,
        # later save attempts will raise and be visible during testing.
        logger.debug("Failed to create debug dir %s: %s", debug_dir, exc)
    return debug_dir


def _save_debug_image(image: np.ndarray, filename: str) -> None:
    """
    Save a debug image to the global debug directory.

    Requirements:
    - Only persist images that are actually sent into the ML model.
      That means:
        - Preprocessed model input tensors (after all transforms).
    """
    debug_dir = _ensure_debug_dir()
    path = os.path.join(debug_dir, filename)
    try:
        # Ensure uint8 grayscale or BGR for consistency
        if image.dtype != np.uint8:
            img = image.astype(np.uint8)
        else:
            img = image

        if img.ndim == 2:
            cv2.imwrite(path, img)
        elif img.ndim == 3:
            # If single-channel in last dim, squeeze; otherwise assume RGB
            if img.shape[2] == 1:
                cv2.imwrite(path, img[:, :, 0])
            else:
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            # Unsupported shape: best-effort flatten first two dims
            h, w = img.shape[0], img.shape[1]
            img2 = img.reshape(h, w)
            cv2.imwrite(path, img2)
    except Exception as exc:
        # Debug saving must not break primary detection.
        logger.debug("Failed to save debug image %s: %s", path, exc)


class ShapeDetector:
    """
    Shape-based detection system.

    Pipeline:
    - Validate and normalize input image.
    - Propose shapes via stroke clustering (preferred) or contour extraction.
    - Normalize each shape and run ML inference.
    - Merge nearby shapes to score combined multi-part patterns.
    - Apply penis-aware heuristic for clustered near-positive shapes.

    Priorities:
    - Robust multi-part penis detection (reduced false negatives).
    - Stronger input validation and explicit error surfaces.
    - Refactored, reusable grouping utilities (no duplication).
    - Cached model input sizing to avoid repeated introspection.
    - Debug image hooks tied only to actual model inputs.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tflite_interpreter: Optional[Any] = None,
        is_tflite: bool = False,
        classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
        min_shape_area: int = MIN_SHAPE_AREA_PX,
        padding_color: int = PADDING_COLOR_BG,
    ):
        self.model = model
        self.tflite_interpreter = tflite_interpreter
        self.is_tflite = is_tflite

        self.classification_threshold = float(classification_threshold)
        self.min_shape_area = int(min_shape_area)
        self.padding_color = int(padding_color)

        # Cache model's expected input size once and reuse.
        self.target_size = self._detect_model_input_size()

    def _detect_model_input_size(self) -> int:
        """
        Proxy to shared detect_model_input_size() helper for backwards compatibility.
        """
        return detect_model_input_size(
            model=self.model,
            tflite_interpreter=self.tflite_interpreter,
            is_tflite=self.is_tflite,
            default=128,
        )

    #
    # Normalization and preprocessing (delegated to extracted modules)
    #

    def normalize_shape(
        self,
        image: np.ndarray,
        bounding_box: Tuple[int, int, int, int],
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract shape region and normalize with padding.
        Delegates to shape_normalization module.
        """
        if target_size is None:
            target_size = self.target_size or 128
        return normalize_shape(
            image=image,
            bounding_box=bounding_box,
            target_size=target_size,
            padding_color=self.padding_color,
        )

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess normalized shape for model input using cached target_size.
        Delegates to shape_normalization module.
        """
        return preprocess_for_model(
            image=image,
            target_size=self.target_size or 128,
            skeletonize=True,  # Enable skeletonization to match QuickDraw training data
        )

    #
    # Model inference (delegated to extracted module)
    #

    def predict_shape(self, normalized_shape: np.ndarray) -> float:
        """
        Run ML inference on normalized shape and return an offensive score in [0,1].
        Delegates to shape_inference module.
        """
        input_array = self.preprocess_for_model(normalized_shape)
        return predict_shape_with_model(
            normalized_input=input_array,
            model=self.model,
            tflite_interpreter=self.tflite_interpreter,
            is_tflite=self.is_tflite,
        )

    #
    # Generic grouping utilities (delegated to extracted module)
    #

    def _merge_nearby_shapes(
        self,
        shapes: List[ShapeInfo],
        distance_thresh: float = NEARBY_MERGE_DISTANCE_PX,
        iou_thresh: float = NEARBY_MERGE_IOU,
    ) -> List[List[ShapeInfo]]:
        """
        Merge ALL nearby shapes into groups (regardless of confidence).
        Delegates to shape_grouping module.
        """
        return merge_nearby_shapes(
            shapes=shapes,
            distance_thresh=distance_thresh,
            iou_thresh=iou_thresh,
        )

    def _merge_positive_shapes(
        self,
        shapes: List[ShapeInfo],
        distance_thresh: float = POSITIVE_MERGE_DISTANCE_PX,
        iou_thresh: float = POSITIVE_MERGE_IOU,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], bool, float]:
        """
        Merge nearby positive shapes into groups (e.g., shaft + 2 balls → one penis).
        Delegates to shape_grouping module.
        """
        return merge_positive_shapes(
            shapes=shapes,
            distance_thresh=distance_thresh,
            iou_thresh=iou_thresh,
        )

    #
    # Internal pipeline helpers
    #

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input image to grayscale ndarray.

        Raises:
            TypeError / ValueError for malformed inputs.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 2:
            return image.copy()
        raise ValueError("image must be 2D grayscale or 3D color array")

    def _propose_shapes(
        self,
        gray: np.ndarray,
        stroke_history: Optional[List[dict]],
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Generate candidate shapes from stroke_history (preferred) or from contours.
        Delegates to shape_extraction module.
        """
        return propose_shapes(
            gray=gray,
            stroke_history=stroke_history,
            min_shape_area=self.min_shape_area,
        )

    def _classify_shapes(
        self,
        gray: np.ndarray,
        shapes_raw: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    ) -> List[ShapeInfo]:
        """
        Normalize and classify candidate shapes independently.
        """
        shape_predictions: List[ShapeInfo] = []

        if not shapes_raw:
            logger.debug("ShapeDetector._classify_shapes: no shapes_raw provided")
            return shape_predictions

        for idx, (contour_or_points, bbox) in enumerate(shapes_raw):
            x, y, w, h = bbox
            logger.debug(
                "ShapeDetector._classify_shapes: shape %d bbox=(%d,%d,%d,%d)",
                idx,
                x,
                y,
                w,
                h,
            )

            normalized = self.normalize_shape(gray, bbox)
            confidence = self.predict_shape(normalized)
            logger.debug(
                "ShapeDetector._classify_shapes: shape %d model_confidence=%.6f",
                idx,
                confidence,
            )

            # Save ONLY the actual model input (post-preprocessing) for debugging
            preprocessed = self.preprocess_for_model(normalized)
            vis = preprocessed[0]
            if vis.ndim == 3 and vis.shape[-1] == 1:
                vis = vis[:, :, 0]
            vis_uint8 = (np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8)
            _save_debug_image(
                vis_uint8,
                f"shape_{idx:03d}_model_input.png",
            )

            is_positive = confidence >= self.classification_threshold

            # Derive area: for clustered strokes, contourArea may fail → fallback to bbox
            area = int(w * h)
            if (
                isinstance(contour_or_points, np.ndarray)
                and contour_or_points.ndim >= 2
            ):
                try:
                    ca = int(cv2.contourArea(contour_or_points))
                    if ca > 0:
                        area = ca
                except Exception as exc:
                    logger.debug(
                        "contourArea failed for shape %d: %s", idx, exc
                    )

            shape_predictions.append(
                ShapeInfo(
                    contour=contour_or_points,
                    bounding_box=bbox,
                    normalized_image=normalized,
                    confidence=confidence,
                    is_positive=is_positive,
                    area=area,
                    shape_id=idx,
                )
            )

        return shape_predictions

    def _evaluate_merged_groups(
        self,
        gray: np.ndarray,
        merged_groups: List[List[ShapeInfo]],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], float]:
        """
        Run secondary inference on merged groups and return:
            (grouped_boxes, grouped_scores, max_group_conf)
        """
        grouped_boxes: List[Tuple[int, int, int, int]] = []
        grouped_scores: List[float] = []
        max_group_conf = 0.0

        for group_idx, group_members in enumerate(merged_groups):
            if not group_members:
                continue

            xs: List[int] = []
            ys: List[int] = []
            x2s: List[int] = []
            y2s: List[int] = []

            for s in group_members:
                x, y, w, h = s.bounding_box
                xs.append(x)
                ys.append(y)
                x2s.append(x + w)
                y2s.append(y + h)

            if not xs:
                continue

            gx = int(min(xs))
            gy = int(min(ys))
            gw = int(max(x2s) - gx)
            gh = int(max(y2s) - gy)
            gbox = (gx, gy, gw, gh)

            group_normalized = self.normalize_shape(gray, gbox)
            group_confidence = self.predict_shape(group_normalized)

            group_preprocessed = self.preprocess_for_model(group_normalized)
            gvis = group_preprocessed[0]
            if gvis.ndim == 3 and gvis.shape[-1] == 1:
                gvis = gvis[:, :, 0]
            gvis_uint8 = (np.clip(gvis, 0.0, 1.0) * 255).astype(np.uint8)
            _save_debug_image(
                gvis_uint8,
                f"group_{group_idx:03d}_model_input.png",
            )

            grouped_boxes.append(gbox)
            grouped_scores.append(group_confidence)
            if group_confidence > max_group_conf:
                max_group_conf = group_confidence

        return grouped_boxes, grouped_scores, max_group_conf

    def _apply_near_positive_heuristic(
        self,
        shape_predictions: List[ShapeInfo],
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]], List[float]]:
        """
        Penis-aware heuristic.
        Delegates to shape_grouping module.
        """
        return apply_near_positive_heuristic(
            shape_predictions=shape_predictions,
            classification_threshold=self.classification_threshold,
        )

    #
    # Public API
    #
 
    def detect(
        self,
        image: np.ndarray,
        stroke_history: Optional[List[dict]] = None,
    ) -> ShapeDetectionResult:
        """
        Detect and classify all shapes in image using all available signals.
 
        Steps:
        - Validate/normalize image to grayscale.
        - Propose shapes via strokes or contours.
        - Classify shapes.
        - Merge nearby shapes and run group-level inference.
        - Apply penis-aware near-positive heuristic.
 
        Design note:
        - If no shapes are detected, we run a single global prediction on the entire
          canvas so the output reflects the drawing instead of a meaningless constant.
        """
        gray = self._validate_image(image)
        canvas_h, canvas_w = gray.shape[:2]
 
        # 1) Propose candidate shapes
        shapes_raw = self._propose_shapes(gray, stroke_history)
        logger.debug(
            "ShapeDetector.detect: proposed %d candidate shapes", len(shapes_raw)
        )
 
        # 2) Classify individual shapes
        shape_predictions = self._classify_shapes(gray, shapes_raw)
        logger.debug(
            "ShapeDetector.detect: classified %d shapes", len(shape_predictions)
        )
 
        # Fallback: if no shapes survived, run a single global prediction on the
        # entire canvas so the output reflects the actual drawing instead of a
        # hard-coded or misleading tiny constant.
        if not shape_predictions:
            logger.warning(
                "ShapeDetector.detect: no shapes detected; running global canvas "
                "fallback prediction"
            )
            try:
                global_norm = self.normalize_shape(
                    gray, (0, 0, canvas_w, canvas_h)
                )
                global_conf = self.predict_shape(global_norm)
                logger.info(
                    "ShapeDetector.detect: global canvas confidence=%.6f", global_conf
                )
 
                global_shape = ShapeInfo(
                    contour=np.array([], dtype=np.int32),
                    bounding_box=(0, 0, canvas_w, canvas_h),
                    normalized_image=global_norm,
                    confidence=global_conf,
                    is_positive=global_conf >= self.classification_threshold,
                    area=int(canvas_w * canvas_h),
                    shape_id=0,
                )
                shape_predictions = [global_shape]
            except Exception as exc:
                logger.error(
                    "ShapeDetector.detect: global fallback prediction failed: %s", exc
                )
                return ShapeDetectionResult(
                    is_positive=False,
                    confidence=0.0,
                    shape_predictions=[],
                    num_shapes_analyzed=0,
                    canvas_dimensions=(canvas_w, canvas_h),
                    grouped_boxes=[],
                    grouped_scores=[],
                )
 
        # 3) Merge nearby shapes (regardless of confidence) and evaluate groups
        merged_groups = self._merge_nearby_shapes(shape_predictions)
        grouped_boxes, grouped_scores, max_group_conf = self._evaluate_merged_groups(
            gray, merged_groups
        )
        logger.debug(
            "ShapeDetector.detect: %d merged groups, grouped_scores=%s, max_group_conf=%.6f",
            len(grouped_boxes),
            grouped_scores,
            max_group_conf,
        )
 
        # 4) Primary verdict from group scores
        any_group_positive = (
            any(score >= self.classification_threshold for score in grouped_scores)
            if grouped_scores
            else False
        )
 
        overall_positive = any_group_positive
        if grouped_scores:
            overall_confidence = max_group_conf
        else:
            overall_confidence = max(s.confidence for s in shape_predictions)
 
        # 5) Penis-aware clustered near-positive heuristic
        if not overall_positive and shape_predictions:
            np_any, np_conf, np_boxes, np_scores = self._apply_near_positive_heuristic(
                shape_predictions
            )
            if np_any:
                overall_positive = True
                overall_confidence = np_conf
                grouped_boxes = np_boxes
                grouped_scores = np_scores
 
        logger.info(
            "ShapeDetector.detect: verdict=%s confidence=%.6f "
            "(num_shapes=%d, num_groups=%d)",
            "POSITIVE" if overall_positive else "NEGATIVE",
            overall_confidence,
            len(shape_predictions),
            len(grouped_boxes),
        )
 
        return ShapeDetectionResult(
            is_positive=overall_positive,
            confidence=overall_confidence,
            shape_predictions=shape_predictions,
            num_shapes_analyzed=len(shape_predictions),
            canvas_dimensions=(canvas_w, canvas_h),
            grouped_boxes=grouped_boxes,
            grouped_scores=grouped_scores,
        )
