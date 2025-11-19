"""
Stroke- and contour-based shape extraction utilities for DoodleHunter.

This module was extracted from src/core/shape_detection.py to keep responsibilities
focused and testable.

Public API:
- extract_shapes_from_strokes(...)
- extract_shapes(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple, Optional

import cv2
import numpy as np

from .shape_types import MIN_SHAPE_AREA_PX

logger = logging.getLogger(__name__)


def extract_shapes_from_strokes(
    stroke_history: List[dict],
    min_shape_area: int = MIN_SHAPE_AREA_PX,
    spatial_eps: float = 32.0,
    time_eps_ms: int = 900,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract multiple shapes from stroke history using spatial + temporal structure.

    Heuristic clustering:
    - Group strokes whose endpoints are spatially close (<= spatial_eps)
      OR whose time gap is small (<= time_eps_ms).
    - Each cluster becomes a candidate shape with its own bounding box.

    Returns:
        List of (points_or_contour, (x, y, w, h)).
    """
    if not stroke_history:
        return []

    if not isinstance(stroke_history, Iterable):
        raise TypeError("stroke_history must be iterable of stroke dicts")

    norm_strokes: List[Dict[str, Any]] = []

    for s in stroke_history:
        if not isinstance(s, dict):
            logger.debug("Skipping non-dict stroke entry: %r", s)
            continue

        pts = s.get("points") or []
        if not isinstance(pts, Iterable):
            logger.debug("Skipping stroke with non-iterable points: %r", s)
            continue

        xs: List[int] = []
        ys: List[int] = []
        ts: List[int] = []

        for p in pts:
            if not isinstance(p, dict):
                continue
            try:
                x = int(p.get("x"))
                y = int(p.get("y"))
            except (TypeError, ValueError):
                # Skip malformed point
                continue
            xs.append(x)
            ys.append(y)
            if "t" in p:
                try:
                    ts.append(int(p["t"]))
                except (TypeError, ValueError):
                    # Ignore malformed timestamp
                    pass

        if not xs or not ys:
            continue

        start = (xs[0], ys[0])
        end = (xs[-1], ys[-1])

        # Prefer explicit stroke timestamp; fallback to last point time; else 0.
        stroke_ts_raw = s.get("timestamp", ts[-1] if ts else 0)
        try:
            stroke_ts = int(stroke_ts_raw)
        except (TypeError, ValueError):
            stroke_ts = 0

        norm_strokes.append(
            {
                "points": np.column_stack([xs, ys]).astype(np.int32),
                "start": start,
                "end": end,
                "t": stroke_ts,
            }
        )

    if not norm_strokes:
        return []

    # Union-find style clustering based on spatial + temporal proximity
    n = len(norm_strokes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx * dx + dy * dy) ** 0.5)

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = norm_strokes[i], norm_strokes[j]
            dt = abs(si["t"] - sj["t"])
            spatial_close = (
                dist(si["end"], sj["start"]) <= spatial_eps
                or dist(si["start"], sj["end"]) <= spatial_eps
                or dist(si["start"], sj["start"]) <= spatial_eps
                or dist(si["end"], sj["end"]) <= spatial_eps
            )
            if spatial_close or (dt and dt <= time_eps_ms):
                union(i, j)

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for idx, s in enumerate(norm_strokes):
        root = find(idx)
        clusters.setdefault(root, []).append(s)

    shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for cluster_strokes in clusters.values():
        pts = np.concatenate([cs["points"] for cs in cluster_strokes], axis=0)
        if pts.size == 0:
            continue

        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))

        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        area = w * h

        if area < min_shape_area:
            logger.debug(
                "extract_shapes_from_strokes: rejected tiny stroke-cluster "
                "bbox=(%d,%d,%d,%d) area=%d < min_shape_area=%d",
                x_min,
                y_min,
                w,
                h,
                area,
                min_shape_area,
            )
            continue

        shapes.append((pts, (x_min, y_min, w, h)))

    return shapes


def extract_shapes(
    image: np.ndarray,
    min_shape_area: int = MIN_SHAPE_AREA_PX,
    use_adaptive_threshold: bool = True,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract individual shapes from image using robust contour detection.

    Returns:
        List of (contour, (x, y, w, h)).
    """
    # Validate and ensure grayscale
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("image must be 2D grayscale or 3D color array")

    # Binarize
    if use_adaptive_threshold:
        img_blur = cv2.GaussianBlur(image, (3, 3), 0)
        _, binary = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # For contour detection with RETR_EXTERNAL, we need dark strokes on white background.
    # BUT: Input is already correctly inverted in app.py (white-on-black for model).
    # After binarization, if mean < 127, we have mostly black pixels (correct).
    # findContours with RETR_EXTERNAL finds white regions, so we need to invert
    # ONLY if strokes are already white (bright mean).
    if np.mean(binary) > 127:
        # Bright image means white strokes on black - invert for contour detection
        binary = cv2.bitwise_not(binary)

    # Light morphological close to connect broken strokes
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    h_img, w_img = image.shape[:2]
    canvas_area = float(w_img * h_img)

    for contour in contours:
        if contour is None or contour.size == 0:
            continue

        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        if area < min_shape_area:
            logger.debug(
                "extract_shapes: rejected tiny contour bbox=%s area=%.1f < min_shape_area=%d",
                cv2.boundingRect(contour),
                area,
                min_shape_area,
            )
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        rel_area = area / max(canvas_area, 1.0)

        # Reject extremely thin, tiny shapes likely due to edges/artifacts
        if aspect_ratio > 20.0 and rel_area < 0.001:
            continue

        shapes.append((contour, (x, y, w, h)))

    return shapes


def propose_shapes(
    gray: np.ndarray,
    stroke_history: Optional[List[dict]],
    min_shape_area: int = MIN_SHAPE_AREA_PX,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Generate candidate shapes from stroke_history (preferred) or from contours.
    """
    stroke_shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    if stroke_history:
        try:
            stroke_shapes = extract_shapes_from_strokes(
                stroke_history,
                min_shape_area=min_shape_area,
            )
        except Exception as exc:
            # Stroke clustering is best-effort; fall back if it fails.
            logger.error(
                "extract_shapes_from_strokes failed; falling back to contours: %s",
                exc,
            )
            stroke_shapes = []

    if stroke_shapes:
        return stroke_shapes

    return extract_shapes(gray, min_shape_area=min_shape_area)