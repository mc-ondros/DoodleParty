"""
Grouping and heuristic utilities for shape-based detection.

Extracted from src/core/shape_detection.py to isolate:
- generic bbox grouping
- positive-group merging
- penis-specific near-positive heuristic (near-threshold clustering)

Public API:
- merge_nearby_shapes(...)
- merge_positive_shapes(...)
- apply_near_positive_heuristic(...)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

from .shape_types import (
    ShapeInfo,
    NEARBY_MERGE_DISTANCE_PX,
    NEARBY_MERGE_IOU,
    POSITIVE_MERGE_DISTANCE_PX,
    POSITIVE_MERGE_IOU,
    NEAR_POSITIVE_MIN_COUNT,
    NEAR_POSITIVE_MIN_SCORE,
)

logger = logging.getLogger(__name__)


def _merge_groups(
    shapes: List[ShapeInfo],
    distance_thresh: float,
    iou_thresh: float,
    filter_fn: Callable[[ShapeInfo], bool],
) -> List[List[ShapeInfo]]:
    """
    Generic union-find grouping for shapes based on bbox distance/IoU.

    Args:
        shapes: Candidate shapes.
        distance_thresh: Max center distance to consider "nearby".
        iou_thresh: Min IoU to consider overlapping.
        filter_fn: Predicate to select shapes that participate in grouping.
    """
    if not shapes:
        return []

    selected = [s for s in shapes if filter_fn(s)]
    n = len(selected)
    if n == 0:
        return []

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

    def bbox_distance(b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
        cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
        dx = cx1 - cx2
        dy = cy1 - cy2
        return float((dx * dx + dy * dy) ** 0.5)

    def bbox_iou(b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter_w = max(0.0, xb - xa)
        inter_h = max(0.0, yb - ya)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area1 = float(w1 * h1)
        area2 = float(w2 * h2)
        union_area = area1 + area2 - inter
        if union_area <= 0.0:
            return 0.0
        return float(inter / union_area)

    for i in range(n):
        bi = selected[i].bounding_box
        for j in range(i + 1, n):
            bj = selected[j].bounding_box

            d = bbox_distance(bi, bj)
            if d <= distance_thresh:
                union(i, j)
                continue

            iou = bbox_iou(bi, bj)
            if iou >= iou_thresh:
                union(i, j)

    groups: Dict[int, List[ShapeInfo]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(selected[idx])

    return list(groups.values())


def merge_nearby_shapes(
    shapes: List[ShapeInfo],
    distance_thresh: float = NEARBY_MERGE_DISTANCE_PX,
    iou_thresh: float = NEARBY_MERGE_IOU,
) -> List[List[ShapeInfo]]:
    """
    Merge ALL nearby shapes into groups (regardless of confidence).

    Used to:
    - Build merged group candidates for secondary inference.
    - Provide spatial context for complex multi-part shapes.
    """
    return _merge_groups(
        shapes,
        distance_thresh=distance_thresh,
        iou_thresh=iou_thresh,
        filter_fn=lambda s: True,
    )


def merge_positive_shapes(
    shapes: List[ShapeInfo],
    distance_thresh: float = POSITIVE_MERGE_DISTANCE_PX,
    iou_thresh: float = POSITIVE_MERGE_IOU,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], bool, float]:
    """
    Merge nearby positive shapes into groups (e.g., shaft + 2 balls → one penis).

    Returns:
        grouped_boxes, grouped_scores, any_group_positive, max_group_conf
    """
    groups = _merge_groups(
        shapes,
        distance_thresh=distance_thresh,
        iou_thresh=iou_thresh,
        filter_fn=lambda s: s.is_positive,
    )

    if not groups:
        return [], [], False, 0.0

    grouped_boxes: List[Tuple[int, int, int, int]] = []
    grouped_scores: List[float] = []
    max_group_conf = 0.0

    for members in groups:
        xs: List[int] = []
        ys: List[int] = []
        x2s: List[int] = []
        y2s: List[int] = []
        group_conf = 0.0

        for s in members:
            x, y, w, h = s.bounding_box
            xs.append(x)
            ys.append(y)
            x2s.append(x + w)
            y2s.append(y + h)
            if s.confidence > group_conf:
                group_conf = s.confidence

        if not xs:
            continue

        gx = int(min(xs))
        gy = int(min(ys))
        gw = int(max(x2s) - gx)
        gh = int(max(y2s) - gy)

        grouped_boxes.append((gx, gy, gw, gh))
        grouped_scores.append(group_conf)
        if group_conf > max_group_conf:
            max_group_conf = group_conf

    any_group_positive = len(grouped_boxes) > 0
    return grouped_boxes, grouped_scores, any_group_positive, max_group_conf


def apply_near_positive_heuristic(
    shape_predictions: List[ShapeInfo],
    classification_threshold: float,
) -> Tuple[bool, float, List[Tuple[int, int, int, int]], List[float]]:
    """
    Penis-aware near-positive heuristic.

    Goal:
        Robustly flag classic 3-part penis-like clusters even when the raw
        model scores are slightly sub-threshold, without impacting generic shapes.

    Returns:
        (any_positive, new_confidence, boxes, scores)
    """
    if not shape_predictions:
        return False, 0.0, [], []

    max_shape_conf = max(s.confidence for s in shape_predictions)
    overall_confidence = max_shape_conf

    # Near-positive band: slightly below the main classification threshold,
    # but not so low that random noise qualifies.
    near_pos_thresh = max(
        NEAR_POSITIVE_MIN_SCORE,
        classification_threshold * 0.9,
    )

    # Select indices of shapes that live in the near-positive band
    near_indices = [
        idx
        for idx, s in enumerate(shape_predictions)
        if s.confidence >= near_pos_thresh
    ]

    if len(near_indices) < NEAR_POSITIVE_MIN_COUNT:
        # Not enough evidence to apply penis-specific clustering.
        return False, overall_confidence, [], []

    # Build a temporary view where only near-positive shapes are marked positive
    tmp_shapes: List[ShapeInfo] = []
    for idx, s in enumerate(shape_predictions):
        is_np = idx in near_indices
        tmp_shapes.append(
            ShapeInfo(
                contour=s.contour,
                bounding_box=s.bounding_box,
                normalized_image=s.normalized_image,
                confidence=s.confidence,
                is_positive=is_np,
                area=s.area,
                shape_id=s.shape_id,
            )
        )

    np_boxes, np_scores, np_any, np_max = merge_positive_shapes(tmp_shapes)

    # If we didn't form any merged positive cluster, abort.
    if not np_any or not np_boxes:
        return False, overall_confidence, [], []

    # We expect classic multi-part penis patterns (shaft + 2 balls) to merge
    # into a single tight group. If we see exactly one group built from at
    # least NEAR_POSITIVE_MIN_COUNT near-positive members, treat this as a
    # strong signal and promote to a high-confidence positive.
    if len(np_boxes) == 1:
        gx, gy, gw, gh = np_boxes[0]
        gx2, gy2 = gx + gw, gy + gh

        participating = 0
        for s in tmp_shapes:
            if not s.is_positive:
                continue
            x, y, w, h = s.bounding_box
            x2, y2 = x + w, y + h
            # Require the near-positive bbox to be mostly inside the merged box.
            if x >= gx and y >= gy and x2 <= gx2 and y2 <= gy2:
                participating += 1

        if participating >= NEAR_POSITIVE_MIN_COUNT:
            base = max(max_shape_conf, np_max)
            combined_conf = max(0.95, min(0.99, base + 0.4))
            return True, combined_conf, np_boxes, [combined_conf]

    # Fallback: preserve existing overall confidence if heuristic conditions not met.
    return False, overall_confidence, [], []