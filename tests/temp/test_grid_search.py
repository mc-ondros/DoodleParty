#!/usr/bin/env python3
"""
Grid-search preprocessing to maximize model confidence for a given image.
"""
import itertools
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, '.')
from src.core.shape_normalization import preprocess_for_model
from src.web.app import tflite_interpreter


def ensure_inverted(gray: np.ndarray) -> np.ndarray:
    # If background is light, invert to white strokes on black background
    return (255 - gray) if gray.mean() > 127 else gray


def auto_bbox(img: np.ndarray, thresh: int = 200):
    ys, xs = np.where(img >= thresh)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def crop_to_bbox(img: np.ndarray, bbox, margin_ratio=0.08):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    m = int(max(bw, bh) * margin_ratio)
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w - 1, x2 + m)
    y2 = min(h - 1, y2 + m)
    return img[y1:y2 + 1, x1:x2 + 1]


def infer(arr: np.ndarray) -> float:
    tflite_interpreter.set_tensor(
        tflite_interpreter.get_input_details()[0]['index'], arr
    )
    tflite_interpreter.invoke()
    out = tflite_interpreter.get_tensor(
        tflite_interpreter.get_output_details()[0]['index']
    )
    return float(out[0, 0])


def try_variant(gray: np.ndarray, blur_k, thr_mode, morph_kind, morph_k, rot, flip):
    img = gray.copy()
    if blur_k:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    # Binarize
    if thr_mode == 'otsu':
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thr_mode == 'fixed200':
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    elif thr_mode == 'adaptive':
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 2)

    # Morphology
    if morph_kind != 'none' and morph_k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        if morph_kind == 'dilate':
            img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=1)
        elif morph_kind == 'close':
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Rotate
    if rot:
        rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        img = cv2.rotate(img, rot_map[rot])

    # Flip
    if flip == 'h':
        img = cv2.flip(img, 1)
    elif flip == 'v':
        img = cv2.flip(img, 0)

    # BBox on white strokes
    bbox = auto_bbox(img, 200)
    if not bbox:
        return None, None, None
    roi = crop_to_bbox(img, bbox, margin_ratio=0.08)

    arr = preprocess_for_model(roi, target_size=128)
    conf = infer(arr)
    return conf, img, roi


def main(image_path: Path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"ERROR: cannot read {image_path}")
        sys.exit(1)

    # Standardize base size and polarity
    gray = ensure_inverted(gray)
    if gray.shape != (512, 512):
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)

    configs = list(itertools.product(
        [None, 3],                 # blur_k
        ['otsu', 'fixed200'],      # thr_mode
        ['none', 'dilate', 'close'],  # morph_kind
        [0, 3],                    # morph_k (0 -> ignored)
        [0, 90, 180, 270],         # rot
        ['none', 'h', 'v'],        # flip
    ))

    best = (-1.0, None, None, None)
    for (blur_k, thr_mode, morph_kind, morph_k, rot, flip) in configs:
        if morph_k == 0:
            mk = None
        else:
            mk = morph_k
        conf, bin_img, roi = try_variant(gray, blur_k, thr_mode, morph_kind, mk, rot, flip)
        if conf is None:
            continue
        if conf > best[0]:
            best = (conf, bin_img, roi, (blur_k, thr_mode, morph_kind, mk, rot, flip))

    conf, bin_img, roi, cfg = best
    print(f"Best confidence: {conf:.6f}")
    print(f"Best config: blur={cfg[0]} thr={cfg[1]} morph={cfg[2]}({cfg[3]}) rot={cfg[4]} flip={cfg[5]}")

    if bin_img is not None:
        cv2.imwrite('/tmp/gs_bin.png', bin_img)
    if roi is not None:
        cv2.imwrite('/tmp/gs_roi.png', roi)

    if conf >= 0.90:
        print("✅ Achieved >90% with grid-search")
    else:
        print("❌ Still <90%. We'll integrate the best config and retry other tweaks if needed.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_grid_search.py <image>")
        sys.exit(1)
    main(Path(sys.argv[1]))
