"""
Inference helpers for shape-based detection.

Encapsulates:
- Model input-size detection
- Single-shape prediction with debug logging

This is imported by the ShapeDetector orchestrator to keep app.py and
shape_detection.py slimmer and easier to test.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def detect_model_input_size(
    model: Optional[Any],
    tflite_interpreter: Optional[Any],
    is_tflite: bool,
    default: int = 128,
) -> int:
    """
    Detect the model's expected input size.

    For QuickDraw production models:
    - (None, 128, 128, 1) or [1, 128, 128, 1]

    Fallback:
    - If shape cannot be read, default to `default`.
    """
    if is_tflite and tflite_interpreter:
        try:
            input_details = tflite_interpreter.get_input_details()
            return int(input_details[0]["shape"][1])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to read TFLite input shape; falling back to %d: %s",
                default,
                exc,
            )
            return default

    if model is not None:
        try:
            input_shape = getattr(model, "input_shape", None)
            if (
                isinstance(input_shape, (list, tuple))
                and len(input_shape) == 4
                and input_shape[1] is not None
                and input_shape[2] is not None
            ):
                return int(input_shape[1])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to read Keras input_shape; falling back to %d: %s",
                default,
                exc,
            )
            return default
        # Default for unknown Keras shapes
        return default

    # No model bound: assume QuickDraw-compatible 128x128
    return default


def predict_shape_with_model(
    normalized_input: np.ndarray,
    model: Optional[Any],
    tflite_interpreter: Optional[Any],
    is_tflite: bool,
) -> float:
    """
    Run ML inference on normalized model-ready input and return an offensive score in [0,1].

    normalized_input:
        - Expected shape (1, H, W, 1), float32 in [0,1].
    """
    logger.info("=== CONFIDENCE DEBUG: predict_shape_with_model called ===")
    logger.info("Input shape: %s", getattr(normalized_input, "shape", None))
    logger.info("Input dtype: %s", getattr(normalized_input, "dtype", None))

    # Basic sanity on input
    if not isinstance(normalized_input, np.ndarray):
        logger.error("predict_shape_with_model: input is not a numpy array")
        return 0.0

    if normalized_input.size == 0:
        logger.error("predict_shape_with_model: empty input array")
        return 0.0

    if is_tflite and tflite_interpreter:
        logger.info("=== Using TFLite inference ===")
        try:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            if not input_details:
                logger.error("TFLite: no input_details")
                return 0.0
            if not output_details:
                logger.error("TFLite: no output_details")
                return 0.0

            arr = normalized_input.astype(np.float32)
            logger.info("TFLite final input shape: %s", arr.shape)

            tflite_interpreter.set_tensor(input_details[0]["index"], arr)
            tflite_interpreter.invoke()
            output = tflite_interpreter.get_tensor(output_details[0]["index"])

            logger.info("Raw TFLite output shape: %s", output.shape)
            logger.info("Raw TFLite output: %s", output)

            flat = np.asarray(output, dtype=np.float32).reshape(-1)
            if flat.size == 0:
                logger.error("TFLite: empty output array")
                return 0.0

            prob = float(flat[0])
            if not np.isfinite(prob):
                logger.error("TFLite: non-finite output %r, forcing 0.0", prob)
                prob = 0.0
            else:
                prob = max(0.0, min(1.0, prob))

            logger.info("FINAL TFLite CONFIDENCE: %f", prob)
            return prob
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("TFLite inference failed: %s", exc)
            return 0.0

    if model is not None:
        logger.info("=== Using Keras inference ===")
        try:
            logger.info("Keras model input shape: %s", getattr(model, "input_shape", None))
            logger.info("Keras model output shape: %s", getattr(model, "output_shape", None))

            preds = model.predict(normalized_input, verbose=0)
            logger.info("Raw Keras prediction shape: %s", preds.shape)
            logger.info("Raw Keras prediction: %s", preds)

            flat = np.asarray(preds, dtype=np.float32).reshape(-1)
            if flat.size == 0:
                logger.error("Keras: empty prediction array")
                return 0.0

            prob = float(flat[0])
            if not np.isfinite(prob):
                logger.error("Keras: non-finite output %r, forcing 0.0", prob)
                prob = 0.0
            else:
                prob = max(0.0, min(1.0, prob))

            logger.info("FINAL KERAS CONFIDENCE: %f", prob)
            return prob
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Keras prediction failed: %s", exc)
            return 0.0

    logger.error(
        "predict_shape_with_model called without a bound model/interpreter; returning 0.0."
    )
    return 0.0