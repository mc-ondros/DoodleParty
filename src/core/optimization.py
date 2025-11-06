"""Model optimization utilities for DoodleHunter.

Provides functions for model optimization including:
- INT8 quantization for TensorFlow Lite models
- Weight pruning to reduce model size
- Knowledge distillation for smaller models
- Graph optimization for TensorFlow models

Related:
- scripts/convert/quantize_int8.py (INT8 quantization script)
- scripts/convert/prune_model.py (pruning script)
- scripts/convert/distill_model.py (distillation script)
- scripts/convert/optimize_graph.py (graph optimization script)

Exports:
- quantize_model_int8: Apply INT8 quantization
- prune_model: Apply weight pruning
- distill_model: Knowledge distillation
- optimize_graph: TensorFlow graph optimization
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Callable, Optional, Tuple


def quantize_model_int8(
    model_path: str,
    calibration_data: np.ndarray,
    output_path: Optional[str] = None
) -> str:
    """
    Apply INT8 post-training quantization to a model.
    
    Args:
        model_path: Path to model (.h5, .keras, or .tflite)
        calibration_data: Representative dataset for calibration
        output_path: Path to save quantized model (default: {model}_int8.tflite)
    
    Returns:
        Path to quantized model
    """
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_int8.tflite"
    else:
        output_path = Path(output_path)
    
    if model_path.suffix in ['.h5', '.keras']:
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def representative_dataset():
        for sample in calibration_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    
    return str(output_path)


def prune_model(
    model: tf.keras.Model,
    target_sparsity: float = 0.5,
    epochs: int = 10
) -> tf.keras.Model:
    """Apply magnitude-based weight pruning to reduce model size."""
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError(
            "tensorflow_model_optimization is required for pruning. "
            "Install with: pip install tensorflow-model-optimization"
        )
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=epochs * 100
        )
    }
    
    return tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)


def distill_model(
    teacher_model: tf.keras.Model,
    student_model: tf.keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    temperature: float = 3.0,
    alpha: float = 0.1,
    epochs: int = 10
) -> tf.keras.Model:
    """Knowledge distillation: train smaller student model from larger teacher."""
    X_train, y_train = train_data
    teacher_predictions = teacher_model.predict(X_train, verbose=0)
    
    def distillation_loss(y_true, y_pred):
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        soft_loss = tf.keras.losses.kullback_leibler_divergence(
            tf.nn.softmax(teacher_predictions / temperature),
            tf.nn.softmax(y_pred / temperature)
        )
        return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)
    
    student_model.compile(optimizer='adam', loss=distillation_loss, metrics=['accuracy'])
    student_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)
    
    return student_model


def optimize_graph(model_path: str, output_path: Optional[str] = None) -> str:
    """Optimize TensorFlow graph for inference."""
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_optimized{model_path.suffix}"
    else:
        output_path = Path(output_path)
    
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    output_path = output_path.with_suffix('.tflite')
    output_path.write_bytes(tflite_model)
    
    return str(output_path)


if __name__ == '__main__':
    print('Model Optimization Utilities')
    print('Use the scripts in scripts/convert/ for command-line tools.')
