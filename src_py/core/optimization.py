"""
Model quantization and optimization for deployment.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class ModelOptimizer:
    """Optimize models for deployment."""
    
    @staticmethod
    def quantize_to_tflite(model_path: str, output_path: str, representative_data: Optional[np.ndarray] = None):
        """
        Quantize model to TensorFlow Lite format.
        
        Args:
            model_path: Path to the trained model
            output_path: Output path for TFLite model
            representative_data: Data for calibration (for quantization)
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(
            tf.keras.models.load_model(model_path)
        )
        
        if representative_data is not None:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def representative_dataset():
                for data in representative_data:
                    yield [np.expand_dims(data, axis=0).astype(np.float32)]
            
            converter.representative_data = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
    
    @staticmethod
    def quantize_int8(model_path: str, output_path: str, representative_data: np.ndarray):
        """
        Quantize model to INT8 precision.
        
        Args:
            model_path: Path to the trained model
            output_path: Output path for quantized model
            representative_data: Calibration data
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(
            tf.keras.models.load_model(model_path)
        )
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset():
            for i in range(min(100, len(representative_data))):
                data = representative_data[i:i+1]
                yield [data.astype(np.float32)]
        
        converter.representative_data = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
    
    @staticmethod
    def pruning_magnitude(model, target_sparsity: float = 0.5):
        """
        Apply magnitude-based pruning to a model.
        
        Args:
            model: Keras model to prune
            target_sparsity: Target sparsity level (0-1)
        
        Returns:
            Pruned model
        """
        import tensorflow_model_optimization as tfmot
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity,
                begin_step=0,
                end_step=-1
            )
        }
        
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        return pruned_model
