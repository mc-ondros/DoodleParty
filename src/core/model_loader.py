"""
Model loading and prediction utilities for content detection
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

TFLITE_RUNTIME_AVAILABLE = False
TF_AVAILABLE = False
ONNX_AVAILABLE = False

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available")

# Try to import TFLite runtime (lightweight interpreter)
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_RUNTIME_AVAILABLE = True
except ImportError:
    TFLITE_RUNTIME_AVAILABLE = False


# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not available")
    ONNX_AVAILABLE = False


def load_model(model_path):
    """
    Load a model from disk
    
    Supports:
        - Keras (.keras, .h5)
        - ONNX (.onnx)
        - TFLite (.tflite)
    
    Args:
        model_path: Path to model file
        
    Returns:
        Model object (format depends on file type)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext in ['.keras', '.h5']:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for loading Keras model")
        logger.info(f"Loading Keras model from {model_path}")
        model = keras.models.load_model(model_path)
        return {'type': 'keras', 'model': model}
    
    elif ext == '.onnx':
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available for loading ONNX model")
        logger.info(f"Loading ONNX model from {model_path}")
        session = ort.InferenceSession(model_path)
        return {'type': 'onnx', 'session': session}
    
    elif ext == '.tflite':
        if TFLITE_RUNTIME_AVAILABLE:
            logger.info(f"Loading TFLite model with tflite-runtime from {model_path}")
            interpreter = TFLiteInterpreter(model_path=model_path)
        else:
            if not TF_AVAILABLE:
                raise RuntimeError("Neither TensorFlow nor tflite-runtime is available for loading TFLite models")
            logger.info(f"Loading TFLite model with TensorFlow from {model_path}")
            interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return {'type': 'tflite', 'interpreter': interpreter}
    
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def predict(model_obj, image_array):
    """
    Run prediction on an image
    
    Args:
        model_obj: Model object from load_model()
        image_array: Preprocessed image array (batch_size, height, width, channels)
        
    Returns:
        dict with:
            - confidence: float (0-1)
            - class_name: str
            - raw_predictions: array
    """
    model_type = model_obj['type']
    
    if model_type == 'keras':
        return _predict_keras(model_obj['model'], image_array)
    
    elif model_type == 'onnx':
        return _predict_onnx(model_obj['session'], image_array)
    
    elif model_type == 'tflite':
        return _predict_tflite(model_obj['interpreter'], image_array)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _predict_keras(model, image_array):
    """Keras model prediction"""
    predictions = model.predict(image_array, verbose=0)
    
    # Assuming binary classification (appropriate vs inappropriate)
    # Output shape: (batch_size, 1) or (batch_size, 2)
    
    if predictions.shape[-1] == 1:
        # Single output neuron (sigmoid)
        confidence = float(predictions[0][0])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    else:
        # Two output neurons (softmax)
        # Assuming class 0 = appropriate, class 1 = inappropriate
        confidence = float(predictions[0][1])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    
    return {
        'confidence': confidence,
        'class_name': class_name,
        'raw_predictions': predictions.tolist()
    }


def _predict_onnx(session, image_array):
    """ONNX model prediction"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    predictions = session.run([output_name], {input_name: image_array})[0]
    
    # Similar logic to Keras
    if predictions.shape[-1] == 1:
        confidence = float(predictions[0][0])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    else:
        confidence = float(predictions[0][1])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    
    return {
        'confidence': confidence,
        'class_name': class_name,
        'raw_predictions': predictions.tolist()
    }


def _predict_tflite(interpreter, image_array):
    """TFLite model prediction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Similar logic to Keras
    if predictions.shape[-1] == 1:
        confidence = float(predictions[0][0])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    else:
        confidence = float(predictions[0][1])
        class_name = 'inappropriate' if confidence >= 0.5 else 'appropriate'
    
    return {
        'confidence': confidence,
        'class_name': class_name,
        'raw_predictions': predictions.tolist()
    }
