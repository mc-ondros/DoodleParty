# Dockerfile to convert Keras model to ONNX in an isolated environment
# Uses TensorFlow 2.11 (compatible with tf2onnx) and installs tf2onnx

FROM tensorflow/tensorflow:2.11.0

# Use the TF image which already includes a compatible TensorFlow wheel.
# Install minimal system deps and tf2onnx + onnx + helpers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install tf2onnx and onnx. Pin tf2onnx to 1.16.0 for compatibility.
RUN pip install --no-cache-dir tf2onnx==1.16.0 onnx numpy pillow

# Copy converter script into the image
COPY convert_to_onnx.py /workspace/convert_to_onnx.py

ENTRYPOINT ["python", "/workspace/convert_to_onnx.py"]
