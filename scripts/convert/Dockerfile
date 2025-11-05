# Dockerfile to convert Keras model to ONNX in an isolated environment
# Uses TensorFlow 2.11 (compatible with tf2onnx) and installs tf2onnx

FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Install Python deps
# Pin to TF 2.11 and tf2onnx compatible with protobuf 3.20.x
RUN pip install --no-cache-dir "tensorflow==2.11.0" "tf2onnx==1.16.0" onnx numpy pillow

# Copy converter script
COPY convert_to_onnx.py /workspace/convert_to_onnx.py

ENTRYPOINT ["python", "/workspace/convert_to_onnx.py"]
