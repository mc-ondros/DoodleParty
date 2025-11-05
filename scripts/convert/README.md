Convert Keras model to ONNX using a Docker container (isolated environment)

This Docker image creates a clean environment with TensorFlow 2.11 and tf2onnx (1.16), avoiding protobuf conflicts with your main venv.

Build the image from the `scripts/convert` directory:

```bash
cd scripts/convert
docker build -t dhl-convert:tf2onnx .
```

Run the converter (mount your repository so the container can access `models/`):

```bash
# Convert to ONNX
docker run --rm -v $(pwd)/../../models:/models -v $(pwd):/workspace dhl-convert:tf2onnx \
  --model /models/quickdraw_model.h5 --output /models/quickdraw_model.onnx --opset 15

# After success, ONNX model will be in models/quickdraw_model.onnx
```

Notes:
- Use opset 13-15 for best compatibility with hardware runtimes. If conversion fails, try a different opset.
- If your model uses custom layers, import them in `convert_to_onnx.py` before loading the model.
- The container keeps system dependencies isolated and avoids changing your main Python environment.
