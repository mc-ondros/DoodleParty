# ML Server Integration

This document describes the local ML server setup for content detection in DoodleParty.

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Browser   │  HTTP   │  ML Server   │  Model  │   Keras/    │
│  (Canvas)   │────────>│  (Flask)     │────────>│   ONNX/     │
│             │  POST   │  Port 5000   │  Infer  │   TFLite    │
└─────────────┘         └──────────────┘         └─────────────┘
       │
       │ WebSocket
       ▼
┌─────────────┐
│   Express   │
│   Server    │
│  Port 3000  │
└─────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors numpy Pillow
# Optional: TensorFlow for model loading
pip install tensorflow
```

### 2. Start Servers

Use the all-in-one startup script:

```bash
./start_with_ml.sh
```

This will:
- Start ML server on port 5000
- Start Express server on port 3000
- Create log files in `logs/`

### 3. Access the App

Open your browser to:
- **Web App**: http://localhost:3000
- **ML Server Health**: http://localhost:5000/health

## Manual Setup

### Start ML Server Only

```bash
python3 ml_server.py --port 5000 --model models/quickdraw_model_int8.tflite --threshold 0.7
```

Options:
- `--port`: Server port (default: 5000)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--model`: Path to model file (default: models/quickdraw_model_int8.tflite)
- `--threshold`: Confidence threshold for inappropriate classification (default: 0.7)
- `--debug`: Enable debug mode

### Start Express Server Only

```bash
cd express_canvas/server
node express-server.cjs
```

## API Reference

### POST /classify

Classify a drawing image for inappropriate content.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `image`: PNG image file
  - `sessionId`: (optional) User session ID
  - `bbox`: (optional) JSON string with bounding box

**Response:**
```json
{
  "is_inappropriate": false,
  "confidence": 0.23,
  "class_name": "appropriate",
  "mock": false
}
```

### GET /health

Check server health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
   "model_path": "models/quickdraw_model_int8.tflite"
}
```

### GET /config

Get current server configuration.

**Response:**
```json
{
  "threshold": 0.7,
   "model_path": "models/quickdraw_model_int8.tflite",
  "model_loaded": true
}
```

### POST /config

Update server configuration.

**Request:**
```json
{
  "threshold": 0.8
}
```

## Model Support

The ML server supports multiple model formats. The repository ships with the quantized QuickDraw TFLite blob, and the default requirements now install `tensorflow-cpu` so TFLite loading works without the separate runtime package.

### QuickDraw TFLite (default)
```bash
python3 ml_server.py --model models/quickdraw_model_int8.tflite
```

### Keras (.keras, .h5)
```bash
pip install tensorflow-cpu==2.18.1
python3 ml_server.py --model models/model_best.keras
```

### ONNX (.onnx)
```bash
pip install onnxruntime
python3 ml_server.py --model models/model_best.onnx
```

### Other TFLite models
Any `.tflite` file works because TensorFlow includes the interpreter. No extra runtime installation is needed beyond `tensorflow-cpu`.
```bash
python3 ml_server.py --model models/your_model.tflite
```

## Client Integration

The frontend (`doodleparty.js`) automatically connects to the ML server:

```javascript
// Every 3 strokes, check for inappropriate content
const ML_SERVER_URL = 'http://localhost:5000/classify';

async function classifyObject(extracted) {
    const blob = await new Promise((resolve) => {
        extracted.canvas.toBlob(resolve, 'image/png');
    });
    
    const formData = new FormData();
    formData.append('image', blob, 'drawing.png');
    
    const response = await fetch(ML_SERVER_URL, {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result.is_inappropriate;
}
```

## Testing Without a Model

If you don't have a trained model, the ML server will use mock predictions:

```bash
# Server will start in mock mode if model file doesn't exist
python3 ml_server.py --model models/nonexistent.tflite
```

Mock predictions:
- Randomly classify ~10% of images as inappropriate
- Used for testing the integration without a real model

## Configuration

### Adjust Detection Threshold

Higher threshold = fewer false positives, more false negatives:

```bash
python3 ml_server.py --threshold 0.9  # More strict
```

Lower threshold = more false positives, fewer false negatives:

```bash
python3 ml_server.py --threshold 0.5  # More sensitive
```

### Change ML Server URL

Edit `doodleparty.js`:

```javascript
const ML_SERVER_URL = 'http://your-server:5000/classify';
```

## Troubleshooting

### ML Server Won't Start

1. Check Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Check if port 5000 is available:
   ```bash
   lsof -i :5000
   ```

### CORS Errors

The server enables CORS by default. If you still see errors:

1. Check browser console for specific error
2. Verify ML_SERVER_URL matches the actual server address
3. Make sure Flask-CORS is installed

### Model Loading Errors

1. Verify model file exists:
   ```bash
   ls -lh models/quickdraw_model_int8.tflite
   ```

2. Check TensorFlow version compatibility
3. Try running in mock mode to test the pipeline

### Detection Not Working

1. Check browser console for errors
2. Verify ML server is running:
   ```bash
   curl http://localhost:5000/health
   ```

3. Check logs:
   ```bash
   tail -f logs/ml_server.log
   ```

4. Test classification manually:
   ```bash
   curl -F "image=@test.png" http://localhost:5000/classify
   ```

## Performance

### Optimization Tips

1. **Use TFLite for faster inference (default QuickDraw model):**
   ```bash
   python3 ml_server.py --model models/quickdraw_model_int8.tflite
   ```

2. **Run on GPU (if available):**
   ```bash
   pip install tensorflow[and-cuda]
   ```

3. **Adjust threshold for speed vs accuracy:**
   - Higher threshold = fewer server calls
   - Lower threshold = more detections

### Expected Performance

- Model inference: ~50-200ms (CPU)
- Total request: ~100-300ms (including network)
- Memory usage: ~500MB-2GB (depends on model)

## Production Deployment

### Using Gunicorn

For production, use Gunicorn instead of Flask dev server:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 ml_server:app
```

### Docker Deployment

```bash
# Build image
docker build -t doodleparty-ml .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models doodleparty-ml
```

## Logs

Logs are stored in `logs/`:

- `ml_server.log`: ML server activity
- `express_server.log`: Express server activity

View logs in real-time:

```bash
tail -f logs/ml_server.log
```

## Related Documentation

- [CONTENT_DETECTION.md](CONTENT_DETECTION.md) - Edge detection algorithm
- [QUICK_START.md](QUICK_START.md) - Content detection quick reference
- [README.md](README.md) - Main project documentation
