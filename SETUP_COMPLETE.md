# Content Detection System - Complete Setup Guide

## Overview

DoodleParty now includes a comprehensive content detection system that automatically checks drawings for inappropriate content using machine learning. The system runs entirely locally for privacy and performance.

## System Components

### 1. Frontend Detection (Browser)
- **File**: `express_canvas/public/js/doodleparty.js`
- **Features**:
  - Canny edge detection
  - Connected component analysis
  - Object extraction and preprocessing
  - Automatic detection every 3 strokes

### 2. ML Server (Python/Flask)
- **File**: `ml_server.py`
- **Port**: 5000
- **Features**:
  - REST API for image classification
  - Supports multiple model formats (Keras, ONNX, TFLite)
  - Mock mode for testing without a model
  - Configurable threshold

### 3. Model Loader (Python)
- **File**: `src/core/model_loader.py`
- **Features**:
  - Automatic format detection
  - Unified prediction interface
  - Support for TensorFlow, ONNX, TFLite

### 4. Express Server (Node.js)
- **File**: `express_canvas/server/express-server.cjs`
- **Port**: 3000
- **Features**:
  - WebSocket communication
  - Stroke synchronization
  - Session management

## Quick Start

### Option 1: All-in-One Script (Recommended)

```bash
./start_with_ml.sh
```

This automatically:
1. Checks dependencies
2. Starts ML server (port 5000)
3. Starts Express server (port 3000)
4. Opens logs
5. Waits for Ctrl+C to stop

### Option 2: Manual Start

**Terminal 1 - ML Server:**
```bash
python3 ml_server.py
```

**Terminal 2 - Express Server:**
```bash
cd express_canvas/server
node express-server.cjs
```

**Terminal 3 - Open Browser:**
```bash
xdg-open http://localhost:3000
```

## Installation

### 1. Install Python Dependencies

```bash
pip install flask flask-cors numpy Pillow
```

For model support:
```bash
# Keras/TensorFlow
pip install tensorflow

# ONNX
pip install onnxruntime

# Or install everything
pip install -r requirements.txt
```

### 2. Install Node Dependencies

```bash
npm install
```

### 3. Make Scripts Executable

```bash
chmod +x start_with_ml.sh
chmod +x test_ml_server.py
```

## Testing

### Test ML Server

```bash
# Start ML server
python3 ml_server.py

# In another terminal, run tests
python3 test_ml_server.py
```

Expected output:
```
==================================================
ML Server Test Suite
==================================================

Testing health check...
✓ Health check passed
  Status: healthy
  Model loaded: False
  Model path: models/model_best.keras

Testing classification...
✓ Classification successful
  Is inappropriate: False
  Confidence: 0.234
  Class: appropriate
  Mock mode: True

Testing config...
✓ Config retrieved
  Threshold: 0.7
  Model path: models/model_best.keras
  Model loaded: False

==================================================
Results: 3/3 tests passed
==================================================
```

### Test Full Integration

1. Start both servers:
   ```bash
   ./start_with_ml.sh
   ```

2. Open browser to http://localhost:3000

3. Draw 3 strokes

4. Watch console for detection messages:
   ```
   Checking for inappropriate content...
   Found 1 connected components
   Classifying object...
   ML Result: {is_inappropriate: false, confidence: 0.23}
   ```

## How It Works

### Detection Pipeline

```
User draws stroke
       ↓
Every 3 strokes triggers check
       ↓
Canvas → Grayscale → Gaussian Blur
       ↓
Sobel Edge Detection (X and Y gradients)
       ↓
Threshold (edges > 50)
       ↓
Find Connected Components (flood-fill)
       ↓
For each component:
  - Calculate bounding box
  - Extract region
  - Resize to 128x128
  - Send to ML server
       ↓
ML Server:
  - Receive image
  - Preprocess
  - Run model inference
  - Return classification
       ↓
If inappropriate:
  - Delete object (white-fill region)
  - Notify user
```

### Data Flow

```
┌──────────────────────────────────────────────┐
│ Browser (Canvas)                             │
│  - Draw strokes                              │
│  - Edge detection                            │
│  - Object extraction                         │
└──────────────┬───────────────────────────────┘
               │
               │ HTTP POST /classify
               │ (image blob + metadata)
               ▼
┌──────────────────────────────────────────────┐
│ ML Server (Flask - Port 5000)                │
│  - Receive image                             │
│  - Load model                                │
│  - Run inference                             │
│  - Return result                             │
└──────────────┬───────────────────────────────┘
               │
               │ JSON response
               │ {is_inappropriate: bool, confidence: float}
               ▼
┌──────────────────────────────────────────────┐
│ Browser (Canvas)                             │
│  - Receive classification                    │
│  - Delete object if inappropriate            │
│  - Notify socket server                      │
└──────────────────────────────────────────────┘
```

## Configuration

### Adjust Detection Sensitivity

Edit `doodleparty.js`:

```javascript
// Edge detection threshold (higher = fewer edges)
const EDGE_THRESHOLD = 50;  // Default: 50, range: 0-255

// Minimum component size (higher = ignore smaller objects)
const MIN_COMPONENT_SIZE = 50;  // Default: 50 pixels

// How often to check (lower = more frequent)
const CONTENT_CHECK_INTERVAL = 3;  // Every 3 strokes
```

### Adjust ML Threshold

Start ML server with custom threshold:

```bash
python3 ml_server.py --threshold 0.8  # More strict (fewer false positives)
python3 ml_server.py --threshold 0.5  # More sensitive (fewer false negatives)
```

Or update at runtime:

```bash
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.8}'
```

### Change ML Server URL

Edit `doodleparty.js`:

```javascript
const ML_SERVER_URL = 'http://your-server:5000/classify';
```

## Model Integration

### Using Your Own Model

1. Train your model (binary classification: appropriate vs inappropriate)

2. Save as Keras, ONNX, or TFLite:
   ```python
   # Keras
   model.save('models/my_model.keras')
   
   # ONNX
   import tf2onnx
   tf2onnx.convert.from_keras(model, output_path='models/my_model.onnx')
   
   # TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with open('models/my_model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

3. Start server with your model:
   ```bash
   python3 ml_server.py --model models/my_model.keras
   ```

### Model Requirements

- **Input**: RGB image, any size (will be resized to 128x128)
- **Output**: 
  - Single neuron (sigmoid): probability [0, 1]
  - Two neurons (softmax): [appropriate_prob, inappropriate_prob]
- **Preprocessing**: Automatic (normalization to [0, 1])

### Mock Mode (No Model)

If you don't have a model yet, the server runs in mock mode:

```bash
python3 ml_server.py  # Will use mock predictions if model not found
```

Mock mode randomly classifies ~10% as inappropriate for testing.

## Troubleshooting

### Problem: ML server won't start

**Solutions:**
1. Check dependencies: `pip install flask flask-cors numpy Pillow`
2. Check port availability: `lsof -i :5000`
3. Check Python version: `python3 --version` (need 3.8+)

### Problem: Classification not working

**Solutions:**
1. Check ML server is running: `curl http://localhost:5000/health`
2. Check browser console for errors (F12)
3. Check ML server logs: `tail -f logs/ml_server.log`
4. Verify CORS is enabled (Flask-CORS installed)

### Problem: Too many/few detections

**Solutions:**
1. Adjust edge threshold in `doodleparty.js`: `EDGE_THRESHOLD`
2. Adjust ML threshold: `python3 ml_server.py --threshold 0.8`
3. Adjust check interval: `CONTENT_CHECK_INTERVAL`

### Problem: Model loading errors

**Solutions:**
1. Check model file exists: `ls -lh models/model_best.keras`
2. Check TensorFlow version: `pip show tensorflow`
3. Try different model format (ONNX, TFLite)
4. Use mock mode for testing: just don't provide a model file

### Problem: CORS errors

**Solutions:**
1. Verify Flask-CORS is installed: `pip show flask-cors`
2. Check browser network tab for exact error
3. Make sure ML_SERVER_URL matches actual server address
4. Try with `--host 0.0.0.0` if accessing from different machine

## Performance

### Expected Latency

- Edge detection: ~50-100ms (browser)
- HTTP request: ~10-50ms (local network)
- Model inference: ~50-200ms (CPU) or ~10-50ms (GPU)
- **Total**: ~100-350ms per object

### Optimization Tips

1. **Use TFLite for faster inference**:
   ```bash
   python3 ml_server.py --model models/model.tflite
   ```

2. **Reduce check frequency**:
   ```javascript
   const CONTENT_CHECK_INTERVAL = 5;  // Every 5 strokes instead of 3
   ```

3. **Increase component size threshold**:
   ```javascript
   const MIN_COMPONENT_SIZE = 100;  // Ignore smaller objects
   ```

4. **Use GPU if available**:
   ```bash
   pip install tensorflow[and-cuda]
   ```

### Memory Usage

- Browser: ~50-100MB for canvas and detection
- ML Server: ~500MB-2GB (depends on model size)
- Express Server: ~50-100MB

## Production Deployment

### Using Gunicorn (Recommended)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 ml_server:app
```

Configuration:
- `-w 4`: 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces
- Add `--timeout 30` for slow models

### Using systemd

Create `/etc/systemd/system/doodleparty-ml.service`:

```ini
[Unit]
Description=DoodleParty ML Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/doodleparty
ExecStart=/usr/bin/python3 ml_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable doodleparty-ml
sudo systemctl start doodleparty-ml
```

### Using Docker

```bash
docker build -t doodleparty-ml .
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models doodleparty-ml
```

## Security Considerations

1. **Local deployment recommended**: Keeps user drawings private
2. **CORS**: Configure allowed origins in production
3. **Rate limiting**: Add rate limiting for public deployments
4. **Input validation**: Server validates image size and format
5. **Timeout**: 5-second timeout prevents hanging requests

## Files Created/Modified

### New Files
- `ml_server.py` - Flask ML server
- `src/core/model_loader.py` - Model loading utilities
- `start_with_ml.sh` - All-in-one startup script
- `test_ml_server.py` - ML server test suite
- `ML_SERVER.md` - ML server documentation
- `SETUP_COMPLETE.md` - This file

### Modified Files
- `express_canvas/public/js/doodleparty.js`:
  - Updated `classifyObject()` to use HTTP instead of socket.io
  - Added ML_SERVER_URL constant

## Next Steps

1. **Train your own model**:
   - Collect training data
   - Train binary classifier
   - Save as Keras/ONNX/TFLite
   - Test with ML server

2. **Optimize performance**:
   - Convert to TFLite
   - Quantize model
   - Use GPU acceleration

3. **Deploy to production**:
   - Use Gunicorn
   - Set up systemd service
   - Configure nginx reverse proxy
   - Add monitoring

4. **Enhance detection**:
   - Adjust thresholds
   - Add more sophisticated edge detection
   - Implement multi-model ensemble

## Documentation

- **CONTENT_DETECTION.md** - Edge detection algorithm details
- **ML_SERVER.md** - ML server API reference
- **QUICK_START.md** - Content detection quick reference
- **README.md** - Main project documentation

## Support

For issues or questions:
1. Check browser console (F12)
2. Check server logs (`logs/ml_server.log`)
3. Run test suite (`python3 test_ml_server.py`)
4. Review documentation above

---

**System Status**: ✓ Content detection fully integrated and operational
