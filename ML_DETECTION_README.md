# DoodleParty ML Detection System

## Overview

The ML detection system uses Canny edge detection to identify objects in drawings and sends them to a TensorFlow Lite model for classification.

## Architecture

```
┌─────────────────┐
│  Web Browser    │
│  (doodleparty)  │
└────────┬────────┘
         │ Socket.IO
         ▼
┌─────────────────┐      ┌──────────────┐
│ Express Server  │◄────►│  ML Server   │
│   (Node.js)     │      │  (Python)    │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│ Stroke History  │      │ TFLite Model │
│ Image Storage   │      │ Visualizations│
└─────────────────┘      └──────────────┘
```

## Quick Start

### 1. Setup (First Time Only)

```bash
./setup_ml.sh
```

This will:
- Create/activate virtual environment
- Install all Python dependencies
- Verify installation

### 2. Start All Services

```bash
./start_doodleparty.sh
```

This starts:
- **Express Server** on `http://localhost:3000`
- **ML Server** (connects via Socket.IO)

### 3. Use the Application

1. Open browser: `http://localhost:3000/doodleparty`
2. Draw something on the canvas
3. Click the **🔍 Detect** button
4. Check console for ML results

## Features

### Object Detection Pipeline

1. **Canny Edge Detection**
   - Gaussian blur (σ=1.4)
   - Sobel gradient calculation
   - Non-maximum suppression
   - Hysteresis thresholding (30/60)

2. **Object Extraction**
   - Flood fill for connected components
   - Minimum object size: 100 pixels
   - Square bounding box with padding (20px)

3. **Preprocessing**
   - Resize to 128x128 without distortion
   - Black padding to maintain aspect ratio
   - Center object in frame

4. **ML Inference**
   - TFLite INT8 quantized model
   - Binary classification (positive/negative)
   - Confidence scores

5. **Visualization**
   - Grid layout of detected objects
   - Color-coded predictions (red/green)
   - Confidence percentages
   - Saved to `data/ml_visualizations/`

## File Structure

```
DoodleParty/
├── start_doodleparty.sh      # Unified startup script
├── setup_ml.sh                # Setup/installation script
├── ml_server.py               # ML inference server (Python)
├── requirements-ml-server.txt # Python dependencies
│
├── express_canvas/
│   ├── server/
│   │   └── express-server.cjs # Node.js server
│   └── public/
│       ├── doodleparty.html   # Main UI
│       └── js/
│           └── doodleparty.js # Client code with Canny detection
│
├── models/
│   └── quickdraw_model_int8.tflite  # TFLite model
│
├── data/
│   ├── ml_detections/         # Saved detection images
│   │   └── [session_id]/
│   │       └── object_*.png
│   └── ml_visualizations/     # Result visualizations
│       └── [session]_[timestamp].png
│
└── logs/
    ├── express.log            # Express server logs
    └── ml.log                 # ML server logs
```

## Socket.IO Events

### Client → Express

- `ml.detectObjects` - Send detected objects for inference
  ```javascript
  {
    sessionId: string,
    objects: [{
      image: base64_png,
      boundingBox: {x1, y1, x2, y2, centerX, centerY},
      index: number
    }],
    timestamp: number
  }
  ```

### Express → ML Server

- `ml.detectObjects` - Forward objects to ML server (same payload)

### ML Server → Express

- `ml.detectionResults` - Return inference results
  ```javascript
  {
    success: boolean,
    sessionId: string,
    results: [{
      prediction: float,
      class: 'positive'|'negative'|'error',
      confidence: float
    }],
    summary: {
      total: number,
      positive: number,
      negative: number
    },
    visualization: string  // path to saved image
  }
  ```

### Express → Client

- `ml.detectionResults` - Broadcast results to all clients

## Development

### Testing Detection Algorithm

```javascript
// In browser console
const objects = detectObjectsInMLCanvas();
console.log('Detected objects:', objects);
```

### Manual ML Request

```javascript
// In browser console
sendObjectsToML();
```

### Check Logs

```bash
# Express server
tail -f logs/express.log

# ML server
tail -f logs/ml.log
```

### View Visualizations

```bash
open data/ml_visualizations/  # macOS
xdg-open data/ml_visualizations/  # Linux
explorer data/ml_visualizations/  # Windows
```

## Configuration

### ML Server

Edit `ml_server.py`:
- `MODEL_PATH` - Path to TFLite model
- `EXPRESS_URL` - Express server URL (default: localhost:3000)

### Detection Parameters

Edit `doodleparty.js` - `extractObjectsForML()`:
- `padding` - Padding around objects (default: 20px)
- `targetSize` - Output size (default: 128x128)

Edit `doodleparty.js` - `findConnectedComponents()`:
- `minSize` - Minimum pixels for valid object (default: 100)

### Canny Parameters

Edit `doodleparty.js` - `hysteresisThreshold()`:
- `lowThreshold` - Low threshold (default: 30)
- `highThreshold` - High threshold (default: 60)

## Troubleshooting

### ML Server Won't Start

1. Check Python dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements-ml-server.txt
   ```

2. Check TFLite model exists:
   ```bash
   ls -lh models/quickdraw_model_int8.tflite
   ```

3. View ML server logs:
   ```bash
   cat logs/ml.log
   ```

### No Objects Detected

- Draw more strokes (minimum 100 pixels)
- Check ML canvas is receiving strokes (check console)
- Verify strokes are white on black background

### Express Server Connection Failed

1. Check if port 3000 is available:
   ```bash
   lsof -i :3000
   ```

2. Start Express manually:
   ```bash
   cd express_canvas/server
   node express-server.cjs
   ```

## Performance

- **Edge Detection**: ~50-100ms for 1920x1080 canvas
- **Object Extraction**: ~10-50ms per object
- **ML Inference**: ~5-20ms per object (TFLite INT8)
- **Total Pipeline**: ~100-300ms for typical drawing

## License

See main repository LICENSE file.
