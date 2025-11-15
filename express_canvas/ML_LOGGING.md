# ML Detection Logging & Visualization

## Overview
Enhanced logging and visualization system for the ML content detection pipeline.

## Features Added

### 1. Client-Side Visual Debugging (`doodleparty.js`)

#### **Detection Log Panel**
- Real-time log panel showing all ML detection activity
- Color-coded messages:
  - 🟢 **Green (info)**: General information
  - 🟡 **Yellow (warn)**: Warnings and inappropriate content detections
  - 🔵 **Cyan (success)**: Successful operations
  - 🔴 **Red (error)**: Errors

#### **Keyboard Shortcuts**
- **Press `D`**: Toggle detection log panel on/off
- **Press `C`**: Manually trigger content check

#### **Visual Indicators**
- **Checking Overlay**: Shows progress when content check is running
- **Detection Popup**: Shows extracted image with classification result
  - Green border = Safe
  - Red border = Inappropriate
  - Displays confidence, bbox, size, category
  - Auto-disappears after 3 seconds

#### **Logged Information**
```
✏️ Stroke completed (2 until next check)
🔍 Starting content check...
✓ Edge detection completed in 45.23ms
✓ Found 3 objects in 12.45ms
📦 Processing object 1/3
  - bbox: (100,50) → (250,200)
  - size: 150×150px
  - pixels: 1234
📡 Sending to ML server...
  - url: http://localhost:5000/classify
  - imageSize: 128×128
  - bbox: {...}
📦 Image blob created: 3.42KB
✓ ML server response received (125.34ms)
  - isInappropriate: false
  - confidence: 45.2%
  - category: safe
  - mock: true
✓ Object 1 classified as safe
✅ Content check complete
  - totalObjects: 3
  - processed: 3
  - inappropriate: 0
  - safe: 3
  - totalTime: 234.56ms
```

### 2. Server-Side Enhanced Logging (`ml_server.py`)

#### **Startup Banner**
```
============================================================
🤖 ML CONTENT DETECTION SERVER
============================================================
📁 Model path: ./models/content_detector.h5
📏 Input size: 128×128
🎯 Confidence threshold: 0.7
🌐 Server port: 5000
============================================================
⚠️  WARNING: No model loaded - using MOCK predictions
   To use real predictions, set ML_MODEL_PATH to a valid model file
============================================================
🚀 Server starting on http://0.0.0.0:5000
📊 Health check: http://localhost:5000/health
🔍 Classify endpoint: http://localhost:5000/classify
============================================================
Press Ctrl+C to stop
```

#### **Request Logging**
Each classification request logs:
```
============================================================
📥 CLASSIFICATION REQUEST
Session: session_1731679234_abc123
Image size: (128, 128), mode: RGB
Bounding box: {'minX': 100, 'maxX': 250, 'minY': 50, 'maxY': 200}
✓ Preprocessing completed in 5.23ms
  Array shape: (1, 128, 128, 3), dtype: float32
✓ Prediction completed in 45.67ms
⚠️ RESULT: INAPPROPRIATE
  Confidence: 0.856 (85.6%)
  Category: inappropriate
  Mock: True
  Total time: 52.45ms
============================================================
```

#### **Health Endpoint Enhanced**
```json
GET /health
{
  "status": "healthy",
  "model": {
    "loaded": false,
    "type": null,
    "path": "./models/content_detector.h5",
    "exists": false,
    "input_shape": null
  },
  "config": {
    "input_size": 128,
    "threshold": 0.7,
    "port": 5000
  },
  "endpoints": {
    "classify": "/classify (POST)",
    "batch_classify": "/batch_classify (POST)",
    "health": "/health (GET)"
  }
}
```

### 3. Performance Metrics

All operations now include timing information:
- Edge detection time
- Component finding time
- Individual object classification time
- Network request/response time
- Total content check time
- Server preprocessing time
- Server inference time

### 4. Configuration

Enable/disable visual debugging:
```javascript
const ML_DEBUG_VISUAL = true; // Set to false to disable popups and log panel
```

Control check frequency:
```javascript
const CONTENT_CHECK_INTERVAL = 3; // Check every N strokes
```

### 5. Usage

#### **During Development**
1. Open the canvas page
2. Press **`D`** to show the detection log
3. Draw 3 strokes to trigger automatic check
4. Or press **`C`** to manually trigger check
5. Watch the log panel for real-time updates
6. See popup visualizations of detected objects

#### **Monitoring ML Server**
1. Check health: `curl http://localhost:5000/health`
2. Watch server console for detailed request logs
3. Monitor response times and detection results

#### **Debugging Issues**
- Check browser console for JavaScript errors
- Check server logs for Python errors
- Verify ML server is running: `http://localhost:5000/health`
- Look for error messages in detection log panel
- Check network tab for failed ML server requests

### 6. Example Flow

```
User draws stroke 1 → Log: "Stroke completed (2 until next check)"
User draws stroke 2 → Log: "Stroke completed (1 until next check)"
User draws stroke 3 → Log: "Stroke completed (0 until next check)"
                    → Log: "🔍 Starting content check..."
                    → Overlay: "🔍 Checking content..."
                    → Edge detection runs
                    → Component detection finds 2 objects
                    → Overlay: "🔍 Analyzing 2 objects..."
                    → Object 1 extracted, sent to ML server
                    → Server logs request
                    → Server returns result
                    → Popup shows object 1 result (3 sec)
                    → Object 2 processed similarly
                    → Log: "✅ Content check complete"
                    → Overlay disappears
```

## Benefits

✅ **Transparency**: See exactly what's being detected and sent
✅ **Debugging**: Quickly identify issues in the ML pipeline
✅ **Performance**: Monitor processing times at each step
✅ **Verification**: Visually confirm extracted objects match expectations
✅ **Monitoring**: Track server health and model status
✅ **Development**: Easier to test and iterate on ML features
