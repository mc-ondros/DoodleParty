# Content Detection - Quick Start Guide

## What Was Implemented

### Client-Side (doodleparty.js)
✅ Canny edge detection algorithm
✅ Connected component detection (object finder)
✅ Bounding box calculation
✅ Image extraction and ML preprocessing
✅ Object deletion (white-out)
✅ Socket communication for ML requests
✅ Automatic trigger every 3 strokes

### Documentation
✅ CONTENT_DETECTION.md - Complete system overview
✅ ML_SERVER_INTEGRATION.md - Server implementation guide
✅ This quick start guide

## How It Works

1. **User draws 3 strokes** → Triggers content check
2. **Canny edge detector** → Finds edges in the drawing
3. **Object finder** → Groups edges into distinct objects
4. **For each object**:
    - Calculate bounding box (smallest square that fits it)
    - Extract and resize to 128x128 (or your ML size)
   - Send to ML model via socket
   - Wait for result
   - If inappropriate: delete it (fill with white)

## Next Steps

### 1. Prepare Your ML Model
You need a model that:
- **Input**: 128x128 image (or adjust `ML_INPUT_SIZE`)
- **Output**: Classification score (0-1)
- **Format**: TensorFlow.js, ONNX, or REST API

### 2. Implement Server-Side
Follow `ML_SERVER_INTEGRATION.md`:
```javascript
// In express-server.cjs
socket.on('ml.classify', async (data) => {
    // Decode image
    // Run ML model
    // Send result back
    socket.emit('ml.result', { isInappropriate: true/false });
});
```

### 3. Test It
```javascript
// In browser console:
// Draw 3 strokes and watch console for:
"Running content check..."
"Found N objects"
"Classifying object..."
"ML Result: ..."
```

## Configuration

### Change Check Frequency
```javascript
// In doodleparty.js
const CONTENT_CHECK_INTERVAL = 5; // Check every 5 strokes instead of 3
```

### Change ML Input Size
```javascript
const ML_INPUT_SIZE = 128; // Use 128x128 instead of 64x64
```

### Change Edge Detection Sensitivity
```javascript
// In cannyEdgeDetection()
const threshold = 30; // Lower = more sensitive (default: 50)
```

### Change Minimum Object Size
```javascript
// In findConnectedComponents()
if (component.length > 100) { // Larger = ignore small objects (default: 50)
```

## Debugging

### Enable Detailed Logging
All console logs are already in place:
- `console.log('Running content check...')`
- `console.log('Found N objects')`
- `console.log('Classifying object...')`
- `console.log('ML Result:', result)`
- `console.warn('Inappropriate content detected!')`

### Visualize Edge Detection
Add this to `checkForInappropriateContent()`:
```javascript
// Create debug canvas to see edges
const debugCanvas = document.createElement('canvas');
debugCanvas.width = canvas.width;
debugCanvas.height = canvas.height;
const debugCtx = debugCanvas.getContext('2d');
const debugImageData = debugCtx.createImageData(canvas.width, canvas.height);

for (let i = 0; i < edges.length; i++) {
    debugImageData.data[i * 4] = edges[i];
    debugImageData.data[i * 4 + 1] = edges[i];
    debugImageData.data[i * 4 + 2] = edges[i];
    debugImageData.data[i * 4 + 3] = 255;
}

debugCtx.putImageData(debugImageData, 0, 0);
document.body.appendChild(debugCanvas); // Shows edge map
```

### Test Without ML Model
The code includes a timeout fallback:
```javascript
// If ML doesn't respond in 5 seconds, defaults to "not inappropriate"
setTimeout(() => resolve(false), 5000);
```

## Socket Events Reference

### Client → Server
```javascript
// Request ML classification
socket.emit('ml.classify', {
    sessionId: 'session_xxx',
    imageData: 'data:image/png;base64,...',
    bbox: { minX, maxX, minY, maxY }
});

// Report violation
socket.emit('content.violation', {
    sessionId: 'session_xxx',
    bbox: { minX, maxX, minY, maxY },
    timestamp: 1234567890
});
```

### Server → Client
```javascript
// ML classification result
socket.emit('ml.result', {
    isInappropriate: boolean,
    confidence: 0.95,
    bbox: { minX, maxX, minY, maxY }
});
```

## Performance Notes

- Edge detection: ~10-50ms on 1024x640 canvas
- Component finding: ~5-20ms
- Image extraction: ~5-10ms
- ML inference: Depends on your model (typically 50-500ms)
- Total: ~100-600ms per check

Runs asynchronously, doesn't block drawing!

## Common Issues

### "No objects found"
- Drawing might be too light/thin
- Adjust edge detection threshold
- Check minimum component size

### "ML timeout"
- Server not responding
- Implement server-side handler first
- Check socket connection

### "Objects not deleted"
- Check console for errors
- Verify bbox coordinates
- Check canvas transformation state

### "Too many/few checks"
- Adjust `CONTENT_CHECK_INTERVAL`
- Check `strokesSinceLastCheck` counter

## Files Modified

- ✅ `/express_canvas/public/js/doodleparty.js` (270 lines added)
- ✅ `/express_canvas/CONTENT_DETECTION.md` (new)
- ✅ `/express_canvas/ML_SERVER_INTEGRATION.md` (new)
- ✅ `/express_canvas/QUICK_START.md` (this file)

## Ready to Go!

The client-side detection is complete and ready to use. Just implement the server-side ML handler and you're done!

For questions, see:
- Full details: `CONTENT_DETECTION.md`
- Server setup: `ML_SERVER_INTEGRATION.md`
