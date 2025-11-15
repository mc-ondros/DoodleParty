# Content Detection System

## Overview
Automated inappropriate content detection system that runs every 3 strokes to identify and remove problematic drawings.

## Architecture

### Client-Side (doodleparty.js)

#### 1. Detection Trigger
- **Frequency**: Every 3 completed strokes
- **Counter**: `strokesSinceLastCheck` tracks strokes since last check
- **Entry Point**: `endStroke()` function triggers `checkForInappropriateContent()`

#### 2. Image Processing Pipeline

##### Step 1: Canny Edge Detection
```
cannyEdgeDetection(imageData) → edges[]
```
- Converts canvas to grayscale
- Applies Gaussian blur (3x3 kernel)
- Calculates gradients using Sobel operator
- Applies threshold (50) to create binary edge map

##### Step 2: Object Detection
```
findConnectedComponents(edges, width, height) → components[]
```
- Uses flood-fill algorithm with 8-connectivity
- Filters out small components (< 50 pixels)
- Returns array of pixel coordinates for each object

##### Step 3: Bounding Box Calculation
```
getBoundingBox(component) → {minX, maxX, minY, maxY}
```
- Finds minimum enclosing rectangle for each object
- Returns coordinates of the smallest square that fits the object

##### Step 4: Image Extraction & Preparation
```
extractObjectForML(bbox) → {canvas, imageData, bbox}
```
- Extracts object region from main canvas
- Resizes to ML_INPUT_SIZE (128x128 or as configured)
- Maintains aspect ratio with white padding
- Centers object in the square

##### Step 5: ML Classification
```
classifyObject(extracted) → isInappropriate (boolean)
```
- Sends prepared image to ML model
- Waits for classification result
- Returns true if content is inappropriate

##### Step 6: Object Removal
```
deleteObject(bbox)
```
- If classified as inappropriate:
  - Fills bounding box with white color
  - Applies current zoom/pan transformations
  - Notifies server via socket event

### Server-Side Integration Points

#### Socket Events to Implement

##### 1. ML Classification Request
```javascript
socket.on('ml.classify', async (data) => {
    // data: { sessionId, imageData, bbox }
    // 1. Decode base64 image
    // 2. Run through ML model
    // 3. Get prediction score
    // 4. Emit result back
});
```

##### 2. ML Result Response
```javascript
socket.emit('ml.result', {
    isInappropriate: boolean,
    confidence: number,
    bbox: { minX, maxX, minY, maxY }
});
```

##### 3. Violation Notification
```javascript
socket.on('content.violation', (data) => {
    // data: { sessionId, bbox, timestamp }
    // 1. Log violation
    // 2. Update user stats
    // 3. Apply penalties if needed
});
```

## Configuration

### Constants (in doodleparty.js)
```javascript
const CONTENT_CHECK_INTERVAL = 3;  // Check every N strokes
const ML_INPUT_SIZE = 128;          // ML model input size (adjust as needed)
```

### Canny Edge Detection Parameters
```javascript
const GAUSSIAN_KERNEL = 3;         // Blur kernel size
const EDGE_THRESHOLD = 50;         // Edge detection threshold
const MIN_COMPONENT_SIZE = 50;     // Minimum pixels for object
```

## ML Model Requirements

### Input Format
- **Size**: ML_INPUT_SIZE × ML_INPUT_SIZE (default: 128×128)
- **Format**: RGB or Grayscale
- **Background**: White (#ffffff)
- **Object**: Centered, aspect-ratio preserved
- **Data**: Base64 PNG or ImageData

### Output Format
```javascript
{
    isInappropriate: boolean,
    confidence: number (0-1),
    categories: string[] (optional)
}
```

## Workflow Diagram

```
User draws stroke
       ↓
   endStroke()
       ↓
  Stroke count++
       ↓
  Count >= 3? → NO → Continue
       ↓ YES
  checkForInappropriateContent()
       ↓
  Get canvas ImageData
       ↓
  cannyEdgeDetection()
       ↓
  findConnectedComponents()
       ↓
  For each component:
       ↓
  getBoundingBox()
       ↓
  extractObjectForML()
       ↓
  Socket → 'ml.classify'
       ↓
  Wait for 'ml.result'
       ↓
  isInappropriate? → NO → Continue
       ↓ YES
  deleteObject()
       ↓
  Socket → 'content.violation'
       ↓
  Reset counter
```

## Performance Considerations

### Client-Side
- Edge detection runs in O(width × height)
- Flood-fill can be O(n) where n = pixels in component
- Runs asynchronously to avoid blocking UI
- 5-second timeout on ML classification

### Optimization Tips
1. Reduce canvas resolution before processing
2. Use Web Workers for heavy computation
3. Cache edge detection results
4. Implement debouncing on rapid strokes

## Error Handling

### Failure Modes
1. **ML timeout**: Defaults to `false` (not inappropriate)
2. **Socket disconnection**: Check skipped, logs error
3. **Canvas access error**: Caught and logged

### Logging
```javascript
console.log('Running content check...');
console.log('Found N objects');
console.warn('Inappropriate content detected!');
console.error('Content check failed:', err);
```

## Testing

### Manual Testing
1. Draw 3 strokes
2. Check console for "Running content check..."
3. Verify objects are detected
4. Test with known inappropriate patterns

### Debug Mode
Set breakpoints in:
- `checkForInappropriateContent()`
- `classifyObject()`
- `deleteObject()`

## Future Enhancements

1. **Real-time detection**: Check while drawing (with throttling)
2. **Object tracking**: Remember removed objects
3. **User warnings**: Show UI notification before removal
4. **Appeal system**: Allow users to contest removals
5. **Confidence threshold**: Only remove high-confidence detections
6. **Pattern caching**: Speed up repeated checks
