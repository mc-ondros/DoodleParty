# Server-Side ML Integration Guide

## Overview
This guide explains how to integrate the ML model for inappropriate content detection with the Express/Socket.IO server.

## Server Setup (express-server.cjs)

### 1. Add ML Classification Socket Handler

```javascript
const path = require('path');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const sharp = require('sharp'); // For image processing
// const tf = require('@tensorflow/tfjs-node'); // If using TensorFlow
// OR
// const onnx = require('onnxruntime-node'); // If using ONNX

// Load your ML model
let mlModel;

async function loadMLModel() {
    // TensorFlow example:
    // mlModel = await tf.loadLayersModel('file://./models/content_detector/model.json');
    
    // ONNX example:
    // const session = await onnx.InferenceSession.create('./models/content_detector.onnx');
    // mlModel = session;
    
    console.log('ML model loaded successfully');
}

// Load model on server start
loadMLModel().catch(err => console.error('Failed to load ML model:', err));

// Socket event handlers
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    // ML Classification handler
    socket.on('ml.classify', async (data) => {
        try {
            const { sessionId, imageData, bbox } = data;
            
            console.log('ML classification request:', {
                sessionId,
                bbox,
                timestamp: new Date().toISOString()
            });
            
            // 1. Decode base64 image
            const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
            const imageBuffer = Buffer.from(base64Data, 'base64');
            
            // 2. Preprocess image
            const processedImage = await preprocessImage(imageBuffer);
            
            // 3. Run ML inference
            const prediction = await runMLInference(processedImage);
            
            // 4. Determine if inappropriate
            const isInappropriate = prediction.score > 0.7; // Threshold
            
            // 5. Send result back
            socket.emit('ml.result', {
                isInappropriate,
                confidence: prediction.score,
                bbox,
                category: prediction.category
            });
            
            // 6. Log result
            console.log('ML Result:', {
                sessionId,
                isInappropriate,
                confidence: prediction.score
            });
            
        } catch (error) {
            console.error('ML classification error:', error);
            // Send safe default (not inappropriate) on error
            socket.emit('ml.result', {
                isInappropriate: false,
                confidence: 0,
                error: true
            });
        }
    });
    
    // Content violation handler
    socket.on('content.violation', async (data) => {
        const { sessionId, bbox, timestamp } = data;
        
        console.warn('Content violation detected:', {
            sessionId,
            bbox,
            timestamp: new Date(timestamp).toISOString()
        });
        
        // TODO: Store violation in database
        // TODO: Update user stats
        // TODO: Apply penalties if needed
        // TODO: Send notification to moderators
        
        // Example: Store in database
        // await db.violations.insert({
        //     sessionId,
        //     bbox,
        //     timestamp,
        //     userAgent: socket.handshake.headers['user-agent']
        // });
    });
});
```

### 2. Image Preprocessing Function

```javascript
async function preprocessImage(imageBuffer) {
    // Resize and normalize image for ML model
    const processedBuffer = await sharp(imageBuffer)
        .resize(64, 64, {
            fit: 'contain',
            background: { r: 255, g: 255, b: 255, alpha: 1 }
        })
        .grayscale() // If model expects grayscale
        .raw()
        .toBuffer();
    
    // Convert to normalized array (0-1)
    const pixels = new Float32Array(processedBuffer.length);
    for (let i = 0; i < processedBuffer.length; i++) {
        pixels[i] = processedBuffer[i] / 255.0;
    }
    
    return pixels;
}
```

### 3. ML Inference Function

#### Option A: TensorFlow.js
```javascript
async function runMLInference(imageData) {
    if (!mlModel) {
        throw new Error('ML model not loaded');
    }
    
    // Reshape to model input shape (e.g., [1, 64, 64, 1] for grayscale)
    const tensor = tf.tensor4d(imageData, [1, 64, 64, 1]);
    
    // Run prediction
    const prediction = await mlModel.predict(tensor);
    const scores = await prediction.data();
    
    // Cleanup
    tensor.dispose();
    prediction.dispose();
    
    return {
        score: scores[0], // Probability of inappropriate content
        category: scores[0] > 0.7 ? 'inappropriate' : 'safe'
    };
}
```

#### Option B: ONNX Runtime
```javascript
async function runMLInference(imageData) {
    if (!mlModel) {
        throw new Error('ML model not loaded');
    }
    
    // Create tensor
    const inputTensor = new onnx.Tensor('float32', imageData, [1, 64, 64, 1]);
    
    // Run inference
    const results = await mlModel.run({ input: inputTensor });
    const outputData = results.output.data;
    
    return {
        score: outputData[0],
        category: outputData[0] > 0.7 ? 'inappropriate' : 'safe'
    };
}
```

#### Option C: External API
```javascript
async function runMLInference(imageBuffer) {
    // Call external ML API
    const response = await fetch('https://your-ml-api.com/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.ML_API_KEY}`
        },
        body: JSON.stringify({
            image: imageBuffer.toString('base64')
        })
    });
    
    const result = await response.json();
    
    return {
        score: result.inappropriate_score,
        category: result.category
    };
}
```

## Package Dependencies

### Add to package.json
```json
{
  "dependencies": {
    "sharp": "^0.33.0",
    "@tensorflow/tfjs-node": "^4.15.0"
  }
}
```

Or for ONNX:
```json
{
  "dependencies": {
    "sharp": "^0.33.0",
    "onnxruntime-node": "^1.16.0"
  }
}
```

## Model Integration

### Directory Structure
```
express_canvas/
├── server/
│   ├── express-server.cjs
│   └── models/
│       ├── content_detector/
│       │   ├── model.json         (TensorFlow)
│       │   ├── weights.bin        (TensorFlow)
│       │   └── model.onnx         (ONNX)
│       └── README.md
```

### Model File Placement

1. **TensorFlow Model**: Place `model.json` and weight files in `models/content_detector/`
2. **ONNX Model**: Place `.onnx` file in `models/`
3. **Update paths** in `loadMLModel()` function

## Environment Variables

Create `.env` file:
```bash
ML_MODEL_PATH=./models/content_detector
ML_CONFIDENCE_THRESHOLD=0.7
ML_API_KEY=your_api_key_if_using_external_service
```

Load in server:
```javascript
require('dotenv').config();

const ML_THRESHOLD = parseFloat(process.env.ML_CONFIDENCE_THRESHOLD) || 0.7;
```

## Testing

### Manual Test
```bash
# Start server
npm start

# In browser console:
const testImage = canvas.toDataURL('image/png');
socket.emit('ml.classify', {
    sessionId: 'test-session',
    imageData: testImage,
    bbox: { minX: 0, maxX: 100, minY: 0, maxY: 100 }
});

socket.on('ml.result', (result) => {
    console.log('Result:', result);
});
```

### Unit Test
```javascript
const io = require('socket.io-client');
const fs = require('fs');

describe('ML Classification', () => {
    it('should classify test image', (done) => {
        const socket = io('http://localhost:3000');
        const testImage = fs.readFileSync('./test/test-image.png', 'base64');
        
        socket.on('connect', () => {
            socket.emit('ml.classify', {
                sessionId: 'test',
                imageData: `data:image/png;base64,${testImage}`,
                bbox: { minX: 0, maxX: 100, minY: 0, maxY: 100 }
            });
        });
        
        socket.on('ml.result', (result) => {
            expect(result).toHaveProperty('isInappropriate');
            expect(result).toHaveProperty('confidence');
            socket.disconnect();
            done();
        });
    });
});
```

## Performance Optimization

### 1. Request Queue
```javascript
const Queue = require('bull');
const mlQueue = new Queue('ml-classification');

mlQueue.process(async (job) => {
    const { imageData } = job.data;
    return await runMLInference(imageData);
});

socket.on('ml.classify', (data) => {
    mlQueue.add(data).then(job => {
        job.finished().then(result => {
            socket.emit('ml.result', result);
        });
    });
});
```

### 2. Caching
```javascript
const NodeCache = require('node-cache');
const mlCache = new NodeCache({ stdTTL: 300 }); // 5 min cache

socket.on('ml.classify', async (data) => {
    const cacheKey = crypto.createHash('md5').update(data.imageData).digest('hex');
    
    let result = mlCache.get(cacheKey);
    if (!result) {
        result = await runMLInference(data.imageData);
        mlCache.set(cacheKey, result);
    }
    
    socket.emit('ml.result', result);
});
```

## Monitoring & Logging

```javascript
const winston = require('winston');

const logger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    transports: [
        new winston.transports.File({ filename: 'ml-violations.log' })
    ]
});

socket.on('content.violation', (data) => {
    logger.warn('Content violation', {
        sessionId: data.sessionId,
        timestamp: new Date(data.timestamp),
        bbox: data.bbox
    });
});
```

## Security Considerations

1. **Rate Limiting**: Prevent ML abuse
```javascript
const rateLimit = require('express-rate-limit');

const mlLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10 // Max 10 requests per minute
});
```

2. **Input Validation**: Verify image data
3. **Sanitization**: Clean base64 strings
4. **Authentication**: Verify session IDs
5. **Resource Limits**: Cap image size

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check file paths
   - Verify model format compatibility
   - Check console for errors

2. **Slow inference**
   - Use GPU acceleration if available
   - Reduce model size
   - Implement queuing

3. **Socket timeout**
   - Increase timeout in client code
   - Add progress events
   - Implement retry logic

4. **Memory leaks**
   - Dispose tensors after use
   - Clear buffers
   - Monitor with `process.memoryUsage()`
