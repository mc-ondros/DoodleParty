// DoodleParty Socket Sender - Combines user/index.html drawing interface with socket.io

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const brushSizeInput = document.getElementById('brushSize');
const clearBtn = document.getElementById('clearBtn');
const sendStrokeBtn = document.getElementById('sendStrokeBtn');
const sendBatchBtn = document.getElementById('sendBatchBtn');
const socketStatus = document.getElementById('socketStatus');
const statusDot = document.getElementById('statusDot');
const inkFill = document.getElementById('inkFill');
const timerDisplay = document.getElementById('timer');
const debugDisableTimerToggle = document.getElementById('debugDisableTimer');

// Constants
const CANVAS_BACKGROUND = '#ffffff';
const ROUND_DURATION_SECONDS = 90;
const INITIAL_INK = 10000; // High value for testing ML detection
const INK_CONSUMPTION_RATE = 0.04;
const QT_SCALE = 255; // QuickDraw coordinate scale
const DEBUG_MODE = false; // Set to true to show manual send buttons
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4;
const INITIAL_ZOOM = 2; // Start zoomed in
const SESSION_STORAGE_KEY = 'doodleparty_session';

// Session Management
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function getOrCreateSessionId() {
    let sessionId = sessionStorage.getItem('sessionId');
    if (!sessionId) {
        sessionId = generateSessionId();
        sessionStorage.setItem('sessionId', sessionId);
        console.log('Created new session:', sessionId);
    } else {
        console.log('Restored session:', sessionId);
    }
    return sessionId;
}

function saveSessionState() {
    const sessionState = {
        sessionId: getOrCreateSessionId(),
        inkAmount,
        remainingTime,
        isLocked,
        zoomLevel,
        offsetX,
        offsetY,
        timestamp: Date.now()
    };
    
    try {
        localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionState));
        console.log('Session state saved');
    } catch (e) {
        console.error('Failed to save session state:', e);
    }
}

function restoreSessionState() {
    try {
        const savedState = localStorage.getItem(SESSION_STORAGE_KEY);
        if (!savedState) return false;
        
        const sessionState = JSON.parse(savedState);
        
        // Check if session is still valid (within 24 hours)
        const age = Date.now() - sessionState.timestamp;
        if (age > 24 * 60 * 60 * 1000) {
            console.log('Session expired, starting fresh');
            localStorage.removeItem(SESSION_STORAGE_KEY);
            return false;
        }
        
        // Restore state (strokes will be restored from server)
        inkAmount = sessionState.inkAmount ?? INITIAL_INK;
        // Don't restore negative or zero timer values
        const savedTime = sessionState.remainingTime ?? ROUND_DURATION_SECONDS;
        remainingTime = savedTime > 0 ? savedTime : ROUND_DURATION_SECONDS;
        // If we had to reset the timer, also unlock the canvas
        isLocked = (savedTime > 0 && sessionState.isLocked) || false;
        zoomLevel = sessionState.zoomLevel ?? INITIAL_ZOOM;
        offsetX = sessionState.offsetX ?? 0;
        offsetY = sessionState.offsetY ?? 0;
        
        console.log('Session state restored:', {
            inkAmount,
            remainingTime,
            isLocked
        });
        
        return true;
    } catch (e) {
        console.error('Failed to restore session state:', e);
        return false;
    }
}

function clearSessionState() {
    localStorage.removeItem(SESSION_STORAGE_KEY);
    sessionStorage.removeItem('sessionId');
    console.log('Session state cleared');
}

// Content Detection System
const CONTENT_CHECK_INTERVAL = 3; // Check every 3 strokes
const ML_INPUT_SIZE = 128; // Size expected by ML model (adjust as needed)
const ML_DEBUG_VISUAL = true; // Enable visual debugging
let strokesSinceLastCheck = 0;
let detectionLogDiv = null; // For logging panel

// Canny Edge Detection
function cannyEdgeDetection(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    // Convert to grayscale
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    // Gaussian blur (simple 3x3)
    const blurred = new Uint8ClampedArray(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            blurred[idx] = Math.round(
                (gray[idx - width - 1] + 2 * gray[idx - width] + gray[idx - width + 1] +
                 2 * gray[idx - 1] + 4 * gray[idx] + 2 * gray[idx + 1] +
                 gray[idx + width - 1] + 2 * gray[idx + width] + gray[idx + width + 1]) / 16
            );
        }
    }
    
    // Sobel operator for gradients
    const gradX = new Float32Array(width * height);
    const gradY = new Float32Array(width * height);
    const magnitude = new Uint8ClampedArray(width * height);
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            
            gradX[idx] = (
                -blurred[idx - width - 1] + blurred[idx - width + 1] +
                -2 * blurred[idx - 1] + 2 * blurred[idx + 1] +
                -blurred[idx + width - 1] + blurred[idx + width + 1]
            );
            
            gradY[idx] = (
                -blurred[idx - width - 1] - 2 * blurred[idx - width] - blurred[idx - width + 1] +
                blurred[idx + width - 1] + 2 * blurred[idx + width] + blurred[idx + width + 1]
            );
            
            magnitude[idx] = Math.min(255, Math.sqrt(gradX[idx] ** 2 + gradY[idx] ** 2));
        }
    }
    
    // Threshold
    const threshold = 50;
    const edges = new Uint8ClampedArray(width * height);
    for (let i = 0; i < magnitude.length; i++) {
        edges[i] = magnitude[i] > threshold ? 255 : 0;
    }
    
    return edges;
}

// Find connected components (objects) in edge image
function findConnectedComponents(edges, width, height) {
    const visited = new Uint8ClampedArray(width * height);
    const components = [];
    
    function floodFill(startX, startY) {
        const pixels = [];
        const stack = [[startX, startY]];
        
        while (stack.length > 0) {
            const [x, y] = stack.pop();
            const idx = y * width + x;
            
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            if (visited[idx] || edges[idx] === 0) continue;
            
            visited[idx] = 1;
            pixels.push([x, y]);
            
            // 8-connectivity
            stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
            stack.push([x + 1, y + 1], [x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1]);
        }
        
        return pixels;
    }
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (edges[idx] > 0 && !visited[idx]) {
                const component = floodFill(x, y);
                if (component.length > 50) { // Minimum size filter
                    components.push(component);
                }
            }
        }
    }
    
    return components;
}

// Get bounding box for a component
function getBoundingBox(component) {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (const [x, y] of component) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }
    
    return { minX, maxX, minY, maxY };
}

// Extract and prepare object for ML
function extractObjectForML(bbox) {
    const { minX, maxX, minY, maxY } = bbox;
    // Add padding so the ML model sees the full object (not a too-tight crop)
    const rawWidth = maxX - minX + 1;
    const rawHeight = maxY - minY + 1;
    const PAD_PCT = 0.80; // pad by 12% of max dimension
    const pad = Math.round(Math.max(rawWidth, rawHeight) * PAD_PCT);

    // Expanded bbox with padding, clamped to canvas bounds
    const exMinX = Math.max(0, minX - pad);
    const exMinY = Math.max(0, minY - pad);
    const exMaxX = Math.min(canvas.width - 1, maxX + pad);
    const exMaxY = Math.min(canvas.height - 1, maxY + pad);

    const width = exMaxX - exMinX + 1;
    const height = exMaxY - exMinY + 1;
    
    // Create temporary canvas for extraction
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Extract the region from main canvas using the expanded bbox
    tempCtx.drawImage(canvas, exMinX, exMinY, width, height, 0, 0, width, height);
    
    // Resize to ML input size with padding to maintain aspect ratio
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = ML_INPUT_SIZE;
    resizeCanvas.height = ML_INPUT_SIZE;
    const resizeCtx = resizeCanvas.getContext('2d');
    
    // Fill with training data background color: #707170 (medium gray)
    resizeCtx.fillStyle = '#707170';
    resizeCtx.fillRect(0, 0, ML_INPUT_SIZE, ML_INPUT_SIZE);
    
    // Calculate scaling to fit while preserving aspect ratio
    const scale = Math.min(ML_INPUT_SIZE / width, ML_INPUT_SIZE / height);
    const scaledWidth = Math.round(width * scale);
    const scaledHeight = Math.round(height * scale);
    const offsetX = Math.round((ML_INPUT_SIZE - scaledWidth) / 2);
    const offsetY = Math.round((ML_INPUT_SIZE - scaledHeight) / 2);
    
    resizeCtx.drawImage(tempCanvas, 0, 0, width, height,
                        offsetX, offsetY, scaledWidth, scaledHeight);
    
    // Convert to grayscale and replace white background with #707170
    // Training data: #707170 background (112, 113, 112), white strokes (255)
    const imageData = resizeCtx.getImageData(0, 0, ML_INPUT_SIZE, ML_INPUT_SIZE);
    const data = imageData.data;
    const BACKGROUND_GRAY = 112; // #707170 in grayscale (approximately)
    
    for (let i = 0; i < data.length; i += 4) {
        // Convert RGB to grayscale using luminance formula
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        
        // Replace white/near-white background with training data gray background
        // If pixel is very light (>240), treat as background and use #707170 gray
        const finalGray = gray > 240 ? BACKGROUND_GRAY : gray;
        
        // Set RGB channels to grayscale value
        data[i] = finalGray;     // R
        data[i + 1] = finalGray; // G
        data[i + 2] = finalGray; // B
        // Alpha stays the same (data[i + 3])
    }
    
    resizeCtx.putImageData(imageData, 0, 0);
    
    return {
        canvas: resizeCanvas,
        imageData: imageData,
        // Return the expanded bbox used for extraction so callers/visualizers know the real crop
        bbox: { minX: exMinX, minY: exMinY, maxX: exMaxX, maxY: exMaxY }
    };
}

// Delete/white out object from canvas
function deleteObject(bbox) {
    const { minX, maxX, minY, maxY } = bbox;
    
    ctx.save();
    
    // Apply current transformations
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(-canvas.width / 2 + offsetX, -canvas.height / 2 + offsetY);
    
    // Fill the bounding box with white
    ctx.fillStyle = CANVAS_BACKGROUND;
    ctx.fillRect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    
    ctx.restore();
    
    console.log('Deleted object at:', bbox);
}

// Visual debugging helpers
function createDetectionLog() {
    if (detectionLogDiv) return;
    
    detectionLogDiv = document.createElement('div');
    detectionLogDiv.id = 'ml-detection-log';
    detectionLogDiv.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        width: 400px;
        max-height: 600px;
        background: rgba(0, 0, 0, 0.9);
        color: #0f0;
        font-family: monospace;
        font-size: 11px;
        padding: 15px;
        border-radius: 8px;
        overflow-y: auto;
        z-index: 10000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    `;
    
    const header = document.createElement('div');
    header.style.cssText = `
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
        color: #0ff;
        border-bottom: 1px solid #0ff;
        padding-bottom: 5px;
    `;
    header.textContent = '🤖 ML Detection Log';
    detectionLogDiv.appendChild(header);
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 10px;
        right: 10px;
        background: #f00;
        color: #fff;
        border: none;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        cursor: pointer;
        font-size: 18px;
        line-height: 1;
    `;
    closeBtn.onclick = () => detectionLogDiv.remove();
    detectionLogDiv.appendChild(closeBtn);
    
    document.body.appendChild(detectionLogDiv);
}

function logDetection(message, type = 'info', data = null) {
    if (!ML_DEBUG_VISUAL) return;
    
    createDetectionLog();
    
    const entry = document.createElement('div');
    const timestamp = new Date().toLocaleTimeString();
    
    const colors = {
        info: '#0f0',
        warn: '#ff0',
        error: '#f00',
        success: '#0ff'
    };
    
    entry.style.cssText = `
        margin-bottom: 8px;
        padding: 5px;
        border-left: 3px solid ${colors[type] || '#0f0'};
        padding-left: 8px;
        background: rgba(255,255,255,0.05);
    `;
    
    let content = `<span style="color: #888">[${timestamp}]</span> <span style="color: ${colors[type]}">${message}</span>`;
    
    if (data) {
        content += `<pre style="margin: 5px 0 0 0; color: #aaa; font-size: 10px; overflow-x: auto;">${JSON.stringify(data, null, 2)}</pre>`;
    }
    
    entry.innerHTML = content;
    detectionLogDiv.appendChild(entry);
    
    // Auto-scroll to bottom
    detectionLogDiv.scrollTop = detectionLogDiv.scrollHeight;
    
    // Keep only last 50 entries
    const entries = detectionLogDiv.querySelectorAll('div');
    if (entries.length > 52) { // 50 + header + close button
        entries[2].remove(); // Remove oldest (skip header)
    }
}

function visualizeDetection(bbox, extracted, result) {
    if (!ML_DEBUG_VISUAL) return;
    
    // Create a temporary overlay to show detected object
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 10px;
        left: 10px;
        background: rgba(0, 0, 0, 0.9);
        padding: 15px;
        border-radius: 8px;
        z-index: 10001;
        color: white;
        font-family: monospace;
        font-size: 12px;
        border: 2px solid ${result.is_inappropriate ? '#ff0000' : '#00ff00'};
    `;
    
    const title = document.createElement('div');
    title.style.cssText = `
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
        color: ${result.is_inappropriate ? '#ff0000' : '#00ff00'};
    `;
    title.textContent = result.is_inappropriate ? '⚠️ INAPPROPRIATE DETECTED' : '✓ Safe Content';
    overlay.appendChild(title);
    
    // Show the extracted image (now in QuickDraw grayscale format)
    const imgLabel = document.createElement('div');
    imgLabel.textContent = 'ML Input (grayscale inverted):';
    imgLabel.style.cssText = `
        font-size: 10px;
        color: #888;
        margin-bottom: 5px;
    `;
    overlay.appendChild(imgLabel);
    
    const img = document.createElement('img');
    img.src = extracted.canvas.toDataURL();
    img.style.cssText = `
        display: block;
        border: 2px solid ${result.is_inappropriate ? '#ff0000' : '#00ff00'};
        margin-bottom: 10px;
        image-rendering: pixelated;
    `;
    overlay.appendChild(img);
    
    // Show details
    const details = document.createElement('div');
    details.innerHTML = `
        <div>Confidence: <span style="color: #0ff">${(result.confidence * 100).toFixed(1)}%</span></div>
        <div>BBox: <span style="color: #0ff">${bbox.minX},${bbox.minY} → ${bbox.maxX},${bbox.maxY}</span></div>
        <div>Size: <span style="color: #0ff">${bbox.maxX - bbox.minX}×${bbox.maxY - bbox.minY}px</span></div>
        <div>Category: <span style="color: #0ff">${result.category}</span></div>
        ${result.mock ? '<div style="color: #ff0">⚠️ Mock Prediction</div>' : ''}
    `;
    overlay.appendChild(details);
    
    document.body.appendChild(overlay);
    
    // Auto-remove after 3 seconds
    setTimeout(() => overlay.remove(), 3000);
}

// Main content check function
async function checkForInappropriateContent() {
    const startTime = performance.now();
    logDetection('🔍 Starting content check...', 'info');
    console.log('Running content check...');
    
    // Show visual indicator that check is running
    const indicator = document.createElement('div');
    indicator.id = 'ml-checking-indicator';
    indicator.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: #0ff;
        padding: 20px 40px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 16px;
        z-index: 9999;
        border: 2px solid #0ff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    `;
    indicator.innerHTML = '🔍 Checking content...';
    document.body.appendChild(indicator);
    
    try {
        // Get current canvas image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Run Canny edge detection
        const edgeStartTime = performance.now();
        const edges = cannyEdgeDetection(imageData);
        const edgeTime = (performance.now() - edgeStartTime).toFixed(2);
        logDetection(`✓ Edge detection completed in ${edgeTime}ms`, 'success');
        
        // Find connected components (objects)
        const componentStartTime = performance.now();
        const components = findConnectedComponents(edges, canvas.width, canvas.height);
        const componentTime = (performance.now() - componentStartTime).toFixed(2);
        
        logDetection(`✓ Found ${components.length} objects in ${componentTime}ms`, 'success');
        console.log(`Found ${components.length} objects`);
        
        indicator.innerHTML = `🔍 Analyzing ${components.length} objects...`;
        
        // Process each component
        let processedCount = 0;
        let inappropriateCount = 0;
        
        for (const component of components) {
            processedCount++;
            indicator.innerHTML = `🔍 Checking object ${processedCount}/${components.length}...`;
            
            const bbox = getBoundingBox(component);
            const extracted = extractObjectForML(bbox);
            
            logDetection(`📦 Processing object ${processedCount}/${components.length}`, 'info', {
                bbox: `(${bbox.minX},${bbox.minY}) → (${bbox.maxX},${bbox.maxY})`,
                size: `${bbox.maxX - bbox.minX}×${bbox.maxY - bbox.minY}px`,
                pixels: component.length,
                format: 'grayscale inverted (QuickDraw format)'
            });
            
            // Send to ML model for classification
            const classifyStartTime = performance.now();
            const result = await classifyObjectWithResult(extracted);
            const classifyTime = (performance.now() - classifyStartTime).toFixed(2);
            
            if (result.is_inappropriate) {
                inappropriateCount++;
                logDetection(`⚠️ INAPPROPRIATE content detected!`, 'warn', {
                    confidence: `${(result.confidence * 100).toFixed(1)}%`,
                    time: `${classifyTime}ms`,
                    action: 'Removing object'
                });
                console.warn('Inappropriate content detected! Removing object...');
                deleteObject(bbox);
                
                // Notify server
                socket.emit('content.violation', {
                    sessionId: getOrCreateSessionId(),
                    bbox,
                    timestamp: Date.now(),
                    confidence: result.confidence
                });
            } else {
                logDetection(`✓ Object ${processedCount} classified as safe`, 'success', {
                    confidence: `${(result.confidence * 100).toFixed(1)}%`,
                    time: `${classifyTime}ms`
                });
            }
            
            // Visualize the detection result
            visualizeDetection(bbox, extracted, result);
        }
        
        const totalTime = (performance.now() - startTime).toFixed(2);
        const summary = {
            totalObjects: components.length,
            processed: processedCount,
            inappropriate: inappropriateCount,
            safe: processedCount - inappropriateCount,
            totalTime: `${totalTime}ms`
        };
        
        logDetection(`✅ Content check complete`, inappropriateCount > 0 ? 'warn' : 'success', summary);
        console.log('Content check complete:', summary);
    } finally {
        // Remove indicator
        indicator.remove();
    }
}

// ML Classification - Connects to local ML server
// Use same host as the web page to work with WSL/remote access
const ML_SERVER_HOST = window.location.hostname || 'localhost';
const ML_SERVER_URL = `http://${ML_SERVER_HOST}:5000/classify`;

console.log(`ML Server URL: ${ML_SERVER_URL} (page served from ${window.location.hostname})`);

async function classifyObjectWithResult(extracted) {
    logDetection('📡 Sending to ML server...', 'info', {
        url: ML_SERVER_URL,
        imageSize: `${ML_INPUT_SIZE}×${ML_INPUT_SIZE}`,
        bbox: extracted.bbox
    });
    
    console.log('Classifying object...', {
        size: `${ML_INPUT_SIZE}x${ML_INPUT_SIZE}`,
        bbox: extracted.bbox
    });
    
    try {
        // Convert canvas to blob
        const blob = await new Promise((resolve, reject) => {
            extracted.canvas.toBlob((b) => {
                if (b) {
                    resolve(b);
                } else {
                    reject(new Error('Failed to create blob from canvas'));
                }
            }, 'image/png');
        });
        
        const blobSize = (blob.size / 1024).toFixed(2);
        logDetection(`📦 Image blob created: ${blobSize}KB`, 'info');
        
        // Create form data
        const formData = new FormData();
        formData.append('image', blob, 'drawing.png');
        formData.append('sessionId', getOrCreateSessionId());
        formData.append('bbox', JSON.stringify(extracted.bbox));
        
        // Send to local ML server
        const fetchStartTime = performance.now();
        
        logDetection(`🌐 Fetching ${ML_SERVER_URL}...`, 'info');
        
        const response = await fetch(ML_SERVER_URL, {
            method: 'POST',
            body: formData,
            mode: 'cors'
        });
        const fetchTime = (performance.now() - fetchStartTime).toFixed(2);
        
        logDetection(`📨 Response status: ${response.status}`, 'info');
        
        if (!response.ok) {
            throw new Error(`ML server error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        logDetection(`✓ ML server response received (${fetchTime}ms)`, 'success', {
            isInappropriate: result.is_inappropriate,
            confidence: `${(result.confidence * 100).toFixed(1)}%`,
            category: result.category,
            mock: result.mock || false
        });
        
        console.log('ML Result:', result);
        
        // Notify socket server of the classification result
        socket.emit('ml.classification', {
            sessionId: getOrCreateSessionId(),
            bbox: extracted.bbox,
            isInappropriate: result.is_inappropriate,
            confidence: result.confidence,
            category: result.category,
            timestamp: Date.now()
        });
        
        return result;
        
    } catch (error) {
        const errorDetails = {
            message: error.message,
            name: error.name,
            stack: error.stack ? error.stack.split('\n')[0] : 'N/A'
        };
        
        // Check for specific error types
        if (error.message.includes('Failed to fetch')) {
            errorDetails.likelyCause = 'ML server not reachable or CORS issue';
            errorDetails.serverUrl = ML_SERVER_URL;
            errorDetails.suggestion = 'Check if ML server is running on port 5000';
        } else if (error.message.includes('NetworkError')) {
            errorDetails.likelyCause = 'Network connection failed';
        } else if (error.message.includes('blob')) {
            errorDetails.likelyCause = 'Canvas to blob conversion failed';
        }
        
        logDetection('❌ ML classification error', 'error', errorDetails);
        console.error('ML classification error:', error);
        console.error('Error details:', errorDetails);
        
        // On error, default to safe (not inappropriate)
        return {
            is_inappropriate: false,
            confidence: 0,
            category: 'error',
            error: true,
            error_message: error.message
        };
    }
}

// Legacy wrapper for backward compatibility
async function classifyObject(extracted) {
    const result = await classifyObjectWithResult(extracted);
    return result.is_inappropriate;
}

// Drawing state
let isDrawing = false;
let currentColor = '#3b82f6';
let brushSize = 8;
let strokes = [];
let currentStroke = null;
let inkAmount = INITIAL_INK;
let isLocked = false;

// Camera/Zoom state
let zoomLevel = INITIAL_ZOOM;
let offsetX = 0;
let offsetY = 0;
let isPanning = false;
let lastPanX = 0;
let lastPanY = 0;

// Socket setup
const socket = io({ transports: ['websocket'] });

socket.on('connect', () => {
    updateSocketStatus('connected');
    console.log('Socket connected');
});

socket.on('disconnect', () => {
    updateSocketStatus('disconnected');
    console.log('Socket disconnected');
});

socket.on('connect_error', () => {
    updateSocketStatus('connecting');
    console.log('Socket connection error');
});

function updateSocketStatus(status) {
    socketStatus.textContent = status;
    statusDot.className = 'status-dot ' + status;
}

// Initialize canvas
function resizeCanvas() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    
    // Only set internal resolution on first load
    if (!canvas.hasAttribute('data-initialized')) {
        canvas.width = 1024;
        canvas.height = 640;
        canvas.setAttribute('data-initialized', 'true');
        resetCanvas();
    }
}

function resetCanvas() {
    ctx.fillStyle = CANVAS_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    currentStroke = null;
    
    // Set random starting position when canvas is reset
    setRandomPosition();
    redrawCanvas();
}

function setRandomPosition() {
    // Random position within canvas bounds considering zoom
    const maxOffsetX = (canvas.width * zoomLevel - canvas.width) / 2;
    const maxOffsetY = (canvas.height * zoomLevel - canvas.height) / 2;
    
    offsetX = -maxOffsetX + Math.random() * maxOffsetX * 2;
    offsetY = -maxOffsetY + Math.random() * maxOffsetY * 2;
    
    // Clamp to valid range
    offsetX = Math.max(-maxOffsetX, Math.min(maxOffsetX, offsetX));
    offsetY = Math.max(-maxOffsetY, Math.min(maxOffsetY, offsetY));
}

function redrawCanvas() {
    ctx.save();
    
    // Clear with background
    ctx.fillStyle = CANVAS_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Apply zoom and pan transformations
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(-canvas.width / 2 + offsetX, -canvas.height / 2 + offsetY);
    
    // Redraw all strokes
    strokes.forEach(stroke => {
        if (stroke.points.length < 2) return;
        
        ctx.strokeStyle = stroke.color;
        ctx.lineWidth = brushSize / zoomLevel; // Adjust brush size for zoom
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
        
        for (let i = 1; i < stroke.points.length; i++) {
            ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
        ctx.stroke();
    });
    
    ctx.restore();
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);
window.addEventListener('orientationchange', () => {
    setTimeout(resizeCanvas, 100);
});

// Keyboard shortcuts for debugging
window.addEventListener('keydown', (e) => {
    // Press 'D' to toggle detection log
    if (e.key === 'd' || e.key === 'D') {
        if (detectionLogDiv && document.body.contains(detectionLogDiv)) {
            detectionLogDiv.remove();
            detectionLogDiv = null;
            console.log('ML Detection log hidden');
        } else {
            createDetectionLog();
            logDetection('🎮 Debug mode activated', 'success', {
                'Press D': 'Toggle this log',
                'Auto-check': `Every ${CONTENT_CHECK_INTERVAL} strokes`,
                'ML Server': ML_SERVER_URL,
                'Input Size': `${ML_INPUT_SIZE}×${ML_INPUT_SIZE}`
            });
            console.log('ML Detection log shown');
        }
    }
    
    // Press 'C' to manually trigger content check
    if (e.key === 'c' || e.key === 'C') {
        logDetection('🔍 Manual content check triggered', 'info');
        console.log('Manual content check triggered');
        checkForInappropriateContent().catch(err => {
            console.error('Manual content check failed:', err);
        });
    }
});

// Color picker
document.querySelectorAll('.color-dot').forEach(dot => {
    dot.addEventListener('click', () => {
        document.querySelectorAll('.color-dot').forEach(d => d.classList.remove('active'));
        dot.classList.add('active');
        currentColor = dot.dataset.color;
        document.documentElement.style.setProperty('--selected-color', currentColor);
    });
});

// Brush size
brushSizeInput.addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value);
});

// Drawing functions
function startStroke(x, y) {
    if (isLocked || inkAmount <= 0) return;
    
    // Transform coordinates based on zoom and pan
    const transformedPos = screenToCanvas(x, y);
    
    isDrawing = true;
    const startTime = performance.now();
    currentStroke = {
        points: [],
        startTime,
        color: currentColor
    };
    
    addPoint(transformedPos.x, transformedPos.y, 0);
}

function screenToCanvas(screenX, screenY) {
    // Convert screen coordinates to canvas coordinates accounting for zoom and pan
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const x = (screenX - centerX) / zoomLevel - offsetX + centerX;
    const y = (screenY - centerY) / zoomLevel - offsetY + centerY;
    
    return { x, y };
}

function addPoint(x, y, t) {
    if (!currentStroke) return;
    
    currentStroke.points.push({
        x: x,
        y: y,
        timestamp: t
    });
}

function drawStroke(x, y) {
    if (!isDrawing || !currentStroke) return;
    
    // Transform coordinates based on zoom and pan
    const transformedPos = screenToCanvas(x, y);
    
    const elapsed = performance.now() - currentStroke.startTime;
    const prevLength = currentStroke.points.length;
    addPoint(transformedPos.x, transformedPos.y, Math.round(elapsed));
    
    // Only draw the new segment incrementally, don't clear the canvas
    if (currentStroke.points.length > 1 && prevLength > 0) {
        ctx.save();
        
        // Apply zoom and pan transformations
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.scale(zoomLevel, zoomLevel);
        ctx.translate(-canvas.width / 2 + offsetX, -canvas.height / 2 + offsetY);
        
        ctx.strokeStyle = currentColor;
        ctx.lineWidth = brushSize / zoomLevel;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        
        const lastPoint = currentStroke.points[currentStroke.points.length - 2];
        const newPoint = currentStroke.points[currentStroke.points.length - 1];
        
        ctx.moveTo(lastPoint.x, lastPoint.y);
        ctx.lineTo(newPoint.x, newPoint.y);
        ctx.stroke();

        ctx.restore();
    }
    
    // Consume ink
    inkAmount = Math.max(0, inkAmount - INK_CONSUMPTION_RATE);
    updateInkMeter();
    
    if (inkAmount <= 0) {
        lockCanvas();
    }

}

function endStroke() {
    if (!isDrawing || !currentStroke) return;
    
    isDrawing = false;
    
    if (currentStroke.points.length > 1) {
        strokes.push(currentStroke);
        
        // Auto-send stroke after completion
        const quickDrawFormat = exportStrokeToQuickDraw(currentStroke);
        socket.emit('quickdraw.stroke', quickDrawFormat);
        console.log('Auto-sent stroke:', quickDrawFormat);
        
        // Save session state after each completed stroke
        saveSessionState();
        
        // Check for inappropriate content every N strokes
        strokesSinceLastCheck++;
        const strokesUntilCheck = CONTENT_CHECK_INTERVAL - strokesSinceLastCheck;
        
        if (ML_DEBUG_VISUAL) {
            logDetection(`✏️ Stroke completed (${strokesUntilCheck} until next check)`, 'info', {
                totalStrokes: strokes.length,
                strokesSinceLastCheck,
                nextCheckIn: strokesUntilCheck
            });
        }
        
        if (strokesSinceLastCheck >= CONTENT_CHECK_INTERVAL) {
            strokesSinceLastCheck = 0;
            checkForInappropriateContent().catch(err => {
                console.error('Content check failed:', err);
                logDetection('❌ Content check failed', 'error', { error: err.message });
            });
        }
    }
    currentStroke = null;
    
    // Final redraw to ensure everything is rendered
    redrawCanvas();
}

function getPointerPos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const clientX = event.clientX || (event.touches && event.touches[0]?.clientX);
    const clientY = event.clientY || (event.touches && event.touches[0]?.clientY);
    
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

// Event listeners
canvas.addEventListener('pointerdown', (e) => {
    // Don't start stroke if it's a touch with 2+ fingers
    if (e.pointerType === 'touch' && isTwoFingerGesture) return;
    
    const pos = getPointerPos(e);
    startStroke(pos.x, pos.y);
});

canvas.addEventListener('pointermove', (e) => {
    const pos = getPointerPos(e);
    drawStroke(pos.x, pos.y);
});

canvas.addEventListener('pointerup', (e) => {
    if (!isTwoFingerGesture) {
        endStroke();
    }
});

canvas.addEventListener('pointerleave', (e) => {
    if (!isTwoFingerGesture) {
        endStroke();
    }
});

// Touch events for better mobile support
canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        isTwoFingerGesture = true;
        
        // End any current stroke immediately
        if (isDrawing) {
            isDrawing = false;
            currentStroke = null;
            redrawCanvas();
        }
        
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        initialPinchDistance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );
        initialZoom = zoomLevel;
        isPanning = true;
    } else if (e.touches.length === 1 && !isTwoFingerGesture) {
        // Don't prevent default here - let pointer events handle it
        // Just set the flag
        isTwoFingerGesture = false;
    }
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && isPanning) {
        e.preventDefault();
        isTwoFingerGesture = true;
        
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const currentDistance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );
        
        const scale = currentDistance / initialPinchDistance;
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoom * scale));
        
        if (newZoom !== zoomLevel) {
            zoomLevel = newZoom;
            redrawCanvas();
        }
    } else if (e.touches.length === 1 && !isTwoFingerGesture) {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getPointerPos(e);
        drawStroke(pos.x, pos.y);
    }
}, { passive: false });

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    
    // Reset two-finger gesture flag when all fingers are lifted
    if (e.touches.length === 0) {
        isTwoFingerGesture = false;
        isPanning = false;
    }
    
    // Only end stroke if we were actually drawing (not zooming)
    if (isDrawing && !isTwoFingerGesture) {
        endStroke();
    }
});

// Zoom with mouse wheel
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoomLevel * delta));
    
    if (newZoom !== zoomLevel) {
        zoomLevel = newZoom;
        redrawCanvas();
    }
}, { passive: false });

// Pan with right-click or two-finger drag
canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
});

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 2) { // Right click
        isPanning = true;
        lastPanX = e.clientX;
        lastPanY = e.clientY;
        canvas.style.cursor = 'grab';
        e.preventDefault();
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (isPanning) {
        const dx = e.clientX - lastPanX;
        const dy = e.clientY - lastPanY;
        
        offsetX += dx / zoomLevel;
        offsetY += dy / zoomLevel;
        
        lastPanX = e.clientX;
        lastPanY = e.clientY;
        
        redrawCanvas();
        e.preventDefault();
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (e.button === 2) {
        isPanning = false;
        canvas.style.cursor = 'crosshair';
    }
});

// Touch pinch zoom
let initialPinchDistance = 0;
let initialZoom = 1;
let isTwoFingerGesture = false;

canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
        e.preventDefault();
        isTwoFingerGesture = true;
        
        // End any current stroke
        if (isDrawing) {
            endStroke();
        }
        
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        initialPinchDistance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );
        initialZoom = zoomLevel;
        isPanning = true;
    } else if (e.touches.length === 1) {
        isTwoFingerGesture = false;
        e.preventDefault();
        const pos = getPointerPos(e);
        startStroke(pos.x, pos.y);
    }
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && isPanning) {
        e.preventDefault();
        isTwoFingerGesture = true;
        
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const currentDistance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );
        
        const scale = currentDistance / initialPinchDistance;
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoom * scale));
        
        if (newZoom !== zoomLevel) {
            zoomLevel = newZoom;
            redrawCanvas();
        }
    } else if (e.touches.length === 1 && !isTwoFingerGesture) {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getPointerPos(e);
        drawStroke(pos.x, pos.y);
    }
}, { passive: false });

// Socket transmission functions
function exportStrokeToQuickDraw(stroke) {
    const xs = [];
    const ys = [];
    const ts = [];
    
    stroke.points.forEach(point => {
        // Scale coordinates to QuickDraw format (0-255)
        const scaledX = Math.round((point.x / canvas.width) * QT_SCALE);
        const scaledY = Math.round((point.y / canvas.height) * QT_SCALE);
        
        xs.push(Math.max(0, Math.min(QT_SCALE, scaledX)));
        ys.push(Math.max(0, Math.min(QT_SCALE, scaledY)));
        ts.push(point.timestamp);
    });
    
    // Return extended format with color and width metadata
    return {
        points: [xs, ys, ts],
        color: stroke.color,
        width: brushSize
    };
}

function sendLastStroke() {
    if (strokes.length === 0) {
        console.log('No strokes to send');
        return;
    }
    
    const lastStroke = strokes[strokes.length - 1];
    const quickDrawFormat = exportStrokeToQuickDraw(lastStroke);
    
    socket.emit('quickdraw.stroke', quickDrawFormat);
    console.log('Sent last stroke:', quickDrawFormat);
}

function sendBatch() {
    if (strokes.length === 0) {
        console.log('No strokes to send');
        return;
    }
    
    const batch = strokes.map(stroke => exportStrokeToQuickDraw(stroke));
    socket.emit('quickdraw.batch', batch);
    console.log('Sent batch of', batch.length, 'strokes');
}

function clearCanvas() {
    resetCanvas();
    inkAmount = INITIAL_INK;
    updateInkMeter();
    isLocked = false;
    remainingTime = ROUND_DURATION_SECONDS;
    if (timerDisabled) {
        timerDisplay.textContent = '∞';
    } else {
        startTimer(false);
    }
    clearSessionState();
    socket.emit('quickdraw.clear');
    console.log('Canvas cleared and session reset');
}

// Button event listeners
clearBtn.addEventListener('click', clearCanvas);
sendStrokeBtn.addEventListener('click', sendLastStroke);
sendBatchBtn.addEventListener('click', sendBatch);

// Ink meter
function updateInkMeter() {
    const percentage = (inkAmount / INITIAL_INK) * 100;
    inkFill.style.width = Math.max(0, percentage) + '%';
}

function lockCanvas() {
    isLocked = true;
    canvas.style.cursor = 'not-allowed';
    console.log('Canvas locked - ink depleted');
}

// Timer
let remainingTime = ROUND_DURATION_SECONDS;
let roundTimerId = null;
let timerDisabled = debugDisableTimerToggle?.checked ?? false;

function updateTimer() {
    const minutes = String(Math.floor(remainingTime / 60)).padStart(2, '0');
    const seconds = String(remainingTime % 60).padStart(2, '0');
    timerDisplay.textContent = `${minutes}:${seconds}`;
}

function stopTimer() {
    if (roundTimerId !== null) {
        clearInterval(roundTimerId);
        roundTimerId = null;
    }
}

function startTimer(resetRemaining = true) {
    stopTimer();

    if (resetRemaining) {
        remainingTime = ROUND_DURATION_SECONDS;
    }

    if (timerDisabled) {
        timerDisplay.textContent = '∞';
        return;
    }

    updateTimer();
    roundTimerId = setInterval(() => {
        remainingTime -= 1;
        updateTimer();

        if (remainingTime <= 0) {
            stopTimer();
            lockCanvas();
            handleRoundEnd();
        }
    }, 1000);
}

function handleRoundEnd() {
    console.log('Round ended');
    
    // Send final drawing
    if (strokes.length > 0) {
        const batch = strokes.map(stroke => exportStrokeToQuickDraw(stroke));
        socket.emit('quickdraw.drawing', batch);
        console.log('Sent final drawing with', batch.length, 'strokes');
    }
}

debugDisableTimerToggle?.addEventListener('change', (event) => {
    timerDisabled = event.target.checked;
    if (timerDisabled) {
        stopTimer();
        timerDisplay.textContent = '∞';
    } else {
        startTimer();
    }
});

// Initialize
const sessionId = getOrCreateSessionId();
console.log('Session ID:', sessionId);

// Try to restore previous session state
const sessionRestored = restoreSessionState();

if (sessionRestored) {
    // Update UI with restored state (strokes will come from server)
    updateInkMeter();
    updateTimer();
    if (isLocked) {
        canvas.style.cursor = 'not-allowed';
    }
    // Start timer without resetting if session was restored with valid time
    if (!timerDisabled && remainingTime > 0) {
        startTimer(false); // Don't reset - use restored time
    }
} else {
    updateInkMeter();
    // Start fresh timer for new session
    if (!timerDisabled) {
        startTimer(true); // Reset to full duration
    }
}

if (timerDisabled) {
    timerDisplay.textContent = '∞';
}
updateSocketStatus('connecting');

// Save session state periodically (every 5 seconds)
setInterval(saveSessionState, 5000);

// Save session state before page unload
window.addEventListener('beforeunload', () => {
    saveSessionState();
});

// Hide control buttons if not in debug mode
if (!DEBUG_MODE) {
    const controlButtons = document.querySelector('.control-buttons');
    if (controlButtons) {
        controlButtons.style.display = 'none';
    }
}
