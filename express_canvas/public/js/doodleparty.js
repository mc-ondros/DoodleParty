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

// Constants
const CANVAS_BACKGROUND = '#ffffff';
const ROUND_DURATION_SECONDS = 90;
const INITIAL_INK = 200;
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
        remainingTime = sessionState.remainingTime ?? ROUND_DURATION_SECONDS;
        isLocked = sessionState.isLocked || false;
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
let strokesSinceLastCheck = 0;

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
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    
    // Create temporary canvas for extraction
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Extract the region from main canvas
    tempCtx.drawImage(canvas, minX, minY, width, height, 0, 0, width, height);
    
    // Resize to ML input size with padding to maintain aspect ratio
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = ML_INPUT_SIZE;
    resizeCanvas.height = ML_INPUT_SIZE;
    const resizeCtx = resizeCanvas.getContext('2d');
    
    // Fill with white background
    resizeCtx.fillStyle = '#ffffff';
    resizeCtx.fillRect(0, 0, ML_INPUT_SIZE, ML_INPUT_SIZE);
    
    // Calculate scaling to fit
    const scale = Math.min(ML_INPUT_SIZE / width, ML_INPUT_SIZE / height);
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;
    const offsetX = (ML_INPUT_SIZE - scaledWidth) / 2;
    const offsetY = (ML_INPUT_SIZE - scaledHeight) / 2;
    
    resizeCtx.drawImage(tempCanvas, 0, 0, width, height, 
                        offsetX, offsetY, scaledWidth, scaledHeight);
    
    return {
        canvas: resizeCanvas,
        imageData: resizeCtx.getImageData(0, 0, ML_INPUT_SIZE, ML_INPUT_SIZE),
        bbox
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

// Main content check function
async function checkForInappropriateContent() {
    console.log('Running content check...');
    
    // Get current canvas image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Run Canny edge detection
    const edges = cannyEdgeDetection(imageData);
    
    // Find connected components (objects)
    const components = findConnectedComponents(edges, canvas.width, canvas.height);
    
    console.log(`Found ${components.length} objects`);
    
    // Process each component
    for (const component of components) {
        const bbox = getBoundingBox(component);
        const extracted = extractObjectForML(bbox);
        
        // Send to ML model for classification
        const isInappropriate = await classifyObject(extracted);
        
        if (isInappropriate) {
            console.warn('Inappropriate content detected! Removing object...');
            deleteObject(bbox);
            
            // Notify server
            socket.emit('content.violation', {
                sessionId: getOrCreateSessionId(),
                bbox,
                timestamp: Date.now()
            });
        }
    }
}

// ML Classification - Connects to local ML server
const ML_SERVER_URL = 'http://localhost:5000/classify';

async function classifyObject(extracted) {
    console.log('Classifying object...', {
        size: `${ML_INPUT_SIZE}x${ML_INPUT_SIZE}`,
        bbox: extracted.bbox
    });
    
    try {
        // Convert canvas to blob
        const blob = await new Promise((resolve) => {
            extracted.canvas.toBlob(resolve, 'image/png');
        });
        
        // Create form data
        const formData = new FormData();
        formData.append('image', blob, 'drawing.png');
        formData.append('sessionId', getOrCreateSessionId());
        formData.append('bbox', JSON.stringify(extracted.bbox));
        
        // Send to local ML server
        const response = await fetch(ML_SERVER_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`ML server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('ML Result:', result);
        
        // Notify socket server of the classification result
        socket.emit('ml.classification', {
            sessionId: getOrCreateSessionId(),
            bbox: extracted.bbox,
            isInappropriate: result.is_inappropriate,
            confidence: result.confidence,
            timestamp: Date.now()
        });
        
        return result.is_inappropriate;
        
    } catch (error) {
        console.error('ML classification error:', error);
        // On error, default to safe (not inappropriate)
        return false;
    }
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
        if (strokesSinceLastCheck >= CONTENT_CHECK_INTERVAL) {
            strokesSinceLastCheck = 0;
            checkForInappropriateContent().catch(err => {
                console.error('Content check failed:', err);
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
    updateTimer();
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

function updateTimer() {
    const minutes = String(Math.floor(remainingTime / 60)).padStart(2, '0');
    const seconds = String(remainingTime % 60).padStart(2, '0');
    timerDisplay.textContent = `${minutes}:${seconds}`;
}

function startTimer() {
    updateTimer();
    
    const timerId = setInterval(() => {
        remainingTime -= 1;
        updateTimer();
        
        if (remainingTime <= 0) {
            clearInterval(timerId);
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
} else {
    updateInkMeter();
}

startTimer();
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
