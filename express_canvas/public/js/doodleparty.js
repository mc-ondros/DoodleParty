// DoodleParty Socket Sender - Combines user/index.html drawing interface with socket.io

console.log('🎨 [STARTUP] doodleparty.js script loading...');
console.log('🎨 [STARTUP] Document ready state:', document.readyState);
console.log('🎨 [STARTUP] Window.io available:', typeof io !== 'undefined');

const canvas = document.getElementById('drawingCanvas');
console.log('🎨 [STARTUP] Canvas element found:', canvas !== null);
const ctx = canvas.getContext('2d');
const brushSizeInput = document.getElementById('brushSize');
const clearBtn = document.getElementById('clearBtn');
const sendStrokeBtn = document.getElementById('sendStrokeBtn');
const sendBatchBtn = document.getElementById('sendBatchBtn');
const socketStatus = document.getElementById('socketStatus');
const statusDot = document.getElementById('statusDot');
const inkFill = document.getElementById('inkFill');
const timerCircle = document.getElementById('timerCircle');
const timerCircleText = document.getElementById('timerCircleText');
const inkCircle = document.getElementById('inkCircle');
const inkCircleText = document.getElementById('inkCircleText');

// Diagnostic: verify critical elements exist
console.log('[init] DOM elements check:');
console.log('  canvas:', !!canvas);
console.log('  socketStatus:', !!socketStatus, socketStatus);
console.log('  statusDot:', !!statusDot, statusDot);

// ML Canvas - Black background, white strokes, fixed 8px width
const mlCanvas = document.getElementById('mlCanvas');
const mlCtx = mlCanvas.getContext('2d');
const ML_STROKE_WIDTH = 8;
const ML_STROKE_COLOR = '#ffffff';
const ML_BACKGROUND_COLOR = '#707070';

// Constants
const CANVAS_BACKGROUND = '#ffffff';
// Local defaults (will be overridden by admin config once received)
const ROUND_DURATION_SECONDS = 90; // fallback only (server sends authoritative timer)
const INITIAL_INK = 100; // base ink capacity for Medium
const INK_CONSUMPTION_RATE = 1;
const WORLD_WIDTH = 1920;
const WORLD_HEIGHT = 1080;
const QT_SCALE = Math.max(WORLD_WIDTH, WORLD_HEIGHT) - 1; // QuickDraw coordinate scale
const DEBUG_MODE = true; // Set to true to show manual send buttons
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4;
const INITIAL_ZOOM = 4; // Start closer to the canvas for detail work
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
        
        // If prior session ended (timer depleted or locked), start fresh
        const restoredTime = sessionState.remainingTime ?? ROUND_DURATION_SECONDS;
        const restoredLocked = sessionState.isLocked || false;

        if (restoredTime <= 0 || restoredLocked) {
            console.log('Previous session ended, starting fresh');
            clearSessionState();
            return false;
        }

        // Restore state (strokes will be restored from server)
        inkAmount = sessionState.inkAmount ?? INITIAL_INK;
        remainingTime = restoredTime;
        isLocked = restoredLocked;
        zoomLevel = sessionState.zoomLevel ?? INITIAL_ZOOM;
        offsetX = sessionState.offsetX ?? 0;
        offsetY = sessionState.offsetY ?? 0;
        
        console.log('Session state restored:', {
            inkAmount,
            remainingTime,
            isLocked
        });

        redrawCanvas();
        
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

// Drawing state
let isDrawing = false;
let currentColor = '#3b82f6';
let brushSize = 8;
let strokes = [];
let currentStroke = null;
let inkCapacity = INITIAL_INK; // dynamic capacity based on admin Ink Limit
let inkAmount = INITIAL_INK;
let isLocked = false;
let myStrokesSinceLastDetection = 0; // Only count this user's strokes
const AUTO_DETECT_INTERVAL = 3; // Run detection every N strokes from this user
let isKicked = false; // becomes true after admin removes this player

// Timer state (server-driven)
let timerState = 'paused'; // 'running' | 'paused' | 'expired'
let timerDuration = 300; // total duration in seconds
let remainingTime = 300; // seconds left

// DOM references (needed early for event handlers)
const lockOverlay = document.getElementById('lockOverlay');
const promptOverlay = document.getElementById('promptOverlay');

// Camera/Zoom state
let zoomLevel = INITIAL_ZOOM;
let offsetX = 0;
let offsetY = 0;
let isPanning = false;
let lastPanX = 0;
let lastPanY = 0;

// Socket setup (allow fallback to polling in environments where WebSocket upgrade fails)
// Check if io is available (loaded from global script)
if (typeof io === 'undefined') {
    console.error('Socket.IO not loaded! Check that /socket.io/socket.io.js is loaded before doodleparty.js');
}

console.log('🔌 [INIT] Initializing Socket.IO connection to /canvas namespace...');
console.log('🔌 [INIT] Socket.IO library available:', typeof io !== 'undefined');
console.log('🔌 [INIT] Current URL:', window.location.href);

const socket = io('/canvas', {
    transports: ['websocket', 'polling'],
    timeout: 8000,
    reconnectionAttempts: 10,
});

console.log('🔌 [INIT] Socket instance created:', socket);
console.log('🔌 [INIT] Socket connecting state:', socket.connected);

// Enhanced connection diagnostics
socket.io.on('reconnect_attempt', (attempt) => {
    console.log(`🔄 [RECONNECT] Attempt ${attempt}...`);
});

socket.io.on('reconnect', (attempt) => {
    console.log(`✅ [RECONNECT] Reconnected after ${attempt} attempts`);
});

socket.io.on('reconnect_error', (err) => {
    console.error('❌ [RECONNECT] Reconnection error:', err);
});

socket.io.on('reconnect_failed', () => {
    console.error('❌ [RECONNECT] All reconnection attempts failed');
});

socket.io.on('ping', () => {
    console.log('🏓 [PING] Ping sent to server');
});

socket.io.on('open', () => {
    console.log('🚪 [ENGINE.IO] Connection opened');
});

socket.io.on('close', (reason) => {
    console.log('🚪 [ENGINE.IO] Connection closed:', reason);
});

socket.on('reconnect_attempt', (attempt) => {
    if (isKicked) {
        console.warn('🚫 [SOCKET] Reconnect blocked (kicked)');
        updateSocketStatus('kicked');
        return;
    }
    console.warn(`🔄 [SOCKET] Reconnect attempt ${attempt}`);
    updateSocketStatus('connecting');
});

socket.on('reconnect_failed', () => {
    console.error('❌ [SOCKET] Reconnect failed');
    updateSocketStatus('error');
});

socket.on('error', (err) => {
    console.error('❌ [SOCKET] Socket error:', err);
    updateSocketStatus('error');
});

socket.io.on('error', (err) => {
    console.error('❌ [ENGINE.IO] Engine error:', err);
});

socket.on('connect_error', (err) => {
    console.error('❌ [CONNECT] Connection error:', err);
    console.error('❌ [CONNECT] Error message:', err.message);
    console.error('❌ [CONNECT] Error type:', err.type);
    console.error('❌ [CONNECT] Error description:', err.description);
    updateSocketStatus('error');
});

socket.on('connect', () => {
    console.log('✅ [CONNECT] CONNECTED successfully!');
    console.log('✅ [CONNECT] Socket ID:', socket.id);
    console.log('✅ [CONNECT] Namespace:', socket.nsp);
    console.log('✅ [CONNECT] Transport:', socket.io.engine.transport.name);
    console.log('✅ [CONNECT] Connected:', socket.connected);
    updateSocketStatus('connected');
    // Request sync immediately on connection to ensure we get existing strokes
    setTimeout(() => {
        socket.emit('quickdraw.requestSync');
        console.log('Requested stroke sync from server');
    }, 100);
});

socket.on('disconnect', (reason) => {
    console.log('🔌 [DISCONNECT] Socket disconnected');
    console.log('🔌 [DISCONNECT] Reason:', reason);
    console.log('🔌 [DISCONNECT] Will reconnect:', socket.io.reconnection());
    updateSocketStatus('disconnected');
});

// Kicked by admin
socket.on('kicked', (payload) => {
    console.warn('[socket] kicked by admin', payload);
    isKicked = true;
    // Prevent further reconnections
    if (socket.io && socket.io.opts) socket.io.opts.reconnection = false;
    // Lock canvas and show overlay message
    lockCanvas('kicked');
    updateSocketStatus('kicked');
    try { socket.disconnect(); } catch (_) {}
});

socket.on('connect_error', () => {
    updateSocketStatus('connecting');
    console.log('Socket connection error');
});

socket.on('quickdraw.sync', (syncedStrokes) => {
    console.log('Received sync with', syncedStrokes.length, 'strokes');
    if (!Array.isArray(syncedStrokes)) {
        console.log('Invalid sync data - not an array');
        return;
    }
    
    // Only clear if we're actually receiving strokes
    if (syncedStrokes.length > 0) {
        strokes = [];
        clearMLCanvas(); // Clear ML canvas when syncing
        
        // Import synced strokes into local canvas
        let imported = 0;
        syncedStrokes.forEach(strokeData => {
            const stroke = importStrokeFromQuickDraw(strokeData);
            if (stroke && stroke.points.length > 1) {
                strokes.push(stroke);
                // Redraw stroke on ML canvas
                redrawStrokeOnMLCanvas(stroke);
                imported++;
            } else {
                console.log('Failed to import stroke:', strokeData);
            }
        });
        
        console.log('Successfully imported', imported, 'of', syncedStrokes.length, 'strokes');
        
        // Force a complete redraw
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        redrawCanvas();
    } else {
        console.log('No strokes to sync');
    }
});

socket.on('quickdraw.stroke', (strokeData) => {
    console.log('Received real-time stroke from another client:', strokeData);
    
    // Visual feedback - flash canvas border
    canvas.style.border = '10px solid lime';
    setTimeout(() => { canvas.style.border = 'none'; }, 500);
    
    const stroke = importStrokeFromQuickDraw(strokeData);
    console.log('Imported stroke:', stroke);
    if (stroke && stroke.points.length > 1) {
        strokes.push(stroke);
        console.log('Added stroke, total strokes:', strokes.length);
        console.log('Calling redrawCanvas...');
        redrawCanvas();
        console.log('Canvas redrawn');
    } else {
        console.warn('Failed to import stroke or not enough points:', stroke);
    }
});

socket.on('quickdraw.clear', () => {
    console.log('Received clear event from server');
    // Clear all drawing state completely
    strokes = [];
    currentStroke = null;
    isDrawing = false;

    // Reset canvas state (ink resets to current capacity determined by admin config)
    inkAmount = inkCapacity || INITIAL_INK;
    updateInkMeter();

    // Unlock if session is open and timer not expired
    if (timerState !== 'expired' && lockOverlay?.style.display === 'flex') {
        unlockCanvas();
    }

    // Clear session storage (local persistence only)
    clearSessionState();

    // Force complete canvas clear - multiple methods for mobile compatibility
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform to identity
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = CANVAS_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    // Redraw with clean state
    redrawCanvas();

    console.log('Canvas fully reset from clear event (authoritative server timer retained)');
});

socket.on('quickdraw.eraseRegion', (payload) => {
    console.log('🗑️ Received erase region event:', payload);
    // Apply the erasure without re-broadcasting
    const { x1, y1, x2, y2 } = payload;
    
    // Remove strokes that intersect with the bounding box
    const initialCount = strokes.length;
    strokes = strokes.filter(stroke => {
        const intersects = stroke.points.some(point => {
            return point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2;
        });
        return !intersects;
    });
    
    console.log(`  Removed ${initialCount - strokes.length} stroke(s)`);
    
    // Clear and redraw ML canvas
    mlCtx.fillStyle = ML_BACKGROUND_COLOR;
    mlCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    
    // Redraw remaining strokes
    redrawCanvas();
    
    // Redraw ML canvas strokes in affected region
    strokes.forEach(stroke => {
        if (stroke.points.length > 1) {
            const hasPointsInRegion = stroke.points.some(point => {
                return point.x >= x1 - 50 && point.x <= x2 + 50 && 
                       point.y >= y1 - 50 && point.y <= y2 + 50;
            });
            
            if (hasPointsInRegion) {
                mlCtx.strokeStyle = ML_STROKE_COLOR;
                mlCtx.lineWidth = ML_STROKE_WIDTH;
                mlCtx.lineCap = 'round';
                mlCtx.lineJoin = 'round';
                mlCtx.beginPath();
                mlCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
                
                for (let i = 1; i < stroke.points.length; i++) {
                    mlCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
                mlCtx.stroke();
            }
        }
    });
});

socket.on('ml.detectionResults', (results) => {
    console.log('🤖 ML Detection Results:', results);
    
    if (results.success && results.results) {
        const summary = results.summary || {};
        console.log(`📊 Summary: ${summary.total} objects detected`);
        console.log(`  🔴 Positive: ${summary.positive}`);
        console.log(`  🟢 Negative: ${summary.negative}`);
        
        // Log individual results
        results.results.forEach((result, idx) => {
            const icon = result.class === 'positive' ? '🔴' : '🟢';
            console.log(`  ${icon} Object ${idx}: ${result.class} (${(result.confidence * 100).toFixed(1)}%)`);
        });
        
        if (results.inputVisualization) {
            console.log(`📥 Input visualization: ${results.inputVisualization}`);
        }
        if (results.resultsVisualization) {
            console.log(`📊 Results visualization: ${results.resultsVisualization}`);
        }
        
        // Handle inappropriate content detection
        if (summary.positive > 0) {
            console.warn(`⚠️ WARNING: ${summary.positive} inappropriate object(s) detected!`);
            
            // Get objects data from results
            const objectsData = results.objectsData || [];
            
            // Remove inappropriate objects
            removeInappropriateObjects(results.results, objectsData);
        }
    } else {
        console.error('❌ ML detection failed:', results.error);
    }
});

function removeInappropriateObjects(detectionResults, objectsData) {
    if (!detectionResults || detectionResults.length === 0) return;
    
    let removedCount = 0;
    
    // Process each detection result
    detectionResults.forEach((result, idx) => {
        if (result.class === 'positive' && objectsData[idx]) {
            const bbox = objectsData[idx].boundingBox;
            if (bbox) {
                // Erase the region on both canvases (without redrawing yet)
                eraseRegion(bbox.x1, bbox.y1, bbox.x2, bbox.y2, false);
                removedCount++;
                console.log(`🗑️ Removed inappropriate object at (${bbox.x1}, ${bbox.y1}) - (${bbox.x2}, ${bbox.y2})`);
            }
        }
    });
    
    if (removedCount > 0) {
        console.log(`✓ Cleaned ${removedCount} inappropriate object(s) from canvas`);
        // Force redraw once after all removals
        redrawCanvas();
    }
}

function eraseRegion(x1, y1, x2, y2, shouldRedraw = true) {
    // Add padding to ensure complete removal
    const padding = 10;
    x1 = Math.max(0, x1 - padding);
    y1 = Math.max(0, y1 - padding);
    x2 = Math.min(WORLD_WIDTH, x2 + padding);
    y2 = Math.min(WORLD_HEIGHT, y2 + padding);
    
    // Remove strokes that intersect with the bounding box
    const initialStrokeCount = strokes.length;
    strokes = strokes.filter(stroke => {
        // Check if any point in the stroke is within the bounding box
        const intersects = stroke.points.some(point => {
            return point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2;
        });
        return !intersects; // Keep strokes that don't intersect
    });
    
    const removedStrokes = initialStrokeCount - strokes.length;
    if (removedStrokes > 0) {
        console.log(`  Removed ${removedStrokes} stroke(s) intersecting with region`);
    }
    
    // Redraw main canvas to reflect removed strokes (if requested)
    if (shouldRedraw) {
        redrawCanvas();
    }
    
    // Clear the region on ML canvas
    mlCtx.fillStyle = ML_BACKGROUND_COLOR;
    mlCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    
    // Redraw remaining strokes on ML canvas in the affected region
    strokes.forEach(stroke => {
        const hasPointsInRegion = stroke.points.some(point => {
            return point.x >= x1 - 50 && point.x <= x2 + 50 && 
                   point.y >= y1 - 50 && point.y <= y2 + 50;
        });
        
        if (hasPointsInRegion && stroke.points.length > 1) {
            mlCtx.strokeStyle = ML_STROKE_COLOR;
            mlCtx.lineWidth = ML_STROKE_WIDTH;
            mlCtx.lineCap = 'round';
            mlCtx.lineJoin = 'round';
            mlCtx.beginPath();
            mlCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
            
            for (let i = 1; i < stroke.points.length; i++) {
                mlCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
            }
            mlCtx.stroke();
        }
    });
    
    // Broadcast removal to other clients
    socket.emit('quickdraw.eraseRegion', { x1, y1, x2, y2 });
}

function updateSocketStatus(status) {
    console.log('[updateSocketStatus] called with:', status);
    console.log('[updateSocketStatus] socketStatus element:', socketStatus);
    console.log('[updateSocketStatus] statusDot element:', statusDot);
    if (socketStatus) {
        socketStatus.textContent = status;
        console.log('[updateSocketStatus] Set text to:', status);
    } else {
        console.error('[updateSocketStatus] socketStatus element is null!');
    }
    if (statusDot) {
        statusDot.className = 'status-dot ' + status;
        console.log('[updateSocketStatus] Set className to:', 'status-dot ' + status);
    } else {
        console.error('[updateSocketStatus] statusDot element is null!');
    }
}

// Initialize canvas
function resizeCanvas() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    
    // Only set internal resolution on first load
    if (!canvas.hasAttribute('data-initialized')) {
        canvas.width = WORLD_WIDTH;
        canvas.height = WORLD_HEIGHT;
        canvas.setAttribute('data-initialized', 'true');
        // Don't reset here - let initialization handle viewport
    }

    redrawCanvas();
}

function resetCanvas(options = {}) {
    const { randomizeViewport = true } = options;
    ctx.fillStyle = CANVAS_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    currentStroke = null;
    
    // Clear ML canvas
    clearMLCanvas();
    
    if (randomizeViewport) {
        initializeRandomViewport();
    }

    redrawCanvas();
}

function initializeRandomViewport() {
    zoomLevel = INITIAL_ZOOM;

    const spawn = getSpawnPointAwayFromCenter();
    centerViewportOn(spawn.x, spawn.y);
    canvas.style.cursor = 'crosshair';
}

function getSpawnPointAwayFromCenter() {
    const margin = WORLD_WIDTH * 0.08;
    const radiusMin = (WORLD_WIDTH / 2) * 0.55;
    const radiusMax = (WORLD_WIDTH / 2) - margin;
    const angle = Math.random() * Math.PI * 2;
    const bias = Math.pow(Math.random(), 0.4); // Bias radius toward the outer ring
    const radius = radiusMin + (radiusMax - radiusMin) * bias;

    const spawnX = WORLD_WIDTH / 2 + Math.cos(angle) * radius;
    const spawnY = WORLD_HEIGHT / 2 + Math.sin(angle) * radius;

    return {
        x: Math.max(margin, Math.min(WORLD_WIDTH - margin, spawnX)),
        y: Math.max(margin, Math.min(WORLD_HEIGHT - margin, spawnY))
    };
}

function centerViewportOn(x, y) {
    const clampedX = Math.max(0, Math.min(WORLD_WIDTH, x));
    const clampedY = Math.max(0, Math.min(WORLD_HEIGHT, y));

    offsetX = canvas.width / 2 - clampedX;
    offsetY = canvas.height / 2 - clampedY;
    clampViewportToWorld();
}

function getMinZoom() {
    // Calculate minimum zoom so viewport can never be larger than world bounds
    // This prevents seeing outside the 1920x1080 area
    const minZoomX = canvas.width / WORLD_WIDTH;
    const minZoomY = canvas.height / WORLD_HEIGHT;
    return Math.max(minZoomX, minZoomY, MIN_ZOOM);
}

function clampViewportToWorld() {
    // Enforce minimum zoom to prevent seeing outside world
    const calculatedMinZoom = getMinZoom();
    if (zoomLevel < calculatedMinZoom) {
        zoomLevel = calculatedMinZoom;
    }
    
    // Calculate visible area in world coordinates
    const viewWidth = canvas.width / zoomLevel;
    const viewHeight = canvas.height / zoomLevel;

    // Get current center position
    let centerX = canvas.width / 2 - offsetX;
    let centerY = canvas.height / 2 - offsetY;

    // Clamp center so viewport edges never go outside world bounds
    const halfViewWidth = viewWidth / 2;
    const halfViewHeight = viewHeight / 2;

    // Don't allow panning beyond world edges
    centerX = Math.max(halfViewWidth, Math.min(WORLD_WIDTH - halfViewWidth, centerX));
    centerY = Math.max(halfViewHeight, Math.min(WORLD_HEIGHT - halfViewHeight, centerY));

    // Convert back to offset
    offsetX = canvas.width / 2 - centerX;
    offsetY = canvas.height / 2 - centerY;
}

// ML Canvas Functions
function initializeMLCanvas() {
    mlCtx.fillStyle = ML_BACKGROUND_COLOR;
    mlCtx.fillRect(0, 0, mlCanvas.width, mlCanvas.height);
    console.log('ML Canvas initialized:', mlCanvas.width, 'x', mlCanvas.height);
}

function clearMLCanvas() {
    mlCtx.fillStyle = ML_BACKGROUND_COLOR;
    mlCtx.fillRect(0, 0, mlCanvas.width, mlCanvas.height);
}

function drawToMLCanvas(lastPoint, newPoint) {
    // Draw stroke segment on ML canvas with fixed width and white color
    mlCtx.strokeStyle = ML_STROKE_COLOR;
    mlCtx.lineWidth = ML_STROKE_WIDTH;
    mlCtx.lineCap = 'round';
    mlCtx.lineJoin = 'round';
    mlCtx.beginPath();
    mlCtx.moveTo(lastPoint.x, lastPoint.y);
    mlCtx.lineTo(newPoint.x, newPoint.y);
    mlCtx.stroke();
}

function redrawStrokeOnMLCanvas(stroke) {
    // Redraw an entire stroke on ML canvas (used when syncing)
    if (!stroke || !stroke.points || stroke.points.length < 2) return;
    
    mlCtx.strokeStyle = ML_STROKE_COLOR;
    mlCtx.lineWidth = ML_STROKE_WIDTH;
    mlCtx.lineCap = 'round';
    mlCtx.lineJoin = 'round';
    mlCtx.beginPath();
    mlCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
    
    for (let i = 1; i < stroke.points.length; i++) {
        mlCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
    }
    mlCtx.stroke();
}

function saveMLCanvas() {
    // Convert ML canvas to data URL for saving/downloading
    return mlCanvas.toDataURL('image/png');
}

function downloadMLCanvas(filename = 'ml_drawing.png') {
    const dataURL = saveMLCanvas();
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataURL;
    link.click();
}

// Canny Edge Detection and Object Extraction
function detectObjectsInMLCanvas() {
    const imageData = mlCtx.getImageData(0, 0, mlCanvas.width, mlCanvas.height);
    const data = imageData.data;
    const width = mlCanvas.width;
    const height = mlCanvas.height;
    
    // Convert to grayscale
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        gray[idx] = data[i]; // Since we have white on black, just use R channel
    }
    
    // Apply Gaussian blur to reduce noise
    const blurred = gaussianBlur(gray, width, height, 1.4);
    
    // Calculate gradients using Sobel operator
    const { magnitude, direction } = sobelOperator(blurred, width, height);
    
    // Non-maximum suppression
    const suppressed = nonMaximumSuppression(magnitude, direction, width, height);
    
    // Double threshold and edge tracking by hysteresis
    const edges = hysteresisThreshold(suppressed, width, height, 30, 60);
    
    // Find connected components (objects)
    const objects = findConnectedComponents(edges, width, height);
    
    console.log('Detected', objects.length, 'objects in ML canvas');
    return objects;
}

function gaussianBlur(data, width, height, sigma) {
    const result = new Float32Array(width * height);
    const kernel = createGaussianKernel(sigma);
    const kernelSize = kernel.length;
    const radius = Math.floor(kernelSize / 2);
    
    // Horizontal pass
    const temp = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            let weightSum = 0;
            for (let i = -radius; i <= radius; i++) {
                const xi = Math.max(0, Math.min(width - 1, x + i));
                sum += data[y * width + xi] * kernel[i + radius];
                weightSum += kernel[i + radius];
            }
            temp[y * width + x] = sum / weightSum;
        }
    }
    
    // Vertical pass
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            let weightSum = 0;
            for (let i = -radius; i <= radius; i++) {
                const yi = Math.max(0, Math.min(height - 1, y + i));
                sum += temp[yi * width + x] * kernel[i + radius];
                weightSum += kernel[i + radius];
            }
            result[y * width + x] = sum / weightSum;
        }
    }
    
    return result;
}

function createGaussianKernel(sigma) {
    const size = Math.ceil(sigma * 3) * 2 + 1;
    const kernel = new Float32Array(size);
    const center = Math.floor(size / 2);
    const twoSigmaSquared = 2 * sigma * sigma;
    
    for (let i = 0; i < size; i++) {
        const x = i - center;
        kernel[i] = Math.exp(-(x * x) / twoSigmaSquared);
    }
    
    return kernel;
}

function sobelOperator(data, width, height) {
    const magnitude = new Float32Array(width * height);
    const direction = new Float32Array(width * height);
    
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let gx = 0;
            let gy = 0;
            
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
                    const kernelIdx = (ky + 1) * 3 + (kx + 1);
                    gx += data[idx] * sobelX[kernelIdx];
                    gy += data[idx] * sobelY[kernelIdx];
                }
            }
            
            const idx = y * width + x;
            magnitude[idx] = Math.sqrt(gx * gx + gy * gy);
            direction[idx] = Math.atan2(gy, gx);
        }
    }
    
    return { magnitude, direction };
}

function nonMaximumSuppression(magnitude, direction, width, height) {
    const result = new Float32Array(width * height);
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const angle = direction[idx] * 180 / Math.PI;
            let q = 255;
            let r = 255;
            
            // Angle quantization
            if ((angle >= -22.5 && angle < 22.5) || (angle >= 157.5 || angle < -157.5)) {
                q = magnitude[idx + 1];
                r = magnitude[idx - 1];
            } else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) {
                q = magnitude[(y + 1) * width + (x + 1)];
                r = magnitude[(y - 1) * width + (x - 1)];
            } else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5)) {
                q = magnitude[(y + 1) * width + x];
                r = magnitude[(y - 1) * width + x];
            } else {
                q = magnitude[(y + 1) * width + (x - 1)];
                r = magnitude[(y - 1) * width + (x + 1)];
            }
            
            if (magnitude[idx] >= q && magnitude[idx] >= r) {
                result[idx] = magnitude[idx];
            }
        }
    }
    
    return result;
}

function hysteresisThreshold(data, width, height, lowThreshold, highThreshold) {
    const result = new Uint8Array(width * height);
    const strong = 255;
    const weak = 75;
    
    // Apply double threshold
    for (let i = 0; i < data.length; i++) {
        if (data[i] >= highThreshold) {
            result[i] = strong;
        } else if (data[i] >= lowThreshold) {
            result[i] = weak;
        }
    }
    
    // Edge tracking by hysteresis
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            if (result[idx] === weak) {
                // Check if connected to strong edge
                let hasStrongNeighbor = false;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nIdx = (y + dy) * width + (x + dx);
                        if (result[nIdx] === strong) {
                            hasStrongNeighbor = true;
                            break;
                        }
                    }
                    if (hasStrongNeighbor) break;
                }
                result[idx] = hasStrongNeighbor ? strong : 0;
            }
        }
    }
    
    return result;
}

function findConnectedComponents(edges, width, height) {
    const visited = new Uint8Array(width * height);
    const objects = [];
    const minSize = 100; // Minimum pixels to consider as object
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (edges[idx] > 0 && !visited[idx]) {
                const component = floodFill(edges, visited, x, y, width, height);
                if (component.pixels.length >= minSize) {
                    objects.push(component);
                }
            }
        }
    }
    
    return objects;
}

function floodFill(edges, visited, startX, startY, width, height) {
    const stack = [[startX, startY]];
    const pixels = [];
    let minX = startX, maxX = startX;
    let minY = startY, maxY = startY;
    
    while (stack.length > 0) {
        const [x, y] = stack.pop();
        const idx = y * width + x;
        
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        if (visited[idx] || edges[idx] === 0) continue;
        
        visited[idx] = 1;
        pixels.push([x, y]);
        
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        
        // 8-connectivity
        stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
        stack.push([x + 1, y + 1], [x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1]);
    }
    
    return { pixels, minX, maxX, minY, maxY };
}

function extractObjectsForML(padding = 20) {
    const objects = detectObjectsInMLCanvas();
    const extractedObjects = [];
    
    for (const obj of objects) {
        const width = obj.maxX - obj.minX + 1;
        const height = obj.maxY - obj.minY + 1;
        
        // Calculate square bounding box with padding
        const size = Math.max(width, height) + padding * 2;
        const centerX = (obj.minX + obj.maxX) / 2;
        const centerY = (obj.minY + obj.maxY) / 2;
        
        const x1 = Math.max(0, Math.floor(centerX - size / 2));
        const y1 = Math.max(0, Math.floor(centerY - size / 2));
        const x2 = Math.min(mlCanvas.width, x1 + size);
        const y2 = Math.min(mlCanvas.height, y1 + size);
        
        // Extract the region
        const extractedWidth = x2 - x1;
        const extractedHeight = y2 - y1;
        
        // Create 128x128 canvas with proper padding
        const targetSize = 128;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = targetSize;
        tempCanvas.height = targetSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fill with black background
        tempCtx.fillStyle = 'ML_BACKGROUND_COLOR';
        tempCtx.fillRect(0, 0, targetSize, targetSize);
        
        // Calculate scaling to fit in 128x128 without distortion
        const scale = Math.min(targetSize / extractedWidth, targetSize / extractedHeight);
        const scaledWidth = extractedWidth * scale;
        const scaledHeight = extractedHeight * scale;
        
        // Center the scaled image
        const offsetX = (targetSize - scaledWidth) / 2;
        const offsetY = (targetSize - scaledHeight) / 2;
        
        // Draw the extracted region
        tempCtx.drawImage(
            mlCanvas,
            x1, y1, extractedWidth, extractedHeight,
            offsetX, offsetY, scaledWidth, scaledHeight
        );
        
        extractedObjects.push({
            canvas: tempCanvas,
            imageData: tempCtx.getImageData(0, 0, targetSize, targetSize),
            boundingBox: { x1, y1, x2, y2, centerX, centerY },
            originalSize: { width: extractedWidth, height: extractedHeight }
        });
    }
    
    console.log('Extracted', extractedObjects.length, 'objects for ML processing');
    return extractedObjects;
}

function sendObjectsToML() {
    const objects = extractObjectsForML(20);
    
    if (objects.length === 0) {
        console.log('No objects found to send to ML');
        return;
    }
    
    // Convert each object to base64 and send to server
    const mlData = objects.map((obj, idx) => ({
        image: obj.canvas.toDataURL('image/png'),
        boundingBox: obj.boundingBox,
        index: idx
    }));
    
    // Emit to server for ML processing
    socket.emit('ml.detectObjects', {
        sessionId: getOrCreateSessionId(),
        objects: mlData,
        timestamp: Date.now()
    });
    
    console.log('Sent', mlData.length, 'objects to ML server');
    
    // Optionally download for debugging
    objects.forEach((obj, idx) => {
        const link = document.createElement('a');
        link.download = `ml_object_${idx}_128x128.png`;
        link.href = obj.canvas.toDataURL('image/png');
        // Uncomment to auto-download each object:
        // link.click();
    });
    
    return objects;
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
        const strokeWidth = stroke.width ?? brushSize;
        ctx.lineWidth = strokeWidth / zoomLevel; // Adjust brush size for zoom
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

// Initialize canvases
resizeCanvas();
initializeMLCanvas();

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
    // Prevent drawing if locked, kicked, or out of ink (unless unlimited)
    if (isLocked || isKicked || (inkCapacity !== Infinity && inkAmount <= 0)) return;
    
    // Transform coordinates based on zoom and pan
    const transformedPos = screenToCanvas(x, y);
    
    isDrawing = true;
    const startTime = performance.now();
    currentStroke = {
        points: [],
        startTime,
        color: currentColor,
        width: brushSize
    };
    
    addPoint(transformedPos.x, transformedPos.y, 0);
}

function screenToCanvas(screenX, screenY) {
    // Convert screen coordinates to canvas coordinates accounting for zoom and pan
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const x = (screenX - centerX) / zoomLevel - offsetX + centerX;
    const y = (screenY - centerY) / zoomLevel - offsetY + centerY;
    
    // Clamp to world bounds so everything stays visible on quickdraw-canvas
    const clampedX = Math.max(0, Math.min(WORLD_WIDTH, x));
    const clampedY = Math.max(0, Math.min(WORLD_HEIGHT, y));
    
    return { x: clampedX, y: clampedY };
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
        const activeWidth = currentStroke.width ?? brushSize;
        ctx.lineWidth = activeWidth / zoomLevel;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        
        const lastPoint = currentStroke.points[currentStroke.points.length - 2];
        const newPoint = currentStroke.points[currentStroke.points.length - 1];
        
        ctx.moveTo(lastPoint.x, lastPoint.y);
        ctx.lineTo(newPoint.x, newPoint.y);
        ctx.stroke();

        ctx.restore();
        
        // Draw to ML canvas (black background, white strokes, fixed 12px width)
        drawToMLCanvas(lastPoint, newPoint);
    }
    
    // Consume ink - scale by zoom level and brush size (skip if Unlimited)
    if (inkCapacity !== Infinity) {
        // Cubic scaling: at zoom 0.5 (zoomed out) = 8x consumption, at zoom 4 (zoomed in) = 0.016x consumption
        // Brush size scaling: larger brushes consume more ink proportionally
        const zoomMultiplier = Math.pow(1 / zoomLevel, 3);
        const brushMultiplier = brushSize / 8; // Normalized to default brush size of 8
        const consumption = INK_CONSUMPTION_RATE * zoomMultiplier * brushMultiplier;
        inkAmount = Math.max(0, inkAmount - consumption);
        updateInkMeter();
        
        if (inkAmount <= 0) {
            lockCanvas('ink-depleted');
            endStroke(); // Force end stroke when ink depletes
        }
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
        
        // Increment MY stroke counter and check for auto-detection (per-user)
        myStrokesSinceLastDetection++;
        console.log(`📝 My stroke count: ${myStrokesSinceLastDetection}/${AUTO_DETECT_INTERVAL}`);
        
        if (myStrokesSinceLastDetection >= AUTO_DETECT_INTERVAL) {
            console.log(`🔍 Auto-detecting after ${myStrokesSinceLastDetection} of my strokes...`);
            myStrokesSinceLastDetection = 0;
            // Run detection after a short delay to ensure stroke is rendered
            setTimeout(() => {
                const objects = sendObjectsToML();
                if (objects && objects.length > 0) {
                    console.log(`✓ Auto-detected ${objects.length} object(s)`);
                }
            }, 100);
        }
        
        // Save session state after each completed stroke
        saveSessionState();
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
        const newZoom = Math.max(getMinZoom(), Math.min(MAX_ZOOM, initialZoom * scale));
        
        if (newZoom !== zoomLevel) {
            zoomLevel = newZoom;
            clampViewportToWorld();
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
    const newZoom = Math.max(getMinZoom(), Math.min(MAX_ZOOM, zoomLevel * delta));
    
    if (newZoom !== zoomLevel) {
        zoomLevel = newZoom;
        clampViewportToWorld();
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

        clampViewportToWorld();
        
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
        
        // Initialize midpoint for panning
        lastPanX = (touch1.clientX + touch2.clientX) / 2;
        lastPanY = (touch1.clientY + touch2.clientY) / 2;
        
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
        
        // Calculate midpoint (pinch center)
        const midX = (touch1.clientX + touch2.clientX) / 2;
        const midY = (touch1.clientY + touch2.clientY) / 2;
        
        // Calculate zoom
        const currentDistance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );
        
        const scale = currentDistance / initialPinchDistance;
        const newZoom = Math.max(getMinZoom(), Math.min(MAX_ZOOM, initialZoom * scale));
        
        // Zoom towards pinch center
        if (newZoom !== zoomLevel) {
            // Get canvas rect for proper coordinate conversion
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            // Convert touch midpoint to canvas coordinates
            const canvasX = (midX - rect.left) * scaleX;
            const canvasY = (midY - rect.top) * scaleY;
            
            // Calculate world position at pinch point before zoom
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const worldX = (canvasX - centerX) / zoomLevel - offsetX + centerX;
            const worldY = (canvasY - centerY) / zoomLevel - offsetY + centerY;
            
            // Apply new zoom
            const oldZoom = zoomLevel;
            zoomLevel = newZoom;
            
            // Adjust offset to keep the same world point under the pinch
            const newWorldX = (canvasX - centerX) / zoomLevel - offsetX + centerX;
            const newWorldY = (canvasY - centerY) / zoomLevel - offsetY + centerY;
            
            offsetX += (newWorldX - worldX);
            offsetY += (newWorldY - worldY);
            
            clampViewportToWorld();
        }
        
        // Pan based on midpoint movement
        if (lastPanX !== 0 && lastPanY !== 0) {
            const dx = midX - lastPanX;
            const dy = midY - lastPanY;
            
            offsetX += dx / zoomLevel;
            offsetY += dy / zoomLevel;
            clampViewportToWorld();
        }
        
        lastPanX = midX;
        lastPanY = midY;
        
        redrawCanvas();
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
        // Scale coordinates to QuickDraw format (0-1023)
        const scaledX = Math.round((point.x / WORLD_WIDTH) * QT_SCALE);
        const scaledY = Math.round((point.y / WORLD_HEIGHT) * QT_SCALE);
        
        xs.push(Math.max(0, Math.min(QT_SCALE, scaledX)));
        ys.push(Math.max(0, Math.min(QT_SCALE, scaledY)));
        ts.push(point.timestamp);
    });
    
    // Return extended format with color and width metadata
    return {
        points: [xs, ys, ts],
        color: stroke.color,
        width: stroke.width ?? brushSize
    };
}

function importStrokeFromQuickDraw(strokeData) {
    if (!strokeData) return null;
    
    let xs = null;
    let ys = null;
    let ts = null;
    
    // Handle different formats
    if (Array.isArray(strokeData)) {
        // Legacy format [xs, ys, ts]
        xs = strokeData[0];
        ys = strokeData[1];
        ts = strokeData[2] || [];
    } else if (strokeData.points) {
        // New format with points object
        if (Array.isArray(strokeData.points)) {
            xs = strokeData.points[0];
            ys = strokeData.points[1];
            ts = strokeData.points[2] || [];
        } else if (strokeData.points.xs) {
            xs = strokeData.points.xs;
            ys = strokeData.points.ys;
            ts = strokeData.points.ts || [];
        }
    }
    
    if (!xs || !ys || !Array.isArray(xs) || !Array.isArray(ys)) {
        return null;
    }
    
    const points = [];
    const count = Math.min(xs.length, ys.length);
    
    for (let i = 0; i < count; i++) {
        // Convert from QuickDraw format (0-1023) back to world coordinates
        const worldX = (xs[i] / QT_SCALE) * WORLD_WIDTH;
        const worldY = (ys[i] / QT_SCALE) * WORLD_HEIGHT;
        const timestamp = ts[i] || (i * 10);
        
        points.push({
            x: worldX,
            y: worldY,
            timestamp
        });
    }
    
    return {
        points,
        color: strokeData.color || '#3b82f6',
        width: strokeData.width || 8,
        startTime: performance.now()
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
    canvas.style.cursor = 'crosshair';
    remainingTime = ROUND_DURATION_SECONDS;
    updateTimer();
    clearSessionState();
    strokesSinceLastDetection = 0; // Reset auto-detection counter
    socket.emit('quickdraw.clear');
    console.log('Canvas cleared and session reset');
}

// Button event listeners
clearBtn.addEventListener('click', clearCanvas);
sendStrokeBtn.addEventListener('click', sendLastStroke);
sendBatchBtn.addEventListener('click', sendBatch);

const downloadMLBtn = document.getElementById('downloadMLBtn');
if (downloadMLBtn) {
    downloadMLBtn.addEventListener('click', () => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const sessionId = getOrCreateSessionId();
        downloadMLCanvas(`ml_${sessionId}_${timestamp}.png`);
        console.log('ML canvas downloaded');
    });
}

const detectObjectsBtn = document.getElementById('detectObjectsBtn');
if (detectObjectsBtn) {
    detectObjectsBtn.addEventListener('click', () => {
        console.log('Running object detection...');
        const objects = sendObjectsToML();
        if (objects && objects.length > 0) {
            alert(`Detected ${objects.length} object(s) and sent to ML server`);
        } else {
            alert('No objects detected on canvas');
        }
    });
}

function lockCanvas(reason) {
    isLocked = true;
    canvas.style.cursor = 'not-allowed';
    if (lockOverlay) {
        lockOverlay.style.display = 'flex';
        if (reason === 'kicked') {
            lockOverlay.textContent = 'Removed by Admin';
        } else if (reason === 'locked') {
            lockOverlay.textContent = 'Session Locked';
        } else if (reason === 'expired') {
            lockOverlay.textContent = 'Time Expired';
        } else {
            lockOverlay.textContent = 'Locked';
        }
    }
    console.log('Canvas locked - reason:', reason);
}

function unlockCanvas() {
    isLocked = false;
    canvas.style.cursor = 'crosshair';
    if (lockOverlay) lockOverlay.style.display = 'none';
    console.log('Canvas unlocked');
}

// Authoritative timer (server-driven) - state defined at top

function updateTimer() {
    updateTimerCircle();
}

// Initialize (client session only for local persistence; gameplay state from server)
const sessionId = getOrCreateSessionId();
console.log('Session ID:', sessionId);

initializeRandomViewport();
updateInkMeter();
updateTimer();
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

// -------------------- Admin / Server Authoritative Hooks --------------------

function applyInkLimit(raw) {
    if (!raw) return;
    const map = {
        Low: 60,
        Medium: 100,
        High: 160,
        Unlimited: Infinity // Unlimited = never run out
    };
    let capacity = map[raw] || INITIAL_INK;
    if (typeof raw === 'string' && raw.endsWith('%')) {
        const pct = parseFloat(raw.replace('%',''));
        if (!isNaN(pct) && pct > 0) capacity = Math.round(INITIAL_INK * (pct/100));
    } else if (Number.isFinite(raw)) {
        capacity = Math.max(1, raw);
    }
    
    const oldCapacity = inkCapacity;
    inkCapacity = capacity;
    
    // For unlimited, set ink to a large value
    if (capacity === Infinity) {
        inkAmount = 999999;
    } else if (capacity > oldCapacity) {
        // If capacity increased, refill to new capacity
        inkAmount = capacity;
    } else if (inkAmount > inkCapacity) {
        // If capacity decreased, clamp current ink
        inkAmount = inkCapacity;
    }
    updateInkMeter();
}

function updateInkMeter() {
    const percentage = (inkAmount / inkCapacity) * 100;
    inkFill.style.width = Math.max(0, Math.min(100, percentage)) + '%';
    updateInkCircle();
}

function updateInkCircle() {
    if (!inkCircle) {
        console.warn('[updateInkCircle] inkCircle element not found');
        return;
    }
    const pct = (inkCapacity === Infinity || inkCapacity <= 0) ? 1 : (inkAmount / inkCapacity);
    const deg = Math.max(0, Math.min(1, pct)) * 360;
    console.log(`[updateInkCircle] inkAmount=${inkAmount}, capacity=${inkCapacity}, pct=${pct}, deg=${deg}`);
    inkCircle.style.background = `conic-gradient(var(--selected-color) 0deg, var(--selected-color) ${deg}deg, #e2e8f0 ${deg}deg 360deg)`;
    if (inkCircleText) {
        inkCircleText.textContent = inkCapacity === Infinity ? '∞' : `${Math.round(pct * 100)}%`;
    }
}

function minutesAndSeconds(sec) {
    const m = String(Math.floor(sec / 60)).padStart(2, '0');
    const s = String(sec % 60).padStart(2, '0');
    return `${m}:${s}`;
}

function updateTimerCircle() {
    if (!timerCircle) {
        console.warn('[updateTimerCircle] timerCircle element not found');
        return;
    }
    const total = timerDuration > 0 ? timerDuration : remainingTime;
    const pct = total > 0 ? (remainingTime / total) : 0;
    const deg = Math.max(0, Math.min(1, pct)) * 360;
    console.log(`[updateTimerCircle] remaining=${remainingTime}, duration=${timerDuration}, total=${total}, pct=${pct}, deg=${deg}`);
    timerCircle.style.background = `conic-gradient(var(--selected-color) 0deg, var(--selected-color) ${deg}deg, #e2e8f0 ${deg}deg 360deg)`;
    if (timerCircleText) {
        timerCircleText.textContent = minutesAndSeconds(remainingTime);
    }
}

function applyConfig(cfg) {
    if (!cfg || typeof cfg !== 'object') return;
    
    // Prompt - show at top of canvas
    const prompt = cfg['Custom Prompt'] || '';
    if (promptOverlay) {
        if (prompt.length > 0) {
            promptOverlay.textContent = prompt;
            promptOverlay.style.display = 'block';
        } else {
            promptOverlay.style.display = 'none';
        }
    }
    
    // Content Mode (could influence palette someday)
    const contentMode = cfg['Content Mode'];
    if (contentMode === 'NSFW') {
        document.documentElement.setAttribute('data-content-mode', 'nsfw');
    } else {
        document.documentElement.removeAttribute('data-content-mode');
    }
    // Ink Limit
    applyInkLimit(cfg['Ink Limit']);
    // Session lock state (Session: 'Locked' | 'Open')
    if (cfg['Session'] === 'Locked') {
        lockCanvas('locked');
    } else if (!isLocked || lockOverlay?.style.display === 'flex') {
        // Only unlock if we were locked due to session (not due to expiration)
        if (timerState !== 'expired') {
            unlockCanvas();
        }
    }
}

function applyTimer(snapshot) {
    if (!snapshot) return;
    console.log('[applyTimer] received snapshot:', snapshot);
    const previousState = timerState;
    timerState = snapshot.state;
    remainingTime = snapshot.remaining;
    timerDuration = snapshot.duration || timerDuration || remainingTime;
    console.log(`[applyTimer] updated: state=${timerState}, remaining=${remainingTime}, duration=${timerDuration}`);
    updateTimer();
    
    // Handle state transitions
    if (timerState === 'expired') {
        lockCanvas('expired');
    } else if (timerState === 'running' || timerState === 'paused') {
        // If transitioning from expired to running/paused, unlock (unless session is locked)
        if (previousState === 'expired' && isLocked) {
            // Check if we should unlock (not locked for other reasons)
            const sessionLocked = lockOverlay && lockOverlay.textContent === 'Session Locked';
            const kicked = isKicked;
            if (!sessionLocked && !kicked) {
                unlockCanvas();
            }
        }
    }
}

function applyStateInit(payload) {
    if (!payload) return;
    if (payload.config) applyConfig(payload.config);
    if (payload.timer) applyTimer(payload.timer);
}

// Socket listeners for authoritative state
socket.on('state:init', (payload) => {
    console.log('[state:init] snapshot received');
    applyStateInit(payload);
});

socket.on('config:update', (cfg) => {
    console.log('[config:update] received');
    applyConfig(cfg);
});

socket.on('admin-config:update', (cfg) => {
    console.log('[admin-config:update] received');
    applyConfig(cfg);
});

socket.on('timer:update', (snapshot) => {
    applyTimer(snapshot);
});

// Fallback: if state:init not received within 1500ms after connect, fetch /api/state
let stateInitReceived = false;
socket.once('state:init', () => { stateInitReceived = true; });
setTimeout(() => {
    if (!stateInitReceived) {
        fetch('/api/state')
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data) applyStateInit(data); })
            .catch(err => console.warn('Fallback /api/state failed', err));
    }
}, 1500);

// Adjust clear behavior: do NOT modify timer; rely on server state
socket.on('quickdraw.clear', () => {
    console.log('Authoritative clear received');
    // Existing logic already clears strokes & ink; ensure ink reset respects capacity
    inkAmount = inkCapacity;
    updateInkMeter();
    // If session is open and timer not expired, allow drawing again
    if (timerState !== 'expired' && (lockOverlay?.style.display === 'flex')) {
        unlockCanvas();
    }
});

console.log('Authoritative admin hooks initialized');
