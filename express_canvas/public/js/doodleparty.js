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
const INITIAL_INK = 100;
const INK_CONSUMPTION_RATE = 1;
const WORLD_WIDTH = 1920;
const WORLD_HEIGHT = 1080;
const QT_SCALE = Math.max(WORLD_WIDTH, WORLD_HEIGHT) - 1; // QuickDraw coordinate scale
const DEBUG_MODE = false; // Set to true to show manual send buttons
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
    // Request sync immediately on connection to ensure we get existing strokes
    setTimeout(() => {
        socket.emit('quickdraw.requestSync');
        console.log('Requested stroke sync from server');
    }, 100);
});

socket.on('disconnect', () => {
    updateSocketStatus('disconnected');
    console.log('Socket disconnected');
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
        
        // Import synced strokes into local canvas
        let imported = 0;
        syncedStrokes.forEach(strokeData => {
            const stroke = importStrokeFromQuickDraw(strokeData);
            if (stroke && stroke.points.length > 1) {
                strokes.push(stroke);
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
    
    // Reset canvas state
    inkAmount = INITIAL_INK;
    updateInkMeter();
    isLocked = false;
    canvas.style.cursor = 'crosshair';
    remainingTime = ROUND_DURATION_SECONDS;
    
    // Restart the timer
    startTimer();
    
    // Clear session storage
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
    
    console.log('Canvas fully reset from clear event');
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
    }
    
    // Consume ink - scale by zoom level and brush size
    // Cubic scaling: at zoom 0.5 (zoomed out) = 8x consumption, at zoom 4 (zoomed in) = 0.016x consumption
    // Brush size scaling: larger brushes consume more ink proportionally
    const zoomMultiplier = Math.pow(1 / zoomLevel, 3);
    const brushMultiplier = brushSize / 8; // Normalized to default brush size of 8
    const consumption = INK_CONSUMPTION_RATE * zoomMultiplier * brushMultiplier;
    inkAmount = Math.max(0, inkAmount - consumption);
    updateInkMeter();
    
    if (inkAmount <= 0) {
        lockCanvas();
        endStroke(); // Force end stroke when ink depletes
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
let activeTimerId = null;

function updateTimer() {
    const minutes = String(Math.floor(remainingTime / 60)).padStart(2, '0');
    const seconds = String(remainingTime % 60).padStart(2, '0');
    timerDisplay.textContent = `${minutes}:${seconds}`;
}

function startTimer() {
    // Clear any existing timer first
    if (activeTimerId) {
        clearInterval(activeTimerId);
    }
    
    updateTimer();
    
    activeTimerId = setInterval(() => {
        remainingTime -= 1;
        updateTimer();
        
        if (remainingTime <= 0) {
            clearInterval(activeTimerId);
            activeTimerId = null;
            lockCanvas();
            handleRoundEnd();
        }
    }, 1000);
}

function handleRoundEnd() {
    console.log('Round ended - canvas locked');
    // Don't send strokes when timer depletes, just lock the canvas
    // Strokes are already sent in real-time as they're drawn
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
    // Initialize random viewport for new session
    initializeRandomViewport();
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
