// Canvas Setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const brushSizeInput = document.getElementById('brushSize');
const sizeDisplay = document.getElementById('sizeDisplay');

// Toggle Buttons
const autoEraseBtn = document.getElementById('autoEraseBtn');
const realTimeBtn = document.getElementById('realTimeBtn');

// Result Elements
const emptyState = document.getElementById('emptyState');
const resultBox = document.getElementById('resultBox');
const loadingBox = document.getElementById('loadingBox');
const errorBox = document.getElementById('errorBox');

// Drawing State
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Real-time analysis
let realTimeAnalysisTimer = null;
let isAnalyzing = false;
const REALTIME_DEBOUNCE_MS = 500;  // Wait 500ms after last stroke before analyzing

// Toggle States
let autoEraseEnabled = true;
let realTimeEnabled = false;

// Statistics tracking
let strokeCount = 0;
let pixelCount = 0;
let drawingStartTime = null;
let drawingTimer = null;
let strokeHistory = [];
let currentStroke = null;
let brushSizes = [];
let totalStrokeLength = 0;

// Helper function to safely set textContent
function safeSetTextContent(id, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
    } else {
        console.warn(`Element with id '${id}' not found`);
    }
}

// Real-time analysis scheduler
function scheduleRealTimeAnalysis() {
    // Only schedule if real-time mode is enabled
    if (!realTimeEnabled) return;
    
    // Clear any existing timer
    if (realTimeAnalysisTimer) {
        clearTimeout(realTimeAnalysisTimer);
    }
    
    // Schedule analysis after debounce period
    realTimeAnalysisTimer = setTimeout(() => {
        if (!isAnalyzing) {
            console.log('Real-time analysis triggered');
            makePrediction();
        }
    }, REALTIME_DEBOUNCE_MS);
}

// Set canvas resolution for better drawing quality
// Using 512x512 for smooth drawing, will be downsampled to 128x128 by backend
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = 512;
    canvas.height = 512;
    ctx.fillStyle = '#f3f4f6';  // Neutral gray background matching design system
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set default drawing properties for smoother lines
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
}

// Initialize canvas
resizeCanvas();

// Statistics Functions
function updateStatistics() {
    // Update stroke count
    safeSetTextContent('totalStrokes', strokeCount);

    // Calculate canvas coverage
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    let drawnPixels = 0;

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // Count non-background pixels
        if (r !== 243 || g !== 244 || b !== 246) {
            drawnPixels++;
        }
    }

    pixelCount = drawnPixels;
    const totalPixels = canvas.width * canvas.height;
    const coverage = ((drawnPixels / totalPixels) * 100).toFixed(2);
    safeSetTextContent('canvasCoverage', coverage + '%');
    safeSetTextContent('pixelsDrawn', drawnPixels.toLocaleString());

    // Calculate average stroke length
    if (strokeCount > 0 && totalStrokeLength > 0) {
        const avgStrokeLength = (totalStrokeLength / strokeCount).toFixed(1);
        safeSetTextContent('avgStrokeLength', avgStrokeLength + 'px');
    } else {
        safeSetTextContent('avgStrokeLength', '-');
    }

    // Calculate max brush size
    if (brushSizes.length > 0) {
        const maxBrushSize = Math.max(...brushSizes);
        safeSetTextContent('maxBrushSize', maxBrushSize + 'px');
    } else {
        safeSetTextContent('maxBrushSize', '-');
    }

    // Calculate drawing speed (pixels per second)
    if (drawingStartTime) {
        const elapsedSeconds = (Date.now() - drawingStartTime) / 1000;
        if (elapsedSeconds > 0 && drawnPixels > 0) {
            const drawingSpeed = (drawnPixels / elapsedSeconds).toFixed(0);
            safeSetTextContent('drawingSpeed', drawingSpeed + ' px/s');
        } else {
            safeSetTextContent('drawingSpeed', '-');
        }
    } else {
        safeSetTextContent('drawingSpeed', '-');
    }

    // Calculate complexity score (0-100)
    if (strokeCount > 0 && drawnPixels > 0) {
        // Base complexity on stroke count, coverage, and average stroke length
        const strokeDensity = strokeCount / (drawnPixels / 1000);
        const avgStrokeLength = totalStrokeLength / strokeCount;
        const complexityScore = Math.min(100, Math.round(
            (strokeDensity * 0.4) + (coverage * 0.3) + (avgStrokeLength / 10 * 0.3)
        ));
        safeSetTextContent('complexityScore', complexityScore.toString());
    } else {
        safeSetTextContent('complexityScore', '-');
    }

    // Calculate pressure variance (simulated based on brush size variations)
    if (brushSizes.length > 1) {
        const avgBrushSize = brushSizes.reduce((a, b) => a + b, 0) / brushSizes.length;
        const variance = brushSizes.reduce((acc, size) => acc + Math.pow(size - avgBrushSize, 2), 0) / brushSizes.length;
        const stdDev = Math.sqrt(variance);
        safeSetTextContent('pressureVariance', stdDev.toFixed(1));
    } else {
        safeSetTextContent('pressureVariance', '0.0');
    }

    // Calculate line quality (based on stroke smoothness and consistency)
    if (strokeCount > 0) {
        const avgBrushSize = brushSizes.length > 0 ? brushSizes.reduce((a, b) => a + b, 0) / brushSizes.length : 0;
        const brushConsistency = brushSizes.length > 1 ?
            100 - (Math.abs(Math.max(...brushSizes) - Math.min(...brushSizes)) / avgBrushSize * 100) : 100;
        const lineQuality = Math.max(0, Math.min(100, Math.round(brushConsistency)));
        safeSetTextContent('lineQuality', lineQuality + '%');
    } else {
        safeSetTextContent('lineQuality', '-');
    }
}

function updateDrawingTime() {
    if (drawingStartTime) {
        const elapsed = Math.floor((Date.now() - drawingStartTime) / 1000);
        safeSetTextContent('drawingTime', elapsed + 's');
    }
}

function resetStatistics() {
    strokeCount = 0;
    pixelCount = 0;
    drawingStartTime = null;
    strokeHistory = [];
    currentStroke = null;
    brushSizes = [];
    totalStrokeLength = 0;
    if (drawingTimer) {
        clearInterval(drawingTimer);
        drawingTimer = null;
    }
    safeSetTextContent('totalStrokes', '0');
    safeSetTextContent('canvasCoverage', '0%');
    safeSetTextContent('drawingTime', '0s');
    safeSetTextContent('pixelsDrawn', '0');
    safeSetTextContent('avgStrokeLength', '-');
    safeSetTextContent('maxBrushSize', '-');
    safeSetTextContent('drawingSpeed', '-');
    safeSetTextContent('complexityScore', '-');
    safeSetTextContent('pressureVariance', '0.0');
    safeSetTextContent('lineQuality', '-');
}

// Drawing Functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    lastX = ((e.clientX || e.touches[0].clientX) - rect.left) * scaleX;
    lastY = ((e.clientY || e.touches[0].clientY) - rect.top) * scaleY;
    
    // Start tracking drawing time on first stroke
    if (!drawingStartTime) {
        drawingStartTime = Date.now();
        drawingTimer = setInterval(updateDrawingTime, 1000);
    }

    // Increment stroke count
    strokeCount++;

    // Track brush size
    const currentBrushSize = parseInt(brushSizeInput.value);
    if (!brushSizes.includes(currentBrushSize)) {
        brushSizes.push(currentBrushSize);
    }

    // Create new stroke object to track this stroke's points
    currentStroke = {
        points: [{x: lastX, y: lastY}],
        brushSize: currentBrushSize,
        timestamp: Date.now(),
        color: '#000',
        length: 0
    };

    // Begin path for smoother drawing
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
}

function draw(e) {
    if (!isDrawing) return;
    
    e.preventDefault(); // Prevent scrolling on touch devices

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = ((e.clientX || e.touches[0].clientX) - rect.left) * scaleX;
    const y = ((e.clientY || e.touches[0].clientY) - rect.top) * scaleY;

    // Add point to current stroke
    if (currentStroke) {
        currentStroke.points.push({x, y});
    }

    // Track stroke length
    if (currentStroke) {
        const distance = Math.sqrt(Math.pow(x - lastX, 2) + Math.pow(y - lastY, 2));
        currentStroke.length += distance;
    }

    // Set drawing properties
    ctx.strokeStyle = '#000';
    ctx.lineWidth = brushSizeInput.value;

    // Use quadratic curve for smoother lines
    const midX = (lastX + x) / 2;
    const midY = (lastY + y) / 2;

    ctx.quadraticCurveTo(lastX, lastY, midX, midY);
    ctx.stroke();

    // Continue the path
    ctx.beginPath();
    ctx.moveTo(midX, midY);

    lastX = x;
    lastY = y;
}

function stopDrawing() {
    if (isDrawing) {
        // Draw final point to complete the stroke
        ctx.lineTo(lastX, lastY);
        ctx.stroke();

        // Save completed stroke to history
        if (currentStroke && currentStroke.points.length > 0) {
            strokeHistory.push(currentStroke);
            totalStrokeLength += currentStroke.length;
            currentStroke = null;
        }

        isDrawing = false;
        // Update statistics after stroke is complete
        updateStatistics();

        // Schedule real-time analysis if enabled
        scheduleRealTimeAnalysis();
    }
}

// Event Listeners for Drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch support for mobile
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e);
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopDrawing();
});

// Brush Size Control
brushSizeInput.addEventListener('input', (e) => {
    sizeDisplay.textContent = e.target.value;
});

// Clear Canvas
clearBtn.addEventListener('click', async () => {
    ctx.fillStyle = '#f3f4f6';  // Neutral gray background matching design system
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetStatistics();
    showEmptyState();
});

// Show/Hide Results
function hideAllResults() {
    emptyState.classList.add('hidden');
    resultBox.classList.add('hidden');
    loadingBox.classList.add('hidden');
    errorBox.classList.remove('show');
    errorBox.classList.add('hidden');
}

function showEmptyState() {
    hideAllResults();
    emptyState.classList.remove('hidden');
}

function showLoading() {
    hideAllResults();
    loadingBox.classList.remove('hidden');
}

function showError(message) {
    hideAllResults();
    safeSetTextContent('errorMessage', message);
    errorBox.classList.add('show');
    errorBox.classList.remove('hidden');
}

function showResult(data) {
    hideAllResults();

    // Update verdict
    const verdict = document.getElementById('verdict');
    const isPositive = data.verdict === 'PENIS';
    verdict.textContent = isPositive ? 'Positive Match' : 'Negative Match';
    
    // Update result box class for styling
    resultBox.classList.remove('in-distribution', 'out-of-distribution');
    if (isPositive) {
        resultBox.classList.add('in-distribution');
    } else {
        resultBox.classList.add('out-of-distribution');
    }

    // Update confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    safeSetTextContent('confidence', confidencePercent + '%');
    document.getElementById('confidenceBar').style.width = confidencePercent + '%';

    // Update raw probability
    safeSetTextContent('rawProb', data.raw_probability.toFixed(4));

    // Update threshold
    safeSetTextContent('thresholdVal', data.threshold.toFixed(1));

    // Update timing statistics if available
    if (data.drawing_statistics) {
        const stats = data.drawing_statistics;
        safeSetTextContent('responseTime', stats.response_time_ms + ' ms');
        safeSetTextContent('inferenceTime', stats.inference_time_ms + ' ms');
    }

    // Update region detection details if available
    const regionDetails = document.getElementById('regionDetails');
    if (data.detection_details) {
        const details = data.detection_details;
        safeSetTextContent('patchesAnalyzed', details.num_patches_analyzed);
        safeSetTextContent('earlyStopped', details.early_stopped ? 'Yes' : 'No');
        safeSetTextContent('aggregationStrategy', details.aggregation_strategy);
        regionDetails.classList.remove('hidden');
    } else {
        regionDetails.classList.add('hidden');
    }

    // Update additional info row
    // Note: Total Time removed from UI, only show model info now
    if (data.model_info) {
        const modelMatch = data.model_info.match(/\((.*?)\)/);
        if (modelMatch) {
            safeSetTextContent('modelType', modelMatch[1]);
        }
    }

    // Show result box
    resultBox.classList.remove('hidden');

    // Handle auto-erasing of inappropriate content (if toggle is enabled)
    if (autoEraseEnabled && isPositive) {
        console.log('Auto-erase enabled - inappropriate content detected, clearing canvas');

        // Clear canvas after a short delay
        setTimeout(() => {
            ctx.fillStyle = '#f3f4f6';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            updateStatistics();
        }, 1000);
    } else if (!autoEraseEnabled && isPositive) {
        console.log('Auto-erase disabled - inappropriate content detected but not erased');
    }
}

// Predict Function
async function makePrediction() {
    // Prevent concurrent analysis
    if (isAnalyzing) {
        console.log('Analysis already in progress, skipping');
        return;
    }
    
    // Validate that something has been drawn
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Check if canvas is not empty (has non-gray pixels)
    // Gray background is #f3f4f6 which is RGB(243, 244, 246)
    let isEmpty = true;
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // If pixel is not the background gray color, canvas is not empty
        if (r !== 243 || g !== 244 || b !== 246) {
            isEmpty = false;
            break;
        }
    }

    if (isEmpty) {
        // In real-time mode, silently skip empty canvas
        if (realTimeEnabled) {
            return;
        }
        showError('Please draw something before requesting a verdict!');
        return;
    }

    // Set analyzing flag
    isAnalyzing = true;

    // Show loading (only show loading UI if not in real-time mode)
    if (!realTimeEnabled) {
        showLoading();
    }
    predictBtn.disabled = true;

    try {
        // Get canvas as base64 image
        const imageData64 = canvas.toDataURL('image/png');

        // Always use simple detection endpoint
        const endpoint = '/api/predict';
        const detectionMode = 'simple';

        // Prepare request body
        const requestBody = {
            image: imageData64
        };

        // Send to backend
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            showResult(result);
        } else {
            showError(result.error || 'Failed to get prediction');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Error connecting to the server: ' + error.message);
    } finally {
        // Reset analyzing flag
        isAnalyzing = false;
        // Re-enable predict button (unless in real-time mode where it should stay enabled)
        predictBtn.disabled = false;
    }
}

// Predict Button
predictBtn.addEventListener('click', makePrediction);

// Toggle Button Event Listeners
autoEraseBtn.addEventListener('click', () => {
    autoEraseEnabled = !autoEraseEnabled;
    autoEraseBtn.classList.toggle('active', autoEraseEnabled);
    console.log('Auto-erase ' + (autoEraseEnabled ? 'enabled' : 'disabled'));
});

realTimeBtn.addEventListener('click', () => {
    realTimeEnabled = !realTimeEnabled;
    realTimeBtn.classList.toggle('active', realTimeEnabled);

    if (realTimeEnabled) {
        console.log('Real-time analysis enabled');
        // Optionally trigger immediate analysis if there's content
        scheduleRealTimeAnalysis();
    } else {
        console.log('Real-time analysis disabled');
        // Clear any pending analysis
        if (realTimeAnalysisTimer) {
            clearTimeout(realTimeAnalysisTimer);
// Initialize toggle button states
autoEraseBtn.classList.add('active');
realTimeBtn.classList.remove('active');

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Server health:', data);
        if (!data.model_loaded) {
            showError('ML model not loaded. Please check the server logs.');
        } else {
            showEmptyState();
        }
    } catch (error) {
        console.warn('Could not reach server:', error);
        showEmptyState();
    }
});

// Allow Enter key to predict
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        makePrediction();
    }
});
