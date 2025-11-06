// Canvas Setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const brushSizeInput = document.getElementById('brushSize');
const sizeDisplay = document.getElementById('sizeDisplay');

// Result Elements
const emptyState = document.getElementById('emptyState');
const resultBox = document.getElementById('resultBox');
const loadingBox = document.getElementById('loadingBox');
const errorBox = document.getElementById('errorBox');

// Drawing State
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Statistics tracking
let strokeCount = 0;
let pixelCount = 0;
let drawingStartTime = null;
let drawingTimer = null;

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
    document.getElementById('totalStrokes').textContent = strokeCount;
    
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
    document.getElementById('canvasCoverage').textContent = coverage + '%';
    document.getElementById('pixelsDrawn').textContent = drawnPixels.toLocaleString();
}

function updateDrawingTime() {
    if (drawingStartTime) {
        const elapsed = Math.floor((Date.now() - drawingStartTime) / 1000);
        document.getElementById('drawingTime').textContent = elapsed + 's';
    }
}

function resetStatistics() {
    strokeCount = 0;
    pixelCount = 0;
    drawingStartTime = null;
    if (drawingTimer) {
        clearInterval(drawingTimer);
        drawingTimer = null;
    }
    document.getElementById('totalStrokes').textContent = '0';
    document.getElementById('canvasCoverage').textContent = '0%';
    document.getElementById('drawingTime').textContent = '0s';
    document.getElementById('pixelsDrawn').textContent = '0';
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
        
        isDrawing = false;
        // Update statistics after stroke is complete
        updateStatistics();
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
clearBtn.addEventListener('click', () => {
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
    document.getElementById('errorMessage').textContent = message;
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
    document.getElementById('confidence').textContent = confidencePercent + '%';
    document.getElementById('confidenceBar').style.width = confidencePercent + '%';

    // Update raw probability
    document.getElementById('rawProb').textContent = data.raw_probability.toFixed(4);

    // Update threshold
    document.getElementById('thresholdVal').textContent = data.threshold.toFixed(1);

    // Update timing statistics if available
    if (data.drawing_statistics) {
        const stats = data.drawing_statistics;
        document.getElementById('responseTime').textContent = stats.response_time_ms + ' ms';
        document.getElementById('inferenceTime').textContent = stats.inference_time_ms + ' ms';
    }

    // Update region detection details if available
    const regionDetails = document.getElementById('regionDetails');
    if (data.detection_details) {
        const details = data.detection_details;
        document.getElementById('patchesAnalyzed').textContent = details.num_patches_analyzed;
        document.getElementById('earlyStopped').textContent = details.early_stopped ? 'Yes' : 'No';
        document.getElementById('aggregationStrategy').textContent = details.aggregation_strategy;
        regionDetails.classList.remove('hidden');
    } else {
        regionDetails.classList.add('hidden');
    }

    // Update additional info row
    if (data.drawing_statistics) {
        document.getElementById('totalTimeResult').textContent = data.drawing_statistics.response_time_ms.toFixed(0) + 'ms';
    }
    
    if (data.model_info) {
        const modelMatch = data.model_info.match(/\((.*?)\)/);
        if (modelMatch) {
            document.getElementById('modelType').textContent = modelMatch[1];
        }
    }

    // Show result box
    resultBox.classList.remove('hidden');
}

// Predict Function
async function makePrediction() {
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
        showError('Please draw something before requesting a verdict!');
        return;
    }

    // Show loading
    showLoading();
    predictBtn.disabled = true;

    try {
        // Get canvas as base64 image
        const imageData64 = canvas.toDataURL('image/png');

        // Use multi-scale region-based detection for robustness
        const endpoint = '/api/predict/region';

        // Send to backend
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData64
            })
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
        predictBtn.disabled = false;
    }
}

// Predict Button
predictBtn.addEventListener('click', makePrediction);

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        if (!health.model_loaded) {
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
