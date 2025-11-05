// Canvas Setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const brushSizeInput = document.getElementById('brushSize');
const sizeDisplay = document.getElementById('sizeDisplay');

// Result Elements
const resultBox = document.getElementById('resultBox');
const loadingBox = document.getElementById('loadingBox');
const errorBox = document.getElementById('errorBox');

// Drawing State
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set canvas resolution for better drawing quality
// Using 512x512 for smooth drawing, will be downsampled to 128x128 by backend
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = 512;
    canvas.height = 512;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Initialize canvas
resizeCanvas();

// Drawing Functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = (e.clientX || e.touches[0].clientX) - rect.left;
    lastY = (e.clientY || e.touches[0].clientY) - rect.top;
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    // Draw line from last point to current point
    ctx.strokeStyle = '#000';
    ctx.lineWidth = brushSizeInput.value;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;
}

function stopDrawing() {
    isDrawing = false;
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
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideAllResults();
});

// Show/Hide Results
function hideAllResults() {
    resultBox.classList.add('hidden');
    loadingBox.classList.add('hidden');
    errorBox.classList.add('hidden');
    errorBox.classList.remove('show');
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
    verdict.textContent = data.verdict === 'IN-DISTRIBUTION' ? '✓ Valid QuickDraw' : '✗ Not QuickDraw';
    
    // Update result box class for styling
    resultBox.classList.remove('in-distribution', 'out-of-distribution');
    if (data.verdict === 'IN-DISTRIBUTION') {
        resultBox.classList.add('in-distribution');
    } else {
        resultBox.classList.add('out-of-distribution');
    }

    // Update verdict text
    document.getElementById('verdictText').textContent = data.verdict_text;

    // Update confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    document.getElementById('confidence').textContent = confidencePercent + '%';
    document.getElementById('confidenceBar').style.width = confidencePercent + '%';

    // Update raw probability
    document.getElementById('rawProb').textContent = data.raw_probability.toFixed(4);

    // Update threshold
    document.getElementById('thresholdVal').textContent = data.threshold.toFixed(1);

    // Show result box
    resultBox.classList.remove('hidden');
}

// Predict Function
async function makePrediction() {
    // Validate that something has been drawn
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Check if canvas is not empty (has non-white pixels)
    let isEmpty = true;
    for (let i = 0; i < data.length; i += 4) {
        // If alpha < 255 or any color channel is not 255 (white), canvas is not empty
        if (data[i + 3] < 255 || data[i] < 255 || data[i + 1] < 255 || data[i + 2] < 255) {
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

        // Send to backend
        const response = await fetch('/api/predict', {
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
        }
    } catch (error) {
        console.warn('Could not reach server:', error);
    }
});

// Allow Enter key to predict
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        makePrediction();
    }
});
