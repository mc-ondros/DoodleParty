// src/scripts/main.js

import DrawerView from '../components/DrawerView.js';
import DrawingCanvas from '../components/DrawingCanvas.js';
import { exportAsQuickDraw, sendDrawingToServer } from './quickDrawExporter.js';

const canvasElement = document.getElementById('drawingCanvas');

// Detect device type
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
const isAndroid = /Android/.test(navigator.userAgent);

// Fix canvas size for mobile devices with platform-specific handling
function resizeCanvas() {
    let width = window.innerWidth;
    let height = window.innerHeight;
    
    if (isIOS) {
        // iOS specific: account for Safari UI
        width = window.innerWidth;
        height = window.innerHeight;
    } else if (isAndroid) {
        // Android specific: use visualViewport if available
        if (window.visualViewport) {
            width = window.visualViewport.width;
            height = window.visualViewport.height;
        }
    }
    
    // Set canvas display size
    canvasElement.style.width = width + 'px';
    canvasElement.style.height = height + 'px';
    
    // Only set internal resolution on first load to prevent clearing
    if (!canvasElement.width || canvasElement.width === 300) {
        canvasElement.width = 1024;
        canvasElement.height = 640;
    }
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);
window.addEventListener('orientationchange', () => {
    setTimeout(resizeCanvas, 100);
});

if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', resizeCanvas);
}

const initialColor =
    document.querySelector('.color-dot.active')?.dataset.color || '#3b82f6';

const drawingCanvas = new DrawingCanvas(canvasElement, {
    initialColor,
    initialBrushSize: 8,
    initialInk: 200,
    consumptionRate: 0.04
});

const drawerView = new DrawerView({
    onColorChange: (color) => drawingCanvas.setBrushColor(color),
    onBrushSizeChange: (size) => drawingCanvas.setBrushSize(size)
});

drawerView.updateBrushSize(drawingCanvas.brushSize);
drawerView.updateInkLevel(100);

drawingCanvas.onInkChange = (inkAmount) => {
    // Convert 0-200 range to 0-100 percentage for display
    const percentage = (inkAmount / 200) * 100;
    drawerView.updateInkLevel(Math.round(percentage));
};

drawingCanvas.onInkDepleted = () => drawingCanvas.lock();

const ROUND_DURATION_SECONDS = 90;
startRoundTimer(ROUND_DURATION_SECONDS);

function startRoundTimer(totalSeconds) {
    let remaining = totalSeconds;
    drawerView.updateTimer(formatTime(remaining));

    const timerId = setInterval(() => {
        remaining -= 1;
        drawerView.updateTimer(formatTime(Math.max(remaining, 0)));

        if (remaining <= 0) {
            clearInterval(timerId);
            drawingCanvas.lock();
            handleRoundEnd();
        }
    }, 1000);
}

function handleRoundEnd() {
    // Export drawing in Quick, Draw! format
    const strokes = drawingCanvas.getStrokes();
    const quickDrawData = exportAsQuickDraw(strokes);
    
    console.log('Drawing completed:', quickDrawData);
    
    // Example: Send to server via socket (uncomment when socket is ready)
    // if (socket) {
    //     sendDrawingToServer(socket, strokes);
    // }
}

function formatTime(totalSeconds) {
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    return `${minutes}:${seconds}`;
}