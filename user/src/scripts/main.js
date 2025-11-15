// src/scripts/main.js

import DrawerView from '../components/DrawerView.js';
import DrawingCanvas from '../components/DrawingCanvas.js';

const canvasElement = document.getElementById('drawingCanvas');
const initialColor =
    document.querySelector('.color-dot.active')?.dataset.color || '#3b82f6';

const drawingCanvas = new DrawingCanvas(canvasElement, {
    initialColor,
    initialBrushSize: 8,
    initialInk: 100
});

const drawerView = new DrawerView({
    onColorChange: (color) => drawingCanvas.setBrushColor(color),
    onBrushSizeChange: (size) => drawingCanvas.setBrushSize(size)
});

drawerView.updateBrushSize(drawingCanvas.brushSize);
drawerView.updateInkLevel(100);

drawingCanvas.onInkChange = (inkPercent) => {
    drawerView.updateInkLevel(Math.round(inkPercent));
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
        }
    }, 1000);
}

function formatTime(totalSeconds) {
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    return `${minutes}:${seconds}`;
}