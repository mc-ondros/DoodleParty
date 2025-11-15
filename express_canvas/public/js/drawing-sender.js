const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const sendStrokeBtn = document.getElementById('sendStrokeBtn');
const sendBatchBtn = document.getElementById('sendBatchBtn');
const socketStatus = document.getElementById('socketStatus');

const DRAW_COLOR = '#f5f7ff';
const BACKGROUND = '#0e0e15';
const QtScale = 255 / canvas.width;

let isDrawing = false;
let strokes = [];
let currentStroke = null;

const socket = io({ transports: ['websocket'] });

socket.on('connect', () => {
    socketStatus.textContent = 'connected';
    socketStatus.style.color = '#4cd964';
});

socket.on('disconnect', () => {
    socketStatus.textContent = 'disconnected';
    socketStatus.style.color = '#ff5c5c';
});

socket.on('connect_error', () => {
    socketStatus.textContent = 'connect error';
    socketStatus.style.color = '#ffb347';
});

function resetCanvas() {
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    currentStroke = null;
}

function beginStroke(x, y) {
    isDrawing = true;
    currentStroke = { xs: [], ys: [], ts: [], start: performance.now() };
    recordPoint(x, y);
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function recordPoint(x, y) {
    const t = performance.now() - currentStroke.start;
    currentStroke.xs.push(Math.round(x * QtScale));
    currentStroke.ys.push(Math.round(y * QtScale));
    currentStroke.ts.push(Math.round(t));
}

function drawLine(x, y) {
    ctx.lineTo(x, y);
    ctx.stroke();
}

function endStroke() {
    if (!isDrawing || !currentStroke) return;
    isDrawing = false;
    ctx.closePath();
    strokes.push(currentStroke);
    currentStroke = null;
}

function handlePointerDown(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    ctx.strokeStyle = DRAW_COLOR;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    beginStroke(x, y);
}

function handlePointerMove(event) {
    if (!isDrawing || !currentStroke) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    recordPoint(x, y);
    drawLine(x, y);
}

function handlePointerUp() {
    endStroke();
}

function sendStroke(stroke) {
    if (!stroke || stroke.xs.length < 2) return;
    socket.emit('quickdraw.stroke', [stroke.xs, stroke.ys, stroke.ts]);
}

function sendBatch() {
    if (!strokes.length) return;
    const batch = strokes.map((stroke) => [stroke.xs, stroke.ys, stroke.ts]);
    socket.emit('quickdraw.batch', batch);
}

clearBtn.addEventListener('click', () => {
    resetCanvas();
    socket.emit('quickdraw.clear');
});

sendStrokeBtn.addEventListener('click', () => {
    if (strokes.length === 0) return;
    sendStroke(strokes[strokes.length - 1]);
});

sendBatchBtn.addEventListener('click', () => {
    sendBatch();
});

canvas.addEventListener('pointerdown', handlePointerDown);
canvas.addEventListener('pointermove', handlePointerMove);
canvas.addEventListener('pointerup', handlePointerUp);
canvas.addEventListener('pointerleave', handlePointerUp);

resetCanvas();
