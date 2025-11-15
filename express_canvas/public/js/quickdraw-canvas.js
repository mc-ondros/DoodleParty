(function () {
    'use strict';

    const canvas = document.getElementById('socketCanvas');
    const ctx = canvas.getContext('2d');
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');

    if (!canvas || !ctx) {
        console.warn('Socket canvas elements are missing');
        return;
    }

    const searchParams = new URLSearchParams(window.location.search);
    const coordMax = Number(searchParams.get('coordMax')) || 255;
    const scaleX = canvas.width / Math.max(coordMax, 1);
    const scaleY = canvas.height / Math.max(coordMax, 1);
    const strokeQueue = [];
    let rafPending = false;

    const PAINT_STYLE = '#0e1726';
    const BACKGROUND = '#ffffff';

    let socket = null;

    function resetCanvas() {
        ctx.save();
        ctx.fillStyle = BACKGROUND;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        strokeQueue.length = 0;
    }

    function updateStatus(message) {
        console.debug(message);
    }

    function drawStrokeFromArray(stroke) {
        if (!Array.isArray(stroke) || stroke.length < 2) return;
        const xs = stroke[0];
        const ys = stroke[1];
        if (!xs || !ys) return;
        const count = Math.min(xs.length, ys.length);
        if (count < 2) return;

        ctx.beginPath();
        ctx.moveTo(xs[0] * scaleX, ys[0] * scaleY);
        for (let i = 1; i < count; i += 1) {
            ctx.lineTo(xs[i] * scaleX, ys[i] * scaleY);
        }
        ctx.stroke();
    }

    function flushQueue() {
        rafPending = false;
        if (strokeQueue.length === 0) return;
        ctx.save();
        ctx.strokeStyle = PAINT_STYLE;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        while (strokeQueue.length) {
            const stroke = strokeQueue.shift();
            drawStrokeFromArray(stroke);
        }
        ctx.restore();
    }

    function scheduleFlush() {
        if (!rafPending) {
            rafPending = true;
            requestAnimationFrame(flushQueue);
        }
    }

    function queueStroke(stroke) {
        strokeQueue.push(stroke);
        scheduleFlush();
    }

    function handleStrokeEvent(payload) {
        const stroke = Array.isArray(payload) ? payload : payload && payload.stroke;
        if (!stroke) return;
        queueStroke(stroke);
    }

    function handleBatchEvent(payload) {
        const strokes = Array.isArray(payload)
            ? payload
            : payload && payload.strokes;
        if (!Array.isArray(strokes)) return;
        strokes.forEach((stroke) => queueStroke(stroke));
    }

    function handleDrawingEvent(payload) {
        const drawing = Array.isArray(payload)
            ? payload
            : payload && payload.drawing;
        if (!Array.isArray(drawing)) return;
        drawing.forEach((stroke) => queueStroke(stroke));
    }

    function handleClearEvent() {
        resetCanvas();
    }

    function initializeSocket() {
        if (typeof io === 'undefined') {
            updateStatus('socket.io missing');
            return;
        }

        const socketUrl = searchParams.get('socketUrl');
        const socketOpts = {
            transports: ['websocket'],
            upgrade: true
        };
        socket = socketUrl ? io(socketUrl, socketOpts) : io(socketOpts);

        socket.on('connect', () => {
            updateStatus('Connected to socket');
        });

        socket.on('disconnect', (reason) => {
            updateStatus(`Socket disconnected (${reason})`);
        });

        socket.on('connect_error', (error) => {
            updateStatus(`Socket connect error: ${error.message || error}`);
        });

        socket.on('quickdraw.stroke', (payload) => {
            handleStrokeEvent(payload);
            updateStatus('Received quickdraw.stroke');
        });

        socket.on('quickdraw.batch', (payload) => {
            handleBatchEvent(payload);
            updateStatus('Received quickdraw.batch');
        });

        socket.on('quickdraw.drawing', (payload) => {
            handleDrawingEvent(payload);
            updateStatus('Received quickdraw.drawing');
        });

        socket.on('quickdraw.clear', () => {
            handleClearEvent();
        });
    }

    clearCanvasBtn.addEventListener('click', () => {
        resetCanvas();
        if (socket && socket.connected) {
            socket.emit('quickdraw.ack', { status: 'cleared' });
        }
    });

    resetCanvas();
    initializeSocket();
}());
