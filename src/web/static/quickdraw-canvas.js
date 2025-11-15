(function () {
    'use strict';

    const canvas = document.getElementById('socketCanvas');
    const ctx = canvas.getContext('2d');
    const connectionStatus = document.getElementById('connectionStatus');
    const lastEvent = document.getElementById('lastEvent');
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');
    const overlayClearBtn = document.getElementById('overlayClearBtn');
    const eventLog = document.getElementById('eventLog');

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
    const logEntries = [];
    const LOG_LIMIT = 3;

    const PAINT_STYLE = '#f5f7ff';
    const BACKGROUND = '#05050a';

    let socket = null;

    function resetCanvas() {
        ctx.save();
        ctx.fillStyle = BACKGROUND;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        strokeQueue.length = 0;
    }

    function updateStatus(message, statusClass) {
        if (!connectionStatus) return;
        connectionStatus.textContent = message;
        connectionStatus.classList.remove('success', 'offline', 'neutral');
        connectionStatus.classList.add(statusClass);
    }

    function logEvent(message) {
        if (!eventLog) return;
        logEntries.unshift(message);
        if (logEntries.length > LOG_LIMIT) {
            logEntries.splice(LOG_LIMIT);
        }
        eventLog.innerHTML = logEntries.map((entry) => `<span>${entry}</span>`).join('');
        if (lastEvent) {
            lastEvent.textContent = message;
        }
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
        logEvent('Canvas cleared');
    }

    function initializeSocket() {
        if (typeof io === 'undefined') {
            updateStatus('socket.io missing', 'offline');
            logEvent('Socket.IO client not loaded');
            return;
        }

        const socketUrl = searchParams.get('socketUrl');
        const socketOpts = {
            transports: ['websocket'],
            upgrade: true
        };
        socket = socketUrl ? io(socketUrl, socketOpts) : io(socketOpts);

        socket.on('connect', () => {
            updateStatus('Connected', 'success');
            logEvent('Connected to socket');
        });

        socket.on('disconnect', (reason) => {
            updateStatus('Disconnected', 'offline');
            logEvent(`Disconnected (${reason})`);
        });

        socket.on('connect_error', (error) => {
            updateStatus('Connect error', 'offline');
            logEvent(`Connect error: ${error.message || error}`);
        });

        socket.on('quickdraw.stroke', (payload) => {
            handleStrokeEvent(payload);
            logEvent('Received quickdraw.stroke');
        });

        socket.on('quickdraw.batch', (payload) => {
            handleBatchEvent(payload);
            logEvent('Received quickdraw.batch');
        });

        socket.on('quickdraw.drawing', (payload) => {
            handleDrawingEvent(payload);
            logEvent('Received quickdraw.drawing');
        });

        socket.on('quickdraw.clear', () => {
            handleClearEvent();
        });
    }

    clearCanvasBtn.addEventListener('click', () => {
        resetCanvas();
        logEvent('Canvas cleared (manual)');
        if (socket && socket.connected) {
            socket.emit('quickdraw.ack', { status: 'cleared' });
        }
    });

    if (overlayClearBtn) {
        overlayClearBtn.addEventListener('click', () => {
            clearCanvasBtn.click();
        });
    }

    resetCanvas();
    initializeSocket();
}());
