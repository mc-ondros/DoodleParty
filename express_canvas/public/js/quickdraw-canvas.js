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
    const DEFAULT_WIDTH = 3;
    const SOURCE_CANVAS_WIDTH = 1024;

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

    function drawNormalizedStroke(stroke) {
        if (!stroke) return;
        const xs = stroke.xs;
        const ys = stroke.ys;
        if (!Array.isArray(xs) || !Array.isArray(ys)) return;
        const count = Math.min(xs.length, ys.length);
        if (count < 2) return;

        ctx.beginPath();
        ctx.moveTo(xs[0] * scaleX, ys[0] * scaleY);
        for (let i = 1; i < count; i += 1) {
            ctx.lineTo(xs[i] * scaleX, ys[i] * scaleY);
        }
        ctx.stroke();
    }

    function normalizeStroke(stroke) {
        if (!stroke) return null;

        // Legacy array format [xs, ys, ts]
        if (Array.isArray(stroke)) {
            const xs = Array.isArray(stroke[0]) ? stroke[0] : null;
            const ys = Array.isArray(stroke[1]) ? stroke[1] : null;
            if (!xs || !ys) return null;
            return {
                xs,
                ys,
                color: PAINT_STYLE,
                width: DEFAULT_WIDTH
            };
        }

        if (typeof stroke !== 'object') return null;

        let xs = null;
        let ys = null;

        if (Array.isArray(stroke.points)) {
            xs = Array.isArray(stroke.points[0]) ? stroke.points[0] : null;
            ys = Array.isArray(stroke.points[1]) ? stroke.points[1] : null;
        } else if (stroke.points && typeof stroke.points === 'object') {
            xs = Array.isArray(stroke.points.xs) ? stroke.points.xs : null;
            ys = Array.isArray(stroke.points.ys) ? stroke.points.ys : null;
        }

        if (!xs || !ys) return null;

        return {
            xs,
            ys,
            color: stroke.color || PAINT_STYLE,
            width: typeof stroke.width === 'number' ? stroke.width : DEFAULT_WIDTH
        };
    }

    function flushQueue() {
        rafPending = false;
        if (strokeQueue.length === 0) return;
        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        const widthScale = canvas.width / SOURCE_CANVAS_WIDTH;
        while (strokeQueue.length) {
            const stroke = strokeQueue.shift();
            if (!stroke) continue;
            ctx.strokeStyle = stroke.color || PAINT_STYLE;
            const strokeWidth = Math.max(1, (stroke.width || DEFAULT_WIDTH) * widthScale);
            ctx.lineWidth = strokeWidth;
            drawNormalizedStroke(stroke);
        }
        ctx.restore();
    }

    function scheduleFlush() {
        if (!rafPending) {
            rafPending = true;
            requestAnimationFrame(flushQueue);
        }
    }

    function queueStroke(rawStroke) {
        const normalized = normalizeStroke(rawStroke);
        if (!normalized) return;
        strokeQueue.push(normalized);
        scheduleFlush();
    }

    function handleStrokeEvent(payload) {
        const stroke = Array.isArray(payload)
            ? payload
            : (payload && (payload.stroke || payload));
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
