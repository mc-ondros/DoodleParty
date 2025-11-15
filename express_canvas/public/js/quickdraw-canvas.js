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
    let coordMax = Number(searchParams.get('coordMax')) || 1919;
    let scaleX = 1;
    let scaleY = 1;
    const strokeQueue = [];
    let rafPending = false;

    const PAINT_STYLE = '#0e1726';
    const BACKGROUND = '#ffffff';
    const DEFAULT_WIDTH = 3;
    const WORLD_WIDTH = 1920;
    const WORLD_HEIGHT = 1080;

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

    function drawNormalizedStroke(stroke, animate = false) {
        if (!stroke) return;
        const xs = stroke.xs;
        const ys = stroke.ys;
        const ts = stroke.ts;
        if (!Array.isArray(xs) || !Array.isArray(ys)) return;
        const count = Math.min(xs.length, ys.length);
        if (count < 2) return;

        if (!animate || !ts || ts.length === 0) {
            // Draw instantly (style already set in context)
            ctx.beginPath();
            ctx.moveTo(xs[0] * scaleX, ys[0] * scaleY);
            for (let i = 1; i < count; i += 1) {
                ctx.lineTo(xs[i] * scaleX, ys[i] * scaleY);
            }
            ctx.stroke();
        } else {
            // Animate in real-time based on timestamps
            let currentIndex = 0;
            let lastDrawnIndex = 0;
            const startTime = performance.now();
            const firstTimestamp = ts[0];
            
            // Use computed style properties from flushQueue
            const color = stroke.computedColor;
            const width = stroke.computedWidth;
            
            function drawNextSegment() {
                const elapsed = performance.now() - startTime;
                
                // Find all points that should be drawn by now
                while (currentIndex < count && (ts[currentIndex] - firstTimestamp) <= elapsed) {
                    currentIndex++;
                }
                
                // Only draw new segments since last frame
                if (currentIndex > lastDrawnIndex) {
                    ctx.save();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = width;
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';
                    
                    ctx.beginPath();
                    if (lastDrawnIndex === 0) {
                        ctx.moveTo(xs[0] * scaleX, ys[0] * scaleY);
                        if (currentIndex > 1) {
                            ctx.lineTo(xs[1] * scaleX, ys[1] * scaleY);
                        }
                        lastDrawnIndex = 1;
                    } else {
                        ctx.moveTo(xs[lastDrawnIndex] * scaleX, ys[lastDrawnIndex] * scaleY);
                    }
                    
                    for (let i = lastDrawnIndex + 1; i < currentIndex; i++) {
                        ctx.lineTo(xs[i] * scaleX, ys[i] * scaleY);
                    }
                    ctx.stroke();
                    ctx.restore();
                    
                    lastDrawnIndex = currentIndex - 1;
                }
                
                if (currentIndex < count) {
                    requestAnimationFrame(drawNextSegment);
                }
            }
            
            drawNextSegment();
        }
    }

    function normalizeStroke(stroke) {
        if (!stroke) return null;

        let xs = null;
        let ys = null;
        let ts = null;

        // Legacy array format [xs, ys, ts]
        if (Array.isArray(stroke)) {
            xs = Array.isArray(stroke[0]) ? stroke[0] : null;
            ys = Array.isArray(stroke[1]) ? stroke[1] : null;
            ts = Array.isArray(stroke[2]) ? stroke[2] : null;
            if (!xs || !ys) return null;
            return {
                xs,
                ys,
                ts,
                color: PAINT_STYLE,
                width: DEFAULT_WIDTH
            };
        }

        if (typeof stroke !== 'object') return null;

        if (Array.isArray(stroke.points)) {
            xs = Array.isArray(stroke.points[0]) ? stroke.points[0] : null;
            ys = Array.isArray(stroke.points[1]) ? stroke.points[1] : null;
            ts = Array.isArray(stroke.points[2]) ? stroke.points[2] : null;
        } else if (stroke.points && typeof stroke.points === 'object') {
            xs = Array.isArray(stroke.points.xs) ? stroke.points.xs : null;
            ys = Array.isArray(stroke.points.ys) ? stroke.points.ys : null;
            ts = Array.isArray(stroke.points.ts) ? stroke.points.ts : null;
        }

        if (!xs || !ys) return null;

        return {
            xs,
            ys,
            ts,
            color: stroke.color || PAINT_STYLE,
            width: typeof stroke.width === 'number' ? stroke.width : DEFAULT_WIDTH
        };
    }

    function flushQueue() {
        rafPending = false;
        if (strokeQueue.length === 0) return;
        const widthScale = scaleX || (canvas.width / WORLD_WIDTH);
        while (strokeQueue.length) {
            const stroke = strokeQueue.shift();
            if (!stroke) continue;
            
            // Calculate and store style properties in the stroke object
            stroke.computedColor = stroke.color || PAINT_STYLE;
            stroke.computedWidth = Math.max(1, (stroke.width || DEFAULT_WIDTH) * widthScale);
            
            if (!stroke.animate) {
                // Draw instantly with proper style
                ctx.save();
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.strokeStyle = stroke.computedColor;
                ctx.lineWidth = stroke.computedWidth;
                drawNormalizedStroke(stroke, false);
                ctx.restore();
            } else {
                // Animate real-time strokes (style will be applied in animation)
                drawNormalizedStroke(stroke, true);
            }
        }
    }

    function scheduleFlush() {
        if (!rafPending) {
            rafPending = true;
            requestAnimationFrame(flushQueue);
        }
    }

    function queueStroke(rawStroke, animate = false) {
        console.log('queueStroke called:', { rawStroke, animate });
        const normalized = normalizeStroke(rawStroke);
        if (!normalized) {
            console.warn('Failed to normalize stroke:', rawStroke);
            return;
        }
        console.log('Normalized stroke:', normalized);
        normalized.animate = animate;
        strokeQueue.push(normalized);
        console.log('Stroke queued, queue length:', strokeQueue.length);
        scheduleFlush();
    }

    function handleStrokeEvent(payload, realTime = false) {
        console.log('handleStrokeEvent called with:', { payload, realTime });
        const stroke = Array.isArray(payload)
            ? payload
            : (payload && (payload.stroke || payload));
        if (!stroke) {
            console.warn('No stroke found in payload');
            return;
        }
        console.log('Processing stroke:', stroke);
        // Animate real-time strokes, pass along immediately for instant rendering
        queueStroke(stroke, realTime);
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

    function updateScaleFactors() {
        scaleX = canvas.width / Math.max(coordMax, 1);
        scaleY = canvas.height / Math.max(coordMax, 1);
    }

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const viewportWidth = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
        const viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
        
        // Always use world dimensions for internal resolution
        const targetWidth = WORLD_WIDTH;
        const targetHeight = WORLD_HEIGHT;

        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            // Let CSS handle display dimensions to maintain aspect ratio
        }

        updateScaleFactors();
        // Don't clear canvas on init - let sync restore previous strokes
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
            console.log('Socket connected, waiting for automatic sync...');
        });

        socket.on('disconnect', (reason) => {
            updateStatus(`Socket disconnected (${reason})`);
        });

        socket.on('connect_error', (error) => {
            updateStatus(`Socket connect error: ${error.message || error}`);
        });

        socket.on('quickdraw.stroke', (payload) => {
            console.log('Received quickdraw.stroke:', payload);
            // Flash the canvas border to show stroke received
            canvas.style.border = '5px solid red';
            setTimeout(() => { canvas.style.border = 'none'; }, 200);
            handleStrokeEvent(payload, true);
            updateStatus('Received quickdraw.stroke at ' + new Date().toLocaleTimeString());
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

        socket.on('quickdraw.sync', (payload) => {
            console.log('Received quickdraw.sync event');
            console.log('Payload type:', Array.isArray(payload) ? 'array' : typeof payload);
            console.log('Payload length:', payload ? payload.length : 'N/A');
            
            if (!Array.isArray(payload)) {
                console.warn('Sync payload is not an array:', payload);
                return;
            }
            
            console.log('Drawing', payload.length, 'synced strokes...');
            // Draw all synced strokes instantly (no animation)
            payload.forEach((stroke, index) => {
                handleStrokeEvent(stroke, false);
                if ((index + 1) % 10 === 0) {
                    console.log('Drew', index + 1, 'strokes...');
                }
            });
            console.log('Sync complete:', payload.length, 'strokes drawn');
            updateStatus('Synced ' + payload.length + ' strokes');
        });
    }

    clearCanvasBtn.addEventListener('click', () => {
        resetCanvas();
        if (socket && socket.connected) {
            socket.emit('quickdraw.clear');
            console.log('Emitted quickdraw.clear event');
        }
    });

    resizeCanvas();
    initializeSocket();

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('orientationchange', () => {
        // Delay resize slightly so orientation dimensions settle
        setTimeout(resizeCanvas, 150);
    });
}());
