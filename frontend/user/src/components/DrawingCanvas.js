class DrawingCanvas {
    constructor(canvas, { initialColor = '#000000', initialBrushSize = 8, initialInk = 100, consumptionRate = 0.08 } = {}) {
        if (!canvas) throw new Error('Canvas element is required for DrawingCanvas.');

        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false });
        this.brushColor = initialColor;
        this.brushSize = initialBrushSize;
        this.inkPercent = initialInk;
        this.consumptionRate = consumptionRate;
        this.isDrawing = false;
        this.locked = false;
        this.activePointerId = null;
        this.pendingDraw = false;
        this.cachedRect = null;

        // Stroke tracking for Quick, Draw! format
        this.currentStroke = [];
        this.allStrokes = [];

        this.onInkChange = null;
        this.onInkDepleted = null;

        this.handlePointerDown = this.handlePointerDown.bind(this);
        this.handlePointerMove = this.handlePointerMove.bind(this);
        this.handlePointerUp = this.handlePointerUp.bind(this);

        this.attachEvents();
    }

    attachEvents() {
        this.canvas.addEventListener('pointerdown', this.handlePointerDown);
        this.canvas.addEventListener('pointermove', this.handlePointerMove);
        window.addEventListener('pointerup', this.handlePointerUp);
    }

    handlePointerDown(event) {
        if (this.locked || this.inkPercent <= 0) return;
        // Ignore if another pointer is already active
        if (this.activePointerId !== null && this.activePointerId !== event.pointerId) {
            event.preventDefault();
            return;
        }
        this.activePointerId = event.pointerId;
        this.isDrawing = true;
        
        // Cache rect for performance
        this.cachedRect = this.canvas.getBoundingClientRect();
        this.lastPoint = this.getCanvasCoords(event);
        
        // Start new stroke
        this.currentStroke = [{
            x: this.lastPoint.x,
            y: this.lastPoint.y,
            timestamp: Date.now()
        }];
        
        this.drawDot(this.lastPoint);
        this.canvas.setPointerCapture?.(event.pointerId);
    }

    handlePointerMove(event) {
        // Ignore events from non-active pointers
        if (event.pointerId !== this.activePointerId) return;
        if (!this.isDrawing || this.locked || this.inkPercent <= 0) return;
        
        event.preventDefault();
        
        // Get all events including predicted for lower latency
        const coalescedEvents = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
        const predictedEvents = event.getPredictedEvents ? event.getPredictedEvents() : [];
        
        // Process coalesced events for actual drawing
        for (const evt of coalescedEvents) {
            const currentPoint = this.getCanvasCoords(evt);
            
            // Add point to current stroke
            this.currentStroke.push({
                x: currentPoint.x,
                y: currentPoint.y,
                timestamp: Date.now()
            });
            
            this.drawStroke(this.lastPoint, currentPoint);
            const distance = this.distanceBetween(this.lastPoint, currentPoint);
            this.consumeInk(distance * this.consumptionRate);
            this.lastPoint = currentPoint;
        }
        
        // Draw predicted path for lower perceived latency
        if (predictedEvents.length > 0) {
            const ctx = this.ctx;
            ctx.save();
            ctx.globalAlpha = 0.5;
            let prevPoint = this.lastPoint;
            for (const evt of predictedEvents) {
                const predictedPoint = this.getCanvasCoords(evt);
                this.drawStroke(prevPoint, predictedPoint);
                prevPoint = predictedPoint;
            }
            ctx.restore();
        }
    }

    handlePointerUp(event) {
        // Only handle up event for active pointer
        if (event.pointerId !== this.activePointerId) return;
        if (this.isDrawing) {
            this.isDrawing = false;
            this.activePointerId = null;
            this.cachedRect = null;
            
            // Save completed stroke
            if (this.currentStroke.length > 0) {
                this.allStrokes.push({ points: this.currentStroke });
                this.currentStroke = [];
            }
            
            this.canvas.releasePointerCapture?.(event.pointerId);
        }
    }

    getCanvasCoords(event) {
        const rect = this.cachedRect || this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    drawStroke(from, to) {
        const ctx = this.ctx;
        ctx.strokeStyle = this.brushColor;
        ctx.lineWidth = this.brushSize;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
    }

    drawDot(point) {
        const ctx = this.ctx;
        ctx.fillStyle = this.brushColor;
        ctx.beginPath();
        ctx.arc(point.x, point.y, this.brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
    }

    distanceBetween(a, b) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    consumeInk(amount) {
        if (amount <= 0) return;
        this.inkPercent = Math.max(this.inkPercent - amount, 0);
        this.onInkChange?.(this.inkPercent);
        if (this.inkPercent === 0) {
            this.onInkDepleted?.();
        }
    }

    setBrushColor(color) {
        this.brushColor = color;
    }

    setBrushSize(size) {
        this.brushSize = size;
    }

    refillInk(percent = 100) {
        this.inkPercent = Math.min(percent, 100);
        this.onInkChange?.(this.inkPercent);
        this.locked = false;
    }

    lock() {
        this.locked = true;
    }

    getStrokes() {
        return this.allStrokes;
    }

    clearStrokes() {
        this.allStrokes = [];
        this.currentStroke = [];
    }
}

export default DrawingCanvas;