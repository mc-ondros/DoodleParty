class DrawingCanvas {
    constructor(canvas, { initialColor = '#000000', initialBrushSize = 8, initialInk = 100, consumptionRate = 0.08 } = {}) {
        if (!canvas) throw new Error('Canvas element is required for DrawingCanvas.');

        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.brushColor = initialColor;
        this.brushSize = initialBrushSize;
        this.inkPercent = initialInk;
        this.consumptionRate = consumptionRate;
        this.isDrawing = false;
        this.locked = false;

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
        this.isDrawing = true;
        this.lastPoint = this.getCanvasCoords(event);
        this.drawDot(this.lastPoint);
        this.canvas.setPointerCapture?.(event.pointerId);
    }

    handlePointerMove(event) {
        if (!this.isDrawing || this.locked || this.inkPercent <= 0) return;
        const currentPoint = this.getCanvasCoords(event);
        this.drawStroke(this.lastPoint, currentPoint);
        const distance = this.distanceBetween(this.lastPoint, currentPoint);
        this.consumeInk(distance * this.consumptionRate);
        this.lastPoint = currentPoint;
    }

    handlePointerUp(event) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.canvas.releasePointerCapture?.(event.pointerId);
        }
    }

    getCanvasCoords(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
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
}

export default DrawingCanvas;