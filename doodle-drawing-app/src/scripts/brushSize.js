function initializeBrushSize(canvasContext) {
    const brushSizeInput = document.getElementById('brushSize');
    
    // Set default brush size
    let brushSize = 5;
    brushSizeInput.value = brushSize;

    // Update brush size on input change
    brushSizeInput.addEventListener('input', (event) => {
        brushSize = event.target.value;
        canvasContext.lineWidth = brushSize;
    });

    return brushSize;
}

export { initializeBrushSize };