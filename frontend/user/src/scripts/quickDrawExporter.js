// src/scripts/quickDrawExporter.js

/**
 * Exports canvas strokes to Google Quick, Draw! format
 * Format: array of strokes, each stroke is [x[], y[], t[]]
 * where x, y are coordinates and t are timestamps in milliseconds
 */
export function exportAsQuickDraw(strokes) {
    return strokes.map(stroke => {
        const x = [];
        const y = [];
        const t = [];
        
        stroke.points.forEach(point => {
            x.push(Math.round(point.x));
            y.push(Math.round(point.y));
            t.push(point.timestamp);
        });
        
        return [x, y, t];
    });
}

/**
 * Send drawing data to server via socket
 */
export function sendDrawingToServer(socket, strokes) {
    const quickDrawFormat = exportAsQuickDraw(strokes);
    socket.emit('drawing-data', {
        drawing: quickDrawFormat,
        timestamp: Date.now()
    });
}
