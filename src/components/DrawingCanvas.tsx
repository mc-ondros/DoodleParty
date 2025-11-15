/**
 * Drawing Canvas Component
 *
 * Provides the core drawing surface where users can create strokes.
 * Handles canvas initialization and basic drawing functionality.
 * 
 * Related:
 * - src/hooks/useDrawingCanvas.tsx (drawing logic)
 * - src/hooks/useDraggable.tsx (drawing interaction)
 * - src/services/gameService.ts (stroke processing)
 * 
 * Exports:
 * - DrawingCanvas (functional component)
 */

import React from 'react';
import { useDrawingCanvas } from '../hooks/useDrawingCanvas';

interface DrawingCanvasProps {
  onInkChange?: (inkPercent: number) => void;
  onInkDepleted?: () => void;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  onInkChange,
  onInkDepleted,
}) => {
  const { canvasRef } = useDrawingCanvas({
    initialColor: '#3b82f6',
    initialBrushSize: 8,
    initialInk: 100,
    consumptionRate: 0.08,
    onInkChange,
    onInkDepleted,
  });

  return (
    <div className="drawing-canvas">
      <canvas ref={canvasRef} id="drawingCanvas" width={800} height={600}></canvas>
    </div>
  );
};
