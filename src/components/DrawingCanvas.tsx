/**
 * Drawing Canvas Component
 *
 * Provides the core drawing surface where users can create strokes.
 * Handles canvas initialization and basic drawing functionality.
 * 
 * Related:
 * - src/hooks/useDraggable.tsx (drawing interaction)
 * - src/services/gameService.ts (stroke processing)
 * 
 * Exports:
 * - DrawingCanvas (functional component)
 */

import React from 'react';

export const DrawingCanvas: React.FC = () => {
  // Main canvas component with drawing logic
  return (
    <div className="drawing-canvas">
      <canvas id="canvas"></canvas>
    </div>
  );
};
