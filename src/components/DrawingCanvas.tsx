import React from 'react';

export const DrawingCanvas: React.FC = () => {
  // Main canvas component with drawing logic
  return (
    <div className="drawing-canvas">
      <canvas id="canvas"></canvas>
    </div>
  );
};
