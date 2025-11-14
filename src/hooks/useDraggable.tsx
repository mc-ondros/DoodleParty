import { useRef, useCallback } from 'react';

export const useDraggable = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);

  const startDrawing = useCallback((e: MouseEvent | TouchEvent) => {
    isDrawing.current = true;
  }, []);

  const stopDrawing = useCallback(() => {
    isDrawing.current = false;
  }, []);

  const draw = useCallback((e: MouseEvent | TouchEvent) => {
    if (!isDrawing.current) return;
    // Drawing logic
  }, []);

  return {
    canvasRef,
    startDrawing,
    stopDrawing,
    draw,
  };
};
