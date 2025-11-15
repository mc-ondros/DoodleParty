/**
 * Drawing Canvas Hook
 *
 * Provides canvas drawing functionality with ink consumption.
 * Handles pointer events for drawing strokes and dots.
 * 
 * Related:
 * - src/components/DrawingCanvas.tsx (canvas component)
 * - src/components/InkMeter.tsx (ink display)
 * 
 * Exports:
 * - useDrawingCanvas (hook)
 */

import { useRef, useCallback, useState, useEffect } from 'react';

interface Point {
  x: number;
  y: number;
}

interface DrawingCanvasOptions {
  initialColor?: string;
  initialBrushSize?: number;
  initialInk?: number;
  consumptionRate?: number;
  onInkChange?: (inkPercent: number) => void;
  onInkDepleted?: () => void;
}

export const useDrawingCanvas = ({
  initialColor = '#000000',
  initialBrushSize = 8,
  initialInk = 100,
  consumptionRate = 0.08,
  onInkChange,
  onInkDepleted,
}: DrawingCanvasOptions = {}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [brushColor, setBrushColor] = useState(initialColor);
  const [brushSize, setBrushSize] = useState(initialBrushSize);
  const [inkPercent, setInkPercent] = useState(initialInk);
  const [locked, setLocked] = useState(false);
  
  const isDrawingRef = useRef(false);
  const lastPointRef = useRef<Point | null>(null);

  const getCanvasCoords = useCallback((event: PointerEvent): Point => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }, []);

  const drawStroke = useCallback((from: Point, to: Point) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }, [brushColor, brushSize]);

  const drawDot = useCallback((point: Point) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = brushColor;
    ctx.beginPath();
    ctx.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
  }, [brushColor, brushSize]);

  const distanceBetween = (a: Point, b: Point): number => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const consumeInk = useCallback((amount: number) => {
    if (amount <= 0) return;
    setInkPercent((prev) => {
      const newInk = Math.max(prev - amount, 0);
      onInkChange?.(newInk);
      if (newInk === 0) {
        onInkDepleted?.();
      }
      return newInk;
    });
  }, [onInkChange, onInkDepleted]);

  const handlePointerDown = useCallback((event: PointerEvent) => {
    if (locked || inkPercent <= 0) return;
    isDrawingRef.current = true;
    lastPointRef.current = getCanvasCoords(event);
    drawDot(lastPointRef.current);
    canvasRef.current?.setPointerCapture?.(event.pointerId);
  }, [locked, inkPercent, getCanvasCoords, drawDot]);

  const handlePointerMove = useCallback((event: PointerEvent) => {
    if (!isDrawingRef.current || locked || inkPercent <= 0) return;
    const currentPoint = getCanvasCoords(event);
    if (lastPointRef.current) {
      drawStroke(lastPointRef.current, currentPoint);
      const distance = distanceBetween(lastPointRef.current, currentPoint);
      consumeInk(distance * consumptionRate);
    }
    lastPointRef.current = currentPoint;
  }, [locked, inkPercent, getCanvasCoords, drawStroke, consumeInk, consumptionRate]);

  const handlePointerUp = useCallback((event: PointerEvent) => {
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      canvasRef.current?.releasePointerCapture?.(event.pointerId);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('pointerdown', handlePointerDown as any);
    canvas.addEventListener('pointermove', handlePointerMove as any);
    window.addEventListener('pointerup', handlePointerUp as any);

    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown as any);
      canvas.removeEventListener('pointermove', handlePointerMove as any);
      window.removeEventListener('pointerup', handlePointerUp as any);
    };
  }, [handlePointerDown, handlePointerMove, handlePointerUp]);

  const refillInk = useCallback((percent = 100) => {
    const newInk = Math.min(percent, 100);
    setInkPercent(newInk);
    onInkChange?.(newInk);
    setLocked(false);
  }, [onInkChange]);

  const lock = useCallback(() => {
    setLocked(true);
  }, []);

  return {
    canvasRef,
    brushColor,
    setBrushColor,
    brushSize,
    setBrushSize,
    inkPercent,
    locked,
    refillInk,
    lock,
  };
};
