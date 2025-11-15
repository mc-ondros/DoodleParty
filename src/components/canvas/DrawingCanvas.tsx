import React, { useEffect, useRef } from 'react';
import type { DrawData } from '../../types';

type Tool = 'brush' | 'bucket';

interface DrawingCanvasProps {
  paths: DrawData[];
  color: string;
  strokeWeight: number;
  tool: Tool;
  onDraw: (data: DrawData) => void;
  onDrawProgress?: (pt: { x: number; y: number }) => void;
  isReadOnly?: boolean;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ paths, color, strokeWeight, tool, onDraw, onDrawProgress, isReadOnly }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const drawing = useRef(false);
  const panning = useRef(false);
  const lastPan = useRef<{ x: number; y: number } | null>(null);
  const currentPath = useRef<{ x: number; y: number }[]>([]);
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });

  const resize = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(container.clientWidth * dpr);
    canvas.height = Math.floor(container.clientHeight * dpr);
    canvas.style.width = container.clientWidth + 'px';
    canvas.style.height = container.clientHeight + 'px';
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.scale(dpr, dpr);
    redraw();
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // apply transform
    const s = scaleRef.current;
    const o = offsetRef.current;
    ctx.setTransform(s, 0, 0, s, o.x, o.y);
    // draw saved paths in world coords
    paths.forEach(p => drawPath(ctx, p));
    // draw current path
    if (currentPath.current.length > 1) {
      drawPath(ctx, { path: currentPath.current, color, strokeWeight, isFill: false });
    }
    ctx.restore();
  };

  const drawPath = (ctx: CanvasRenderingContext2D, data: DrawData) => {
    if (data.isFill) {
      ctx.fillStyle = data.color;
      data.path.forEach(pt => ctx.fillRect(pt.x, pt.y, 1, 1));
      return;
    }
    ctx.strokeStyle = data.color;
    ctx.lineWidth = data.strokeWeight;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    data.path.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();
  };

  const performFloodFillFromCanvas = (wx: number, wy: number, fillColor: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    // render current world into an offscreen with world transform
    const off = document.createElement('canvas');
    off.width = canvas.width;
    off.height = canvas.height;
    const octx = off.getContext('2d');
    if (!octx) return;
    // draw with same transform
    octx.save();
    const s = scaleRef.current; const o = offsetRef.current;
    octx.setTransform(s, 0, 0, s, o.x, o.y);
    // white bg + black strokes for robust boundary
    octx.fillStyle = 'white';
    octx.fillRect(0,0,off.width, off.height);
    paths.forEach(p => {
      if (p.isFill) {
        octx.fillStyle = '#000';
        p.path.forEach(pt => octx.fillRect(pt.x, pt.y, 1, 1));
      } else {
        octx.strokeStyle = '#000';
        octx.lineWidth = Math.max(2, p.strokeWeight);
        octx.lineJoin = 'round'; octx.lineCap = 'round';
        octx.beginPath();
        p.path.forEach((pt, i)=>{ if(i===0) octx.moveTo(pt.x, pt.y); else octx.lineTo(pt.x, pt.y); });
        octx.stroke();
      }
    });
    octx.restore();

    // seed in screen coords
    const sx = wx * s + o.x;
    const sy = wy * s + o.y;
    const img = octx.getImageData(0, 0, off.width, off.height);
    const data = img.data;
    const W = off.width, H = off.height;
    const idx = (x:number,y:number)=> (y*W + x)*4;
    const inBounds = (x:number,y:number)=> x>=0 && x<W && y>=0 && y<H;

    // we consider white as fillable space; black as boundary
    const isBoundary = (x:number,y:number)=> {
      const i = idx(x,y);
      return data[i] < 128; // dark pixel
    };

    const queue: [number,number][] = [];
    const visited = new Uint8Array(W*H);
    const startX = Math.floor(sx), startY = Math.floor(sy);
    if (!inBounds(startX,startY) || isBoundary(startX,startY)) return;
    queue.push([startX,startY]);
    visited[startY*W+startX] = 1;
    const pts: {x:number;y:number}[] = [];
    const step = 1; // pixel step
    while(queue.length && pts.length < 200000){
      const [x,y] = queue.shift()!;
      // record world-space point
      const wxp = (x - o.x)/s;
      const wyp = (y - o.y)/s;
      pts.push({ x: wxp, y: wyp });
      // neighbors
      const nb = [[1,0],[-1,0],[0,1],[0,-1]];
      for (const [dx,dy] of nb){
        const nx = x + dx*step, ny = y + dy*step;
        const key = ny*W + nx;
        if (!inBounds(nx,ny) || visited[key]) continue;
        if (!isBoundary(nx,ny)) { visited[key]=1; queue.push([nx,ny]); }
      }
    }
    if (pts.length) onDraw({ path: pts, color: fillColor, strokeWeight: 1, isFill: true });
  };

  useEffect(() => {
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  useEffect(() => {
    redraw();
  }, [paths, color, strokeWeight]);

  const screenToWorld = (sx: number, sy: number) => {
    const s = scaleRef.current;
    const o = offsetRef.current;
    return { x: (sx - o.x) / s, y: (sy - o.y) / s };
  };

  const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    return screenToWorld(sx, sy);
  };

  const onDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isReadOnly) return;
    if (e.button === 1 || e.button === 2) {
      panning.current = true;
      lastPan.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const pos = getPos(e);
    if (tool === 'bucket') {
      performFloodFillFromCanvas(pos.x, pos.y, color);
      return;
    }
    drawing.current = true;
    currentPath.current = [pos];
    redraw();
  };

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (panning.current && lastPan.current) {
      const dx = e.clientX - lastPan.current.x;
      const dy = e.clientY - lastPan.current.y;
      lastPan.current = { x: e.clientX, y: e.clientY };
      offsetRef.current.x += dx;
      offsetRef.current.y += dy;
      redraw();
      return;
    }
    if (!drawing.current || isReadOnly) return;
    const pos = getPos(e);
    currentPath.current.push(pos);
    if (onDrawProgress) onDrawProgress(pos);
    redraw();
  };

  const onUp = () => {
    if (panning.current) {
      panning.current = false;
      lastPan.current = null;
      return;
    }
    if (!drawing.current || isReadOnly) return;
    drawing.current = false;
    if (currentPath.current.length > 1) {
      onDraw({ path: [...currentPath.current], color, strokeWeight, isFill: false });
    }
    currentPath.current = [];
    redraw();
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    if (e.ctrlKey) {
      // zoom
      const zoom = Math.exp(-e.deltaY * 0.001);
      const prevScale = scaleRef.current;
      const nextScale = Math.min(4, Math.max(0.25, prevScale * zoom));
      const wx = (sx - offsetRef.current.x) / prevScale;
      const wy = (sy - offsetRef.current.y) / prevScale;
      offsetRef.current.x = sx - wx * nextScale;
      offsetRef.current.y = sy - wy * nextScale;
      scaleRef.current = nextScale;
      redraw();
    } else {
      // pan via wheel
      offsetRef.current.x -= e.deltaX;
      offsetRef.current.y -= e.deltaY;
      redraw();
    }
  };

  return (
    <div ref={containerRef} className="w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full block cursor-crosshair"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
        onContextMenu={(e)=>e.preventDefault()}
        onWheel={onWheel}
      />
    </div>
  );
};

export default DrawingCanvas;
