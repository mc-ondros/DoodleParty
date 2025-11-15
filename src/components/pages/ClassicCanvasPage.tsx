import React, { useMemo, useState } from 'react';
import DrawingCanvas from '../canvas/DrawingCanvas';
import Toolbar from '../canvas/Toolbar';
import NavigationToolbar from '../canvas/NavigationToolbar';
import InkMeter from '../canvas/InkMeter';
import type { DrawData } from '../../types';
import type { Page } from '../../App';

interface Props { setCurrentPage: (p: Page) => void; }

const ClassicCanvasPage: React.FC<Props> = ({ setCurrentPage }) => {
  const [paths, setPaths] = useState<DrawData[]>([]);
  const [redoStack, setRedo] = useState<DrawData[]>([]);
  const [color, setColor] = useState('#22c55e');
  const [stroke, setStroke] = useState(8);
  const [tool, setTool] = useState<'brush' | 'bucket'>('brush');
  const [ink, setInk] = useState(100);
  const [refill, setRefill] = useState(5000);

  // Simulate ink drain on draw progress
  const onDrawProgress = () => setInk(v => Math.max(0, v - 0.1));

  // Simulate timed refill
  React.useEffect(() => {
    const iv = setInterval(() => setInk(v => Math.min(100, v + 0.5)), 500);
    const tick = setInterval(() => setRefill(t => (t <= 1000 ? 5000 : t - 500)), 500);
    return () => { clearInterval(iv); clearInterval(tick); };
  }, []);

  const handleDraw = (data: DrawData) => {
    setPaths(p => [...p, data]);
    setRedo([]);
  };

  const canUndo = paths.length > 0;
  const canRedo = redoStack.length > 0;

  const undo = () => {
    if (!canUndo) return;
    setPaths(p => {
      const next = [...p];
      const popped = next.pop();
      if (popped) setRedo(r => [popped, ...r]);
      return next;
    });
  };

  const redo = () => {
    if (!canRedo) return;
    setRedo(r => {
      const [first, ...rest] = r;
      if (first) setPaths(p => [...p, first]);
      return rest;
    });
  };

  const canvasKey = useMemo(() => 'classic-canvas', []);

  return (
    <div className="flex flex-col h-full bg-zinc-900 text-white">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <NavigationToolbar onBack={() => setCurrentPage('explore')} onUndo={undo} onRedo={redo} canUndo={canUndo} canRedo={canRedo} />
        <div className="flex items-center gap-3">
          <InkMeter inkLevel={ink} timeToRefillMs={refill} color={color} />
        </div>
      </div>

      <div className="flex-1 grid grid-rows-[1fr_auto]">
        <div className="relative">
          <DrawingCanvas
            key={canvasKey}
            paths={paths}
            color={color}
            strokeWeight={stroke}
            tool={tool}
            onDraw={handleDraw}
            onDrawProgress={onDrawProgress}
          />
        </div>
        <div className="px-4 py-3 border-t border-zinc-800 bg-zinc-900">
          <Toolbar color={color} setColor={setColor} strokeWeight={stroke} setStrokeWeight={setStroke} tool={tool} setTool={setTool} />
        </div>
      </div>
    </div>
  );
};

export default ClassicCanvasPage;
