import React from 'react';

type Tool = 'brush' | 'bucket';

interface ToolbarProps {
  color: string;
  setColor: (c: string) => void;
  strokeWeight: number;
  setStrokeWeight: (n: number) => void;
  tool: Tool;
  setTool: (t: Tool) => void;
}

const palette = ['#22c55e','#16a34a','#86efac','#ef4444','#f97316','#eab308','#06b6d4','#0ea5e9','#8b5cf6','#a855f7','#ec4899','#ffffff','#000000'];

const Toolbar: React.FC<ToolbarProps> = ({ color, setColor, strokeWeight, setStrokeWeight, tool, setTool }) => {
  return (
    <div className="flex items-center gap-4 bg-zinc-900/70 border border-zinc-800 rounded-xl px-4 py-3">
      <div className="flex items-center gap-2">
        {palette.map(c => {
          const active = color === c;
          return (
            <button
              key={c}
              onClick={() => setColor(c)}
              aria-label={`Pick color ${c}`}
              className={`w-7 h-7 rounded-full transition-all duration-200 cursor-pointer flex-shrink-0 relative overflow-hidden ${
                active
                  ? 'ring-2 ring-white/60 ring-offset-1 ring-offset-zinc-900 scale-110'
                  : 'hover:scale-110'
              }`}
              style={{
                boxShadow: active
                  ? `0 4px 12px ${c}40, 0 0 0 1px rgba(255,255,255,0.2), inset 0 1px 0 rgba(255,255,255,0.35), inset -2px -2px 4px rgba(0,0,0,0.2)`
                  : `inset -2px -2px 4px rgba(0,0,0,0.25)`,
                background: `radial-gradient(circle at 30% 30%, rgba(255,255,255,0.28), transparent 60%), ${c}`,
              }}
            >
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background:
                    'linear-gradient(135deg, rgba(255,255,255,0.4) 0%, transparent 50%, rgba(0,0,0,0.12) 100%)',
                }}
              />
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-white">Size</span>
        <input
          type="range"
          min={1}
          max={50}
          value={strokeWeight}
          onChange={e=>setStrokeWeight(Number(e.target.value))}
          className="w-32 dp-range"
          style={{ color }}
        />
        <div className="relative w-6 h-6 rounded-full bg-transparent overflow-hidden">
          <div className="absolute inset-0 m-auto rounded-full" style={{ background: color, width: '80%', height: '80%', transform: `scale(${Math.min(1, Math.max(0.2, strokeWeight/20))})` }} />
        </div>
      </div>
      <style>{`
        .dp-range{ -webkit-appearance:none; appearance:none; background:transparent; }
        .dp-range:focus{ outline:none; }
        .dp-range::-webkit-slider-runnable-track{ background:#3f3f46; height:8px; border-radius:9999px; }
        .dp-range::-moz-range-track{ background:#3f3f46; height:8px; border-radius:9999px; }
        .dp-range::-webkit-slider-thumb{ -webkit-appearance:none; appearance:none; width:16px; height:16px; border-radius:9999px; background: currentColor; border:2px solid #18181b; margin-top:-4px; }
        .dp-range::-moz-range-thumb{ width:16px; height:16px; border-radius:9999px; background: currentColor; border:2px solid #18181b; }
      `}</style>
      <div className="flex items-center gap-2">
        <button className={`px-3 py-1 rounded-md text-sm border ${tool==='brush'?'text-black':'text-white'}`} style={tool==='brush'?{ background: color, borderColor: color }:{ background: '#27272a', borderColor: '#3f3f46' }} onClick={()=>setTool('brush')}>Brush</button>
        <button className={`px-3 py-1 rounded-md text-sm border ${tool==='bucket'?'text-black':'text-white'}`} style={tool==='bucket'?{ background: color, borderColor: color }:{ background: '#27272a', borderColor: '#3f3f46' }} onClick={()=>setTool('bucket')}>Bucket</button>
      </div>
    </div>
  );
};

export default Toolbar;
