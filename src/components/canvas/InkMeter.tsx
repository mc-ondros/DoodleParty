import React from 'react';

interface Props {
  inkLevel: number; // 0-100
  timeToRefillMs: number;
  color: string;
}

const InkMeter: React.FC<Props> = ({ inkLevel, timeToRefillMs, color }) => {
  const circumference = 2 * Math.PI * 18;
  const progress = Math.max(0, Math.min(1, inkLevel / 100));
  const dash = circumference * progress;

  return (
    <div className="flex items-center gap-2 bg-zinc-900/70 border border-zinc-800 rounded-xl px-3 py-2">
      <svg className="w-10 h-10 -rotate-90" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="none" stroke="#3f3f46" strokeWidth="4" />
        <circle cx="20" cy="20" r="18" fill="none" stroke={color} strokeWidth="4" strokeDasharray={circumference} strokeDashoffset={circumference - dash} strokeLinecap="round" />
      </svg>
      <div>
        <div className="text-xs text-zinc-400">Ink</div>
        <div className="text-sm font-semibold">{Math.round(inkLevel)}% · {Math.ceil(timeToRefillMs/1000)}s</div>
      </div>
    </div>
  );
};

export default InkMeter;
