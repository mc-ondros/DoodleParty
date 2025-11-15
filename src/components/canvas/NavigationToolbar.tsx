import React from 'react';
import { ArrowLeft, RotateCcw, RotateCw } from 'lucide-react';

interface Props {
  onBack: () => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

const NavigationToolbar: React.FC<Props> = ({ onBack, onUndo, onRedo, canUndo, canRedo }) => {
  const btn = 'w-10 h-10 rounded-lg border border-zinc-700 bg-zinc-900 text-white hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center';
  return (
    <div className="flex items-center gap-2 bg-zinc-900/70 border border-zinc-800 rounded-xl px-3 py-2">
      <button className={btn} onClick={onBack} title="Back">
        <ArrowLeft className="w-5 h-5" />
      </button>
      <div className="w-px h-6 bg-zinc-800" />
      <button className={btn} onClick={onUndo} disabled={!canUndo} title="Undo">
        <RotateCcw className="w-5 h-5" />
      </button>
      <button className={btn} onClick={onRedo} disabled={!canRedo} title="Redo">
        <RotateCw className="w-5 h-5" />
      </button>
    </div>
  );
};

export default NavigationToolbar;
