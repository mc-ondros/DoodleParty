/**
 * Drawer View Component
 *
 * Side drawer with color picker, brush controls, and game information.
 * Displays ink meter and timer.
 * 
 * Related:
 * - src/components/InkMeter.tsx (ink display)
 * - src/components/Timer.tsx (timer display)
 * 
 * Exports:
 * - DrawerView (functional component)
 */

import React, { useState, useEffect } from 'react';

interface DrawerViewProps {
  onColorChange?: (color: string) => void;
  onBrushSizeChange?: (size: number) => void;
  inkLevel?: number;
  timeRemaining?: string;
}

const COLOR_PRESETS = [
  { name: 'Blue', color: '#3b82f6' },
  { name: 'Red', color: '#ef4444' },
  { name: 'Green', color: '#10b981' },
  { name: 'Yellow', color: '#f59e0b' },
  { name: 'Purple', color: '#8b5cf6' },
  { name: 'Black', color: '#000000' },
];

export const DrawerView: React.FC<DrawerViewProps> = ({
  onColorChange,
  onBrushSizeChange,
  inkLevel = 100,
  timeRemaining = '00:00',
}) => {
  const [activeColor, setActiveColor] = useState(COLOR_PRESETS[0].color);
  const [brushSize, setBrushSize] = useState(8);

  useEffect(() => {
    // Set initial color
    onColorChange?.(activeColor);
  }, []);

  const handleColorClick = (color: string) => {
    setActiveColor(color);
    onColorChange?.(color);
    document.documentElement.style.setProperty('--selected-color', color);
  };

  const handleBrushSizeChange = (size: number) => {
    setBrushSize(size);
    onBrushSizeChange?.(size);
  };

  return (
    <div className="drawer-view">
      <aside className="drawer">
        <div className="drawer-section">
          <h3>Colors</h3>
          <div className="color-picker">
            {COLOR_PRESETS.map(({ name, color }) => (
              <button
                key={color}
                className={`color-dot ${activeColor === color ? 'active' : ''}`}
                data-color={color}
                style={{ backgroundColor: color }}
                onClick={() => handleColorClick(color)}
                aria-label={`Select ${name}`}
              />
            ))}
          </div>
        </div>

        <div className="drawer-section">
          <h3>Brush Size</h3>
          <input
            type="range"
            id="brushSize"
            min="2"
            max="20"
            value={brushSize}
            onChange={(e) => handleBrushSizeChange(Number(e.target.value))}
          />
          <span id="brushSizeValue">{brushSize}px</span>
        </div>

        <div className="drawer-section">
          <h3>Ink Level</h3>
          <div className="ink-meter">
            <div className="ink-meter-bar">
              <div
                id="inkFill"
                className="ink-meter-fill"
                style={{ width: `${inkLevel}%` }}
              />
            </div>
            <span id="inkValue">{Math.round(inkLevel)}%</span>
          </div>
        </div>

        <div className="drawer-section">
          <h3>Time</h3>
          <div id="timer" className="timer">
            {timeRemaining}
          </div>
        </div>
      </aside>
    </div>
  );
};
