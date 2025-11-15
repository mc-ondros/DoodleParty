/**
 * Game Mode Selector Component
 *
 * Provides UI for selecting game modes (freeDraw, challenge, collaborative).
 * 
 * Related:
 * - src/hooks/useGameMode.tsx (game mode logic)
 * - src/services/gameService.ts (game logic)
 * 
 * Exports:
 * - GameModeSelector (functional component)
 */

import React, { useState } from 'react';

interface GameModeSelectorProps {
  onModeSelect: (mode: string) => void;
}

type GameMode = 'freeDraw' | 'challenge' | 'collaborative';

const GAME_MODES: { id: GameMode; label: string }[] = [
  { id: 'freeDraw', label: 'Free Draw' },
  { id: 'challenge', label: 'Challenge' },
  { id: 'collaborative', label: 'Collaborative' },
];

export const GameModeSelector: React.FC<GameModeSelectorProps> = ({ onModeSelect }) => {
  const [selectedMode, setSelectedMode] = useState<GameMode>('freeDraw');

  const handleModeClick = (mode: GameMode) => {
    setSelectedMode(mode);
    onModeSelect(mode);
  };

  return (
    <div className="game-mode-selector">
      <h3>Game Mode</h3>
      <div className="mode-buttons">
        {GAME_MODES.map(({ id, label }) => (
          <button
            key={id}
            className={`mode-button ${selectedMode === id ? 'active' : ''}`}
            onClick={() => handleModeClick(id)}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};
  );
};
