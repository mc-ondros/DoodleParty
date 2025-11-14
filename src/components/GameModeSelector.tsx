import React from 'react';

interface GameModeSelectorProps {
  onModeSelect: (mode: string) => void;
}

export const GameModeSelector: React.FC<GameModeSelectorProps> = ({ onModeSelect }) => {
  // Game mode selection interface
  return (
    <div className="game-mode-selector">
      <div className="modes">
        {/* Game mode buttons */}
      </div>
    </div>
  );
};
