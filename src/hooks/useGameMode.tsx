import { useState, useCallback } from 'react';

export type GameMode = 'classic' | 'survival' | 'relay';

export const useGameMode = () => {
  const [mode, setMode] = useState<GameMode | null>(null);
  const [isActive, setIsActive] = useState(false);

  const selectMode = useCallback((selectedMode: GameMode) => {
    setMode(selectedMode);
    setIsActive(true);
  }, []);

  const endGame = useCallback(() => {
    setIsActive(false);
  }, []);

  return {
    mode,
    isActive,
    selectMode,
    endGame,
  };
};
