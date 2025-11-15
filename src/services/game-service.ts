/**
 * Game Service
 *
 * Manages game state, scoring, and timing functionality.
 * Handles game mode selection, score tracking, and timer management.
 *
 * Related:
 * - src/hooks/useGameMode.tsx (React hook for game modes)
 * - src/components/GameModeSelector.tsx (UI for selecting game modes)
 *
 * Exports:
 * - GameService (class), gameService (instance)
 */

class GameService {
  private currentMode: string | null = null;
  private score = 0;
  private timer: NodeJS.Timeout | null = null;

  // Constants
  private static readonly TIMER_INTERVAL_MS = 1000;

  startGame(mode: string) {
    this.currentMode = mode;
    this.score = 0;
    this.startTimer();
  }

  private startTimer() {
    this.timer = setInterval(() => {
      // Timer logic
    }, GameService.TIMER_INTERVAL_MS);
  }

  stopGame() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.currentMode = null;
  }

  addScore(points: number) {
    this.score += points;
  }

  getScore() {
    return this.score;
  }

  getCurrentMode() {
    return this.currentMode;
  }

  resetGame() {
    this.stopGame();
    this.score = 0;
  }
}

export const gameService = new GameService();
