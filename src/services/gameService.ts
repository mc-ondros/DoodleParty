class GameService {
  private currentMode: string | null = null;
  private score = 0;
  private timer: NodeJS.Timeout | null = null;

  startGame(mode: string) {
    this.currentMode = mode;
    this.score = 0;
    this.startTimer();
  }

  private startTimer() {
    this.timer = setInterval(() => {
      // Timer logic
    }, 1000);
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
