export interface Point {
  x: number;
  y: number;
}

export interface DrawingData {
  points: Point[];
  color: string;
  thickness: number;
  timestamp: number;
}

export interface GameState {
  mode: 'classic' | 'survival' | 'relay' | null;
  isActive: boolean;
  score: number;
  timeRemaining: number;
}

export interface ClassificationResult {
  label: string;
  confidence: number;
}

export interface ModerationResult {
  status: 'safe' | 'warning' | 'blocked';
  confidence: number;
  reason?: string;
}

export interface LeaderboardEntry {
  rank: number;
  name: string;
  score: number;
  timestamp: number;
}
