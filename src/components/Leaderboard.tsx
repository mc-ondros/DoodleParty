import React from 'react';

interface LeaderboardEntry {
  rank: number;
  name: string;
  score: number;
}

interface LeaderboardProps {
  entries: LeaderboardEntry[];
}

export const Leaderboard: React.FC<LeaderboardProps> = ({ entries }) => {
  // Real-time leaderboard display
  return (
    <div className="leaderboard">
      <h2>Leaderboard</h2>
      <ul>
        {entries.map((entry) => (
          <li key={entry.rank}>
            {entry.rank}. {entry.name}: {entry.score}
          </li>
        ))}
      </ul>
    </div>
  );
};
