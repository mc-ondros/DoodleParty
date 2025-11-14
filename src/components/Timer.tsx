import React from 'react';

interface TimerProps {
  timeLeft: number;
}

export const Timer: React.FC<TimerProps> = ({ timeLeft }) => {
  // Game timer display
  return (
    <div className="timer">
      <span className="time">{timeLeft}s</span>
    </div>
  );
};
