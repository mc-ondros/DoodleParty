import React from 'react';

interface InkMeterProps {
  level: number;
}

export const InkMeter: React.FC<InkMeterProps> = ({ level }) => {
  // Ink level indicator
  return (
    <div className="ink-meter">
      <div className="ink-level" style={{ width: `${level}%` }}></div>
    </div>
  );
};
