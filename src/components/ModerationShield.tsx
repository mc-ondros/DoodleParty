import React from 'react';

interface ModerationShieldProps {
  status: 'safe' | 'warning' | 'blocked';
  message?: string;
}

export const ModerationShield: React.FC<ModerationShieldProps> = ({ status, message }) => {
  // Content moderation feedback UI
  return (
    <div className={`moderation-shield ${status}`}>
      {message && <p>{message}</p>}
    </div>
  );
};
