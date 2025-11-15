import React from 'react';
import { ModerationStatus } from '../../types';

const bg: Record<ModerationStatus, string> = {
  PENDING: 'bg-blue-500/10',
  SAFE: 'bg-green-500/10',
  UNSAFE: 'bg-red-500/10',
  ERROR: 'bg-yellow-500/10',
};

const txt: Record<ModerationStatus, string> = {
  PENDING: 'text-blue-300',
  SAFE: 'text-green-300',
  UNSAFE: 'text-red-300',
  ERROR: 'text-yellow-300',
};

interface Props { status: ModerationStatus; }

const ModerationShield: React.FC<Props> = ({ status }) => (
  <div className={`absolute inset-0 flex items-center justify-center ${bg[status]} backdrop-blur-sm`}>
    <div className={`text-sm font-semibold ${txt[status]}`}>
      {status === 'PENDING' && 'Checking your doodle...'}
      {status === 'SAFE' && 'Looks good!'}
      {status === 'UNSAFE' && 'Content not allowed.'}
      {status === 'ERROR' && 'Moderation error. Try again.'}
    </div>
  </div>
);

export default ModerationShield;
