import React from 'react';
import { UsersIcon, TrophyIcon, CheckCircleIcon, BrushIcon } from '../../constants';

interface Notification {
  id: number;
  type: 'achievement' | 'social' | 'activity' | 'system';
  icon: React.ElementType;
  title: string;
  message: string;
  time: string;
  read: boolean;
}

const NotificationsPage: React.FC = () => {
  const notifications: Notification[] = [
    {
      id: 1,
      type: 'achievement',
      icon: TrophyIcon,
      title: 'New Achievement Unlocked!',
      message: 'You earned the "Collaborative Creator" badge for contributing to 5 canvases.',
      time: '2 hours ago',
      read: false
    },
    {
      id: 2,
      type: 'social',
      icon: UsersIcon,
      title: 'New Follower',
      message: 'NovaSketch started following you.',
      time: '5 hours ago',
      read: false
    },
    {
      id: 3,
      type: 'activity',
      icon: BrushIcon,
      title: 'Canvas Session Ending Soon',
      message: 'The "Cyberpunk Nature" canvas session ends in 2 days. Make your final contributions!',
      time: '1 day ago',
      read: true
    },
    {
      id: 4,
      type: 'system',
      icon: CheckCircleIcon,
      title: 'Weekly Leaderboard Updated',
      message: 'You ranked #42 this week! Keep drawing to climb higher.',
      time: '2 days ago',
      read: true
    },
    {
      id: 5,
      type: 'social',
      icon: UsersIcon,
      title: 'New Community Event',
      message: 'Join the "Weekly Sketch Jam" starting this Friday at 7 PM.',
      time: '3 days ago',
      read: true
    },
    {
      id: 6,
      type: 'achievement',
      icon: TrophyIcon,
      title: 'Streak Milestone!',
      message: 'You have maintained a 7-day drawing streak. Keep it up!',
      time: '4 days ago',
      read: true
    },
    {
      id: 7,
      type: 'activity',
      icon: BrushIcon,
      title: 'Gallery Updated',
      message: 'The "Ocean Dreams" timelapse is now available in the gallery.',
      time: '5 days ago',
      read: true
    }
  ];

  const getIconColor = (type: string) => {
    switch (type) {
      case 'achievement':
        return 'text-yellow-400';
      case 'social':
        return 'text-blue-400';
      case 'activity':
        return 'text-green-400';
      case 'system':
        return 'text-purple-400';
      default:
        return 'text-zinc-400';
    }
  };

  return (
    <div className="p-10 bg-black text-white">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold">Notifications</h1>
            <p className="text-zinc-400 mt-2">Stay updated with your activity and achievements.</p>
          </div>
          <button className="text-sm text-green-400 hover:text-green-300 transition-colors">
            Mark all as read
          </button>
        </div>

        <div className="mt-10 space-y-3">
          {notifications.map((notification) => {
            const Icon = notification.icon;
            return (
              <div
                key={notification.id}
                className={`p-5 rounded-lg border transition-all duration-200 ${
                  notification.read
                    ? 'border-zinc-800 bg-zinc-900/40'
                    : 'border-green-500/50 bg-zinc-900/60'
                } hover:border-green-500 hover:bg-zinc-900/80`}
              >
                <div className="flex gap-4">
                  <div className={`flex-shrink-0 ${getIconColor(notification.type)}`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex-grow">
                    <div className="flex items-start justify-between">
                      <h3 className="font-semibold text-white">{notification.title}</h3>
                      {!notification.read && (
                        <div className="w-2 h-2 bg-green-400 rounded-full flex-shrink-0 mt-2" />
                      )}
                    </div>
                    <p className="text-sm text-zinc-400 mt-1">{notification.message}</p>
                    <p className="text-xs text-zinc-500 mt-2">{notification.time}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {notifications.length === 0 && (
          <div className="mt-20 text-center">
            <div className="text-zinc-600 mb-4">
              <CheckCircleIcon className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-xl font-semibold text-zinc-400">You're all caught up!</h3>
            <p className="text-sm text-zinc-500 mt-2">No new notifications at the moment.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NotificationsPage;
