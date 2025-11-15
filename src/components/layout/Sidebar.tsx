
import React from 'react';
import type { Page } from '../../App';
import { DoodlePartyLogo, CompassIcon, UsersIcon, TrophyIcon, NotificationsIcon } from '../../constants';

interface SidebarProps {
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
  isMobile?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ currentPage, setCurrentPage, isMobile = false }) => {
  const navItems: { icon: React.ElementType; label: string; page: Page }[] = [
    { icon: CompassIcon, label: 'Explore', page: 'explore' },
    { icon: UsersIcon, label: 'Community', page: 'community' },
    { icon: TrophyIcon, label: 'Leaderboards', page: 'leaderboards' },
    { icon: NotificationsIcon, label: 'Notifications', page: 'notifications' },
  ];
  
  const bottomIcons = [
    { avatar: 'https://picsum.photos/seed/user1/32/32' },
    { avatar: 'https://picsum.photos/seed/user2/32/32' },
    { avatar: 'https://picsum.photos/seed/user3/32/32' },
  ];

  // Mobile horizontal layout
  if (isMobile) {
    return (
      <div className="w-full bg-black flex items-center justify-around py-3 border-t border-zinc-800">
        {navItems.map((item) => (
          <button
            key={item.label}
            onClick={() => setCurrentPage(item.page)}
            className={`flex flex-col items-center gap-1 p-2 transition-colors duration-200 ${
              currentPage === item.page ? 'text-green-500' : 'text-zinc-400'
            }`}
          >
            <item.icon className="h-6 w-6" />
            <span className="text-xs">{item.label}</span>
          </button>
        ))}
      </div>
    );
  }

  // Desktop vertical layout
  return (
    <div className="w-20 bg-black flex flex-col items-center py-4 border-r border-zinc-800">
      <div className="p-2 mb-8">
        <DoodlePartyLogo className="h-8 w-8 text-white" />
      </div>
      <nav className="flex flex-col items-center gap-6 flex-grow">
        {navItems.map((item) => (
          <div key={item.label} className="group relative">
            <button
              onClick={() => setCurrentPage(item.page)}
              className={`p-3 transition-colors duration-200 ${currentPage === item.page ? 'bg-green-500 text-black' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}
              style={{ borderRadius: '20%' }}
            >
              <item.icon className="h-6 w-6" />
            </button>
            <span className="absolute left-full ml-4 top-1/2 -translate-y-1/2 bg-zinc-800 text-white text-sm px-2 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              {item.label}
            </span>
          </div>
        ))}
      </nav>
      <div className="flex flex-col items-center gap-4">
        {bottomIcons.map((item, index) => (
          <img key={index} src={item.avatar} alt="User avatar" className="w-10 h-10 rounded-full" />
        ))}
        <div className="w-10 h-10 bg-green-500 flex items-center justify-center font-bold text-black" style={{ borderRadius: '20%' }}>
          G
        </div>
      </div>
    </div>
  );
};

export default Sidebar;