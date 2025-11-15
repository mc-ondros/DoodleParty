import React, { useEffect, useState } from 'react';
import type { Page } from '../../App';
import { SearchIcon, CompassIcon, UsersIcon, TrophyIcon, SettingsIcon, LibraryIcon, NotificationsIcon, StoreIcon, PlayIcon, ClockIcon, UserIcon, PaletteIcon, ShieldIcon, CreditCardIcon } from '../../constants';
import ChatPanel from '../chat/ChatPanel';

interface NavItemProps {
  icon: React.ElementType;
  label: string;
  active?: boolean;
  onClick?: () => void;
}

const NavItem: React.FC<NavItemProps> = ({ icon: Icon, label, active = false, onClick }) => (
  <button onClick={onClick} className={`flex items-center w-full px-3 py-2 rounded-md text-left transition-colors duration-200 ${active ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:bg-zinc-700/50 hover:text-white'}`}>
    <Icon className="w-5 h-5 mr-3 flex-shrink-0" />
    <span className="text-sm md:text-base">{label}</span>
  </button>
);

interface NavigationProps {
    currentPage: Page;
    setCurrentPage: (page: Page) => void;
    navWidth?: number;
    setNavWidth?: (n: number) => void;
    settingsCategory?: string;
    setSettingsCategory?: (cat: string) => void;
    onNavigate?: () => void;
}

const Navigation: React.FC<NavigationProps> = ({ currentPage, setCurrentPage, navWidth, settingsCategory, setSettingsCategory, onNavigate }) => {
  const navItems: { icon: React.ElementType; label: string; page: Page }[] = [
    { icon: CompassIcon, label: 'Explore', page: 'explore' },
    { icon: UsersIcon, label: 'Community', page: 'community' },
    { icon: TrophyIcon, label: 'Leaderboards', page: 'leaderboards' },
  ];

  const displayedItems = navItems.filter(item => item.page === currentPage);

  // Track active section while scrolling on Explore
  const [activeSection, setActiveSection] = useState<string>('explore');

  useEffect(() => {
    if (currentPage !== 'explore') return;
    const ids = ['explore', 'gallery', 'pricing', 'game-modes', 'classic-canvas', 'speed-sketch'];
    const els = ids
      .map((id) => ({ id, el: document.getElementById(id) }))
      .filter((x): x is { id: string; el: Element } => Boolean(x.el));

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) {
          const id = visible[0].target.getAttribute('id');
          if (id) setActiveSection(id);
        }
      },
      { root: null, rootMargin: '0px 0px -60% 0px', threshold: [0, 0.25, 0.5, 1] }
    );

    els.forEach(({ el }) => observer.observe(el));
    return () => observer.disconnect();
  }, [currentPage]);

  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="bg-zinc-900 flex flex-col p-4 border-r border-zinc-800 relative h-full" style={{ width: navWidth ?? 256 }}>
      {currentPage !== 'classic-canvas' && (
        <div className="relative mb-4">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
          <input type="text" placeholder="Search..." className="w-full bg-zinc-800 border border-zinc-700 rounded-md pl-10 pr-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-green-500" />
        </div>
      )}

      <nav className="flex-grow overflow-hidden min-h-0">
        {currentPage === 'classic-canvas' ? (
          <div className="flex flex-col h-full min-h-0">
            <div>
              <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Navigate</h2>
              <div className="space-y-2">
                <NavItem icon={CompassIcon} label="Explore" onClick={() => { setCurrentPage('explore'); onNavigate?.(); }} />
                <NavItem icon={LibraryIcon} label="Gallery" onClick={() => { setCurrentPage('gallery'); onNavigate?.(); }} />
              </div>
              <div className="mt-6">
                <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Game Modes</h2>
                <div className="space-y-2">
                  <NavItem icon={PlayIcon} label="Classic Canvas" active onClick={() => { onNavigate?.(); }} />
                  <NavItem icon={ClockIcon} label="Speed Sketch" onClick={() => { onNavigate?.(); }} />
                </div>
              </div>
            </div>
            <div className="flex-1 min-h-0 mt-6 mb-3 border-t border-zinc-800 pt-4 pr-2">
              <ChatPanel />
            </div>
          </div>
        ) : currentPage === 'settings' ? (
          <>
            <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Settings</h2>
            <div className="space-y-2">
              <NavItem icon={UserIcon} label="Profile" active={settingsCategory === 'profile'} onClick={() => setSettingsCategory?.('profile')} />
              <NavItem icon={NotificationsIcon} label="Notifications" active={settingsCategory === 'notifications'} onClick={() => setSettingsCategory?.('notifications')} />
              <NavItem icon={PaletteIcon} label="Appearance" active={settingsCategory === 'appearance'} onClick={() => setSettingsCategory?.('appearance')} />
              <NavItem icon={ShieldIcon} label="Privacy & Security" active={settingsCategory === 'privacy'} onClick={() => setSettingsCategory?.('privacy')} />
              <NavItem icon={CreditCardIcon} label="Billing" active={settingsCategory === 'billing'} onClick={() => setSettingsCategory?.('billing')} />
              <NavItem icon={ShieldIcon} label="Admin Dashboard" active={settingsCategory === 'admin'} onClick={() => setSettingsCategory?.('admin')} />
            </div>
          </>
        ) : (
          <>
            <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Navigate</h2>
            <div className="space-y-2">
              {displayedItems.map((item) => (
                <NavItem
                  key={item.page}
                  icon={item.icon}
                  label={item.label}
                  active={item.page === 'explore' ? activeSection === 'explore' : currentPage === item.page}
                  onClick={item.page === 'explore' ? () => scrollToId('explore') : () => setCurrentPage(item.page)}
                />
              ))}
            </div>

            {currentPage === 'explore' && (
              <>
                <div className="mt-2 space-y-2">
                  <NavItem icon={LibraryIcon} label="Gallery" active={activeSection==='gallery'} onClick={() => scrollToId('gallery')} />
                  <NavItem icon={StoreIcon} label="Pricing" active={activeSection==='pricing'} onClick={() => scrollToId('pricing')} />
                </div>

                <div className="mt-6">
                  <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Game Modes</h2>
                  <div className="space-y-2">
                    <NavItem icon={PlayIcon} label="Classic Canvas" active={activeSection==='classic-canvas'} onClick={() => scrollToId('classic-canvas')} />
                    <NavItem icon={ClockIcon} label="Speed Sketch" active={activeSection==='speed-sketch'} onClick={() => scrollToId('speed-sketch')} />
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </nav>

      <div className="mt-auto">
        <div className="flex items-center p-2 rounded-md hover:bg-zinc-800 transition-colors duration-200">
          <img src="https://picsum.photos/seed/enrique/40/40" alt="User" className="w-10 h-10 rounded-full mr-3" />
          <div className="flex-grow">
            <p className="font-semibold text-white">EnriqueP</p>
            <p className="text-xs text-green-400 bg-green-900/50 px-2 py-0.5 rounded-full inline-block">LEVEL 1</p>
          </div>
          <button onClick={() => setCurrentPage('settings')} className="hover:text-white transition-colors">
            <SettingsIcon className="w-5 h-5 text-zinc-400" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navigation;