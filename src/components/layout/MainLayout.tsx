
import React, { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './Sidebar';
import Navigation from './Navigation';
import RightSidebar from './RightSidebar';
import type { Page } from '../../App';
import { MenuIcon, XIcon } from '../../constants';

interface MainLayoutProps {
  children: React.ReactNode;
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
  settingsCategory?: string;
  setSettingsCategory?: (cat: string) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, currentPage, setCurrentPage, settingsCategory, setSettingsCategory }) => {
  const [navWidth, setNavWidth] = useState<number>(256); // default w-64
  const [navOpen, setNavOpen] = useState<boolean>(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState<boolean>(false);
  const isResizing = useRef(false);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    const next = Math.min(420, Math.max(220, navWidth + e.movementX));
    setNavWidth(next);
  }, [navWidth]);

  const onMouseUp = useCallback(() => {
    isResizing.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const startResize = () => {
    isResizing.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  return (
    <div className="flex h-screen relative">
      {/* Mobile Menu Button */}
      <button 
        className="lg:hidden fixed top-4 left-4 z-50 bg-zinc-800 p-2 rounded-md text-white hover:bg-zinc-700"
        onClick={() => setNavOpen(!navOpen)}
      >
        {navOpen ? <XIcon className="w-6 h-6" /> : <MenuIcon className="w-6 h-6" />}
      </button>

      {/* Sidebar - Hidden on mobile, shown as bottom nav */}
      <div className="hidden lg:flex">
        <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      </div>

      {/* Navigation - Overlay on mobile */}
      <div 
        className={`
          fixed lg:relative
          inset-y-0 left-0
          transform lg:transform-none
          ${navOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          transition-transform duration-300 ease-in-out
          z-40 lg:z-auto
          h-full
        `}
        style={{ width: navWidth, minWidth: 220, maxWidth: 420 }}
      >
        <Navigation 
          currentPage={currentPage} 
          setCurrentPage={setCurrentPage} 
          navWidth={navWidth} 
          setNavWidth={setNavWidth} 
          settingsCategory={settingsCategory} 
          setSettingsCategory={setSettingsCategory}
          onNavigate={() => setNavOpen(false)}
        />
        <div
          className="hidden lg:block absolute top-0 right-0 h-full w-1 cursor-col-resize"
          onMouseDown={startResize}
          style={{ background: 'transparent' }}
        />
      </div>

      {/* Overlay for mobile menu */}
      {navOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setNavOpen(false)}
        />
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto lg:pb-0 pb-20">
        {children}
      </main>

      {/* Right Sidebar - Hidden on mobile */}
      <div className="hidden lg:flex h-full">
        <RightSidebar currentPage={currentPage}/>
      </div>

      {/* Mobile Bottom Navigation (Sidebar replacement) */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-30">
        <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} isMobile={true} />
      </div>
    </div>
  );
};

export default MainLayout;
