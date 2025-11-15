
import React, { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './Sidebar';
import Navigation from './Navigation';
import RightSidebar from './RightSidebar';
import type { Page } from '../../App';

interface MainLayoutProps {
  children: React.ReactNode;
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, currentPage, setCurrentPage }) => {
  const [navWidth, setNavWidth] = useState<number>(256); // default w-64
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
    <div className="flex h-screen">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <div className="relative h-full" style={{ width: navWidth, minWidth: 220, maxWidth: 420 }}>
        <Navigation currentPage={currentPage} setCurrentPage={setCurrentPage} navWidth={navWidth} setNavWidth={setNavWidth} />
        <div
          className="absolute top-0 right-0 h-full w-1 cursor-col-resize"
          onMouseDown={startResize}
          style={{ background: 'transparent' }}
        />
      </div>
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
      <RightSidebar currentPage={currentPage}/>
    </div>
  );
};

export default MainLayout;
