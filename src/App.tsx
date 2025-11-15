import React, { useState } from 'react';
import MainLayout from './components/layout/MainLayout';
import ExplorePage from './components/pages/ExplorePage';
import CommunityPage from './components/pages/CommunityPage';
import LeaderboardsPage from './components/pages/LeaderboardsPage';
import NavigatePage from './components/pages/NavigatePage';
import GalleryPage from './components/pages/GalleryPage';
import NotificationsPage from './components/pages/NotificationsPage';
import ClassicCanvasPage from './components/pages/ClassicCanvasPage';

export type Page = 'explore' | 'community' | 'leaderboards' | 'navigate' | 'gallery' | 'notifications' | 'classic-canvas';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('explore');

  const renderPage = () => {
    switch (currentPage) {
      case 'explore':
        return <ExplorePage setCurrentPage={setCurrentPage} />;
      case 'community':
        return <CommunityPage />;
      case 'leaderboards':
        return <LeaderboardsPage />;
      case 'navigate':
        return <NavigatePage />;
      case 'gallery':
        return <GalleryPage />;
      case 'notifications':
        return <NotificationsPage />;
      case 'classic-canvas':
        return <ClassicCanvasPage setCurrentPage={setCurrentPage} />;
      default:
        return <ExplorePage setCurrentPage={setCurrentPage} />;
    }
  };

  return (
    <div className="bg-black min-h-screen text-white">
      <MainLayout currentPage={currentPage} setCurrentPage={setCurrentPage}>
        {renderPage()}
      </MainLayout>
    </div>
  );
};

export default App;
