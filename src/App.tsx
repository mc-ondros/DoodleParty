import React, { useState } from 'react';
import MainLayout from './components/layout/MainLayout';
import ExplorePage from './components/pages/ExplorePage';
import CommunityPage from './components/pages/CommunityPage';
import LeaderboardsPage from './components/pages/LeaderboardsPage';
import NavigatePage from './components/pages/NavigatePage';
import GalleryPage from './components/pages/GalleryPage';
import NotificationsPage from './components/pages/NotificationsPage';
import ClassicCanvasPage from './components/pages/ClassicCanvasPage';
import SettingsPage from './components/pages/SettingsPage';
import { SharedCanvasProvider } from './context/SharedCanvasContext';

export type Page = 'explore' | 'community' | 'leaderboards' | 'navigate' | 'gallery' | 'notifications' | 'classic-canvas' | 'settings';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('explore');
  const [settingsCategory, setSettingsCategory] = useState<string>('profile');

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
      case 'settings':
        return <SettingsPage activeCategory={settingsCategory} setActiveCategory={setSettingsCategory} />;
      default:
        return <ExplorePage setCurrentPage={setCurrentPage} />;
    }
  };

  return (
    <SharedCanvasProvider>
      <div className="bg-zinc-900 min-h-screen text-white">
        <MainLayout currentPage={currentPage} setCurrentPage={setCurrentPage} settingsCategory={settingsCategory} setSettingsCategory={setSettingsCategory}>
          {renderPage()}
        </MainLayout>
      </div>
    </SharedCanvasProvider>
  );
};

export default App;
