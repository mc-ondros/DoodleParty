import React from 'react';
import { CompassIcon, UsersIcon, TrophyIcon, LibraryIcon, NotificationsIcon, StoreIcon, PlayIcon, ClockIcon, BrushIcon } from '../../constants';

interface QuickLink {
  icon: React.ElementType;
  title: string;
  description: string;
  color: string;
}

const NavigatePage: React.FC = () => {
  const mainSections: QuickLink[] = [
    {
      icon: CompassIcon,
      title: 'Explore',
      description: 'Join the live collaborative canvas and start creating',
      color: 'bg-green-500/10 border-green-500/50 hover:border-green-500'
    },
    {
      icon: UsersIcon,
      title: 'Community',
      description: 'Connect with fellow artists and join events',
      color: 'bg-blue-500/10 border-blue-500/50 hover:border-blue-500'
    },
    {
      icon: TrophyIcon,
      title: 'Leaderboards',
      description: 'See top creators and trending communities',
      color: 'bg-yellow-500/10 border-yellow-500/50 hover:border-yellow-500'
    }
  ];

  const additionalLinks: QuickLink[] = [
    {
      icon: LibraryIcon,
      title: 'Gallery',
      description: 'Browse completed artworks and timelapses',
      color: 'bg-purple-500/10 border-purple-500/50 hover:border-purple-500'
    },
    {
      icon: NotificationsIcon,
      title: 'Notifications',
      description: 'Stay updated with your activity',
      color: 'bg-pink-500/10 border-pink-500/50 hover:border-pink-500'
    },
    {
      icon: StoreIcon,
      title: 'Pricing',
      description: 'Upgrade for more tools and features',
      color: 'bg-indigo-500/10 border-indigo-500/50 hover:border-indigo-500'
    }
  ];

  const gameModes: QuickLink[] = [
    {
      icon: PlayIcon,
      title: 'Classic Canvas',
      description: 'Free-form collaborative drawing',
      color: 'bg-green-500/10 border-green-500/50 hover:border-green-500'
    },
    {
      icon: ClockIcon,
      title: 'Speed Sketch',
      description: 'Quick drawing challenges',
      color: 'bg-orange-500/10 border-orange-500/50 hover:border-orange-500'
    }
  ];

  const renderLinkCard = (link: QuickLink, index: number) => {
    const Icon = link.icon;
    return (
      <button
        key={index}
        className={`group p-6 rounded-xl border-2 transition-all duration-300 text-left ${link.color}`}
      >
        <Icon className="w-10 h-10 mb-4 group-hover:scale-110 transition-transform duration-300" />
        <h3 className="text-xl font-bold text-white mb-2">{link.title}</h3>
        <p className="text-sm text-zinc-400">{link.description}</p>
      </button>
    );
  };

  return (
    <div className="p-10 bg-black text-white min-h-full">
      <div className="max-w-6xl mx-auto">
        <div className="mb-10">
          <h1 className="text-4xl font-bold">Navigate DoodleParty</h1>
          <p className="text-zinc-400 mt-2">
            Your hub for everything creative. Choose where you want to go.
          </p>
        </div>

        {/* Main Sections */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">Main Sections</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {mainSections.map((link, index) => renderLinkCard(link, index))}
          </div>
        </section>

        {/* Quick Access */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">Quick Access</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {additionalLinks.map((link, index) => renderLinkCard(link, index))}
          </div>
        </section>

        {/* Game Modes */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">Game Modes</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {gameModes.map((link, index) => renderLinkCard(link, index))}
          </div>
        </section>

        {/* Current Activity Banner */}
        <section className="mt-16">
          <div className="relative overflow-hidden rounded-xl border border-green-500/50 bg-gradient-to-r from-green-500/10 to-transparent p-8">
            <div className="flex items-center gap-6">
              <div className="flex-shrink-0">
                <BrushIcon className="w-16 h-16 text-green-400" />
              </div>
              <div className="flex-grow">
                <h3 className="text-2xl font-bold mb-2">Current Canvas: Cyberpunk Nature</h3>
                <p className="text-zinc-300 mb-4">
                  Join 24+ artists creating in real-time. Session ends in 3 days!
                </p>
                <button className="bg-green-500 text-black font-bold py-2 px-6 rounded-lg hover:bg-green-400 transition-colors">
                  JOIN NOW
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default NavigatePage;
