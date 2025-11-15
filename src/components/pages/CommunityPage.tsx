import React, { useState } from 'react';
import type { DiscussionPost } from '../../types';

const DiscussionPostItem: React.FC<{ post: DiscussionPost }> = ({ post }) => (
  <div className="flex items-start space-x-2 md:space-x-4 p-3 md:p-4 hover:bg-zinc-800/50 rounded-lg transition-colors">
    <div className="flex flex-col items-center text-zinc-400 flex-shrink-0">
      <button className="hover:text-green-400 transition-colors">▲</button>
      <span className="font-bold text-white text-sm md:text-base">{post.votes}</span>
      <button className="hover:text-red-400 transition-colors">▼</button>
    </div>
    <div className="flex-grow min-w-0">
      <p className="text-base md:text-lg font-semibold text-white mb-1">{post.title}</p>
      <p className="text-xs md:text-sm text-zinc-400 mb-2">Posted by {post.author} · {post.postedAgo}</p>
      <div className="flex items-center flex-wrap gap-3 md:gap-4 text-xs md:text-sm text-zinc-400">
        <button className="hover:text-white transition-colors">💬 {post.comments} comments</button>
        <button className="hover:text-white transition-colors hidden sm:inline">Share</button>
        <button className="hover:text-white transition-colors hidden sm:inline">Save</button>
      </div>
    </div>
    <div className="hidden sm:flex -space-x-2 flex-shrink-0">
      {post.userAvatars.map((avatar, index) => (
        <img key={index} src={avatar} alt="user" className="w-6 h-6 rounded-full border-2 border-zinc-900" />
      ))}
    </div>
  </div>
);

const CommunityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('Trending');
  const discussionPosts: DiscussionPost[] = [
    { id: 1, author: 'ArtfulAntics', title: "My attempt at the 'Cyberpunk Nature' theme!", postedAgo: '2 hours ago', votes: 25, comments: 24, userAvatars: ['https://picsum.photos/seed/u1/24/24', 'https://picsum.photos/seed/u2/24/24', 'https://picsum.photos/seed/u3/24/24'] },
    { id: 2, author: 'PixelPioneer', title: "A technique for blending colors I just discovered.", postedAgo: '1 day ago', votes: 21, comments: 22, userAvatars: ['https://picsum.photos/seed/u4/24/24', 'https://picsum.photos/seed/u5/24/24'] },
    { id: 3, author: 'SketchMaster', title: 'What do you think of my new character design?', postedAgo: '3 days ago', votes: 22, comments: 18, userAvatars: ['https://picsum.photos/seed/u6/24/24', 'https://picsum.photos/seed/u7/24/24'] },
  ];

  const chatLobbies = [
    { name: '#general-lounge', description: 'A public place to discuss everything about art and doodling.', users: 16 },
    { name: '#collaboration-station', description: 'Looking for a partner for your next masterpiece? Find them here!', users: 27 },
    { name: '#feedback-forum', description: 'Want constructive criticism? Share your work here.', users: 21 },
    { name: '#prompt-ideas', description: 'Share your ideas for future canvas themes and challenges.', users: 14 },
  ];

  return (
    <div className="bg-zinc-900/50 min-h-full">
      <header className="text-white bg-black border-b border-zinc-800">
        {/* Hero section with background image */}
        <div className="relative h-full w-full overflow-hidden bg-black">
          <img
            src="https://images.unsplash.com/photo-1522542550221-31fd19575a2d?auto=format&fit=crop&w=1200&q=80"
            alt="Artists collaborating"
            onError={(e) => {
              e.currentTarget.onerror = null;
              e.currentTarget.src = 'https://picsum.photos/seed/community-hero/1200/400';
            }}
            className="absolute inset-0 h-full w-full object-cover"
            style={{
              maskImage: 'linear-gradient(to bottom, black 60%, transparent 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, black 60%, transparent 100%)',
            }}
          />

          {/* Overlay for contrast */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/40 to-transparent" />

          {/* Content */}
          <div className="relative z-10 flex min-h-[280px] md:min-h-[360px] flex-col justify-between px-4 py-8 md:px-12 md:py-12">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs md:text-sm font-semibold uppercase tracking-wider">Community</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-xs md:text-sm font-semibold uppercase tracking-wider">Connect</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-xs md:text-sm font-semibold uppercase tracking-wider">Create</span>
                </div>
              </div>
            </div>

            <div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold mb-3 md:mb-4">
                Community Hub
              </h1>
              <p className="text-base md:text-lg text-zinc-300 mb-4 max-w-2xl">
                Connect, share, and get inspired with fellow artists.
              </p>
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2 bg-black/40 backdrop-blur-sm border border-zinc-700 rounded-full px-4 py-1.5">
                  <span className="text-xl md:text-2xl">👥</span>
                  <span className="text-xs md:text-sm font-semibold">1,247 Active</span>
                </div>
                <div className="flex items-center gap-2 bg-black/40 backdrop-blur-sm border border-green-500/50 rounded-full px-4 py-1.5">
                  <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  <span className="text-xs md:text-sm font-semibold text-green-400">89 Online</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="px-4 md:px-12 mt-4 md:mt-6 pb-4 flex flex-wrap gap-4 md:gap-8 text-zinc-300">
          {['Gallery', 'Discussions', 'Artists', 'Events'].map(item => (
            <button 
              key={item} 
              className={`font-semibold pb-2 border-b-2 transition-colors text-sm md:text-base ${
                item === 'Discussions' 
                  ? 'text-white border-white' 
                  : 'border-transparent hover:text-white hover:border-zinc-600'
              }`}
            >
              {item}
            </button>
          ))}
        </nav>
      </header>

      <main className="p-4 md:p-8">
        <div className="bg-zinc-800 p-4 md:p-6 rounded-lg mb-6 md:mb-8 flex flex-col sm:flex-row items-start sm:items-center gap-4 justify-between">
          <div className="flex-grow">
            <h2 className="text-lg md:text-xl font-bold mb-2">Doodle of the Week</h2>
            <p className="text-sm md:text-base text-zinc-400">Every week we feature an outstanding piece from the community canvas. <span className="text-white font-semibold">Vote for your favorites from the gallery to help us choose the next winner!</span></p>
          </div>
          <img src="https://picsum.photos/seed/doodle-winner/200/100" className="rounded-md object-cover w-full sm:w-48 flex-shrink-0" alt="doodle of the week"/>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8">
          <div className="lg:col-span-2">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
              <h2 className="text-xl md:text-2xl font-bold">Discussions</h2>
              <button className="bg-zinc-700 hover:bg-zinc-600 text-white font-semibold py-2 px-4 rounded-md text-xs md:text-sm whitespace-nowrap">CREATE A DISCUSSION</button>
            </div>
            <div className="flex space-x-3 md:space-x-4 border-b border-zinc-700 mb-4 overflow-x-auto">
              {['Trending', 'Newest', 'Most Popular'].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} className={`py-2 px-1 font-semibold text-sm md:text-base whitespace-nowrap ${activeTab === tab ? 'text-white border-b-2 border-white' : 'text-zinc-400'}`}>
                  {tab}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              {discussionPosts.map(post => <DiscussionPostItem key={post.id} post={post} />)}
            </div>
          </div>
          <div>
            <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4">Art Lounges</h2>
            <p className="text-zinc-400 text-xs md:text-sm mb-3 md:mb-4">Connect with other artists around you to form a party, discuss news or just have a nice talk.</p>
            <div className="space-y-3 md:space-y-4">
              {chatLobbies.map(lobby => (
                <div key={lobby.name} className="p-3 md:p-4 bg-zinc-800 rounded-lg hover:bg-zinc-750 transition-colors">
                  <p className="font-semibold text-white text-sm md:text-base">{lobby.name}</p>
                  <p className="text-xs md:text-sm text-zinc-400 mt-1">{lobby.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <p className="text-xs text-green-400 font-semibold">{lobby.users} users online</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default CommunityPage;
