import React, { useState } from 'react';
import type { DiscussionPost } from '../../types';

const DiscussionPostItem: React.FC<{ post: DiscussionPost }> = ({ post }) => (
  <div className="flex items-start space-x-4 p-4 hover:bg-zinc-800/50 rounded-lg">
    <div className="flex flex-col items-center text-zinc-400">
      <button>▲</button>
      <span className="font-bold text-white">{post.votes}</span>
      <button>▼</button>
    </div>
    <div className="flex-grow">
      <p className="text-lg font-semibold text-white">{post.title}</p>
      <p className="text-sm text-zinc-400">Posted by {post.author} · {post.postedAgo}</p>
      <div className="flex items-center space-x-4 mt-2 text-sm text-zinc-400">
        <span>{post.comments} comments</span>
        <span>Share</span>
        <span>Save</span>
      </div>
    </div>
    <div className="flex -space-x-2">
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
      <header className="p-8 border-b border-zinc-800 bg-cover bg-center" style={{backgroundImage: "url('https://picsum.photos/seed/doodle-bg/1200/300')"}}>
        <div className="bg-black/50 p-4 rounded-lg backdrop-blur-sm inline-block">
          <h1 className="text-5xl font-extrabold text-white">Community Hub</h1>
          <p className="text-zinc-300 mt-2">Connect, share, and get inspired with fellow artists.</p>
        </div>
        <nav className="mt-6 flex space-x-8 text-zinc-300">
          {['Gallery', 'Discussions', 'Artists', 'Events'].map(item => (
            <button key={item} className={`font-semibold pb-2 border-b-2 ${item === 'Discussions' ? 'text-white border-white' : 'border-transparent hover:text-white'}`}>
              {item}
            </button>
          ))}
        </nav>
      </header>

      <main className="p-8">
        <div className="bg-zinc-800 p-6 rounded-lg mb-8 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Doodle of the Week</h2>
            <p className="text-zinc-400">Every week we feature an outstanding piece from the community canvas. <span className="text-white font-semibold">Vote for your favorites from the gallery to help us choose the next winner!</span></p>
          </div>
          <img src="https://picsum.photos/seed/doodle-winner/200/100" className="rounded-md object-cover" alt="doodle of the week"/>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Discussions</h2>
              <button className="bg-zinc-700 hover:bg-zinc-600 text-white font-semibold py-2 px-4 rounded-md text-sm">CREATE A DISCUSSION</button>
            </div>
            <div className="flex space-x-4 border-b border-zinc-700 mb-4">
              {['Trending', 'Newest', 'Most Popular'].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} className={`py-2 px-1 font-semibold ${activeTab === tab ? 'text-white border-b-2 border-white' : 'text-zinc-400'}`}>
                  {tab}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              {discussionPosts.map(post => <DiscussionPostItem key={post.id} post={post} />)}
            </div>
          </div>
          <div>
            <h2 className="text-2xl font-bold mb-4">Art Lounges</h2>
            <p className="text-zinc-400 text-sm mb-4">Connect with other artists around you to form a party, discuss news or just have a nice talk.</p>
            <div className="space-y-4">
              {chatLobbies.map(lobby => (
                <div key={lobby.name} className="p-4 bg-zinc-800 rounded-lg">
                  <p className="font-semibold text-white">{lobby.name}</p>
                  <p className="text-sm text-zinc-400">{lobby.description}</p>
                  <p className="text-xs text-green-400 mt-2">{lobby.users} users</p>
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
