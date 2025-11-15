import React, { useState } from 'react';
import type { Artwork, User } from '../../types';
import { SearchIcon } from '../../constants';

const mockArtists: User[] = [
  { name: 'PixelPioneer', avatarUrl: 'https://picsum.photos/seed/elon/32/32' },
  { name: 'SketchMaster', avatarUrl: 'https://picsum.photos/seed/user99/32/32' },
  { name: 'CuriousCanvas', avatarUrl: 'https://picsum.photos/seed/potato/32/32' },
  { name: 'DoodleDame', avatarUrl: 'https://picsum.photos/seed/cozy/32/32' },
  { name: 'ArtfulAntics', avatarUrl: 'https://picsum.photos/seed/sparta/32/32' },
];

const mockArtworks: Artwork[] = Array.from({ length: 15 }, (_, i) => ({
  id: i + 1,
  title: `Cyberpunk Blossom ${i + 1}`,
  imageUrl: `https://picsum.photos/seed/art${i}/500/${600 + Math.floor(Math.random() * 200)}`,
  artist: mockArtists[i % mockArtists.length],
  theme: 'Cyberpunk Nature',
  votes: Math.floor(Math.random() * 200) + 10,
}));

const ArtworkCard: React.FC<{ artwork: Artwork; onSelect: (a: Artwork) => void }> = ({ artwork, onSelect }) => (
  <div className="relative group cursor-pointer" onClick={() => onSelect(artwork)}>
    <img src={artwork.imageUrl} alt={artwork.title} className="w-full h-auto object-cover rounded-lg transition-transform duration-300 group-hover:scale-105" />
    <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg">
      <div className="absolute bottom-4 left-4 text-white">
        <h3 className="font-bold">{artwork.title}</h3>
        <p className="text-sm text-zinc-300">by {artwork.artist.name}</p>
      </div>
    </div>
  </div>
);

const ArtworkModal: React.FC<{ artwork: Artwork; onClose: () => void }> = ({ artwork, onClose }) => (
  <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center" onClick={onClose}>
    <div className="bg-zinc-900 rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] flex overflow-hidden" onClick={(e) => e.stopPropagation()}>
      <img src={artwork.imageUrl} alt={artwork.title} className="w-2/3 object-cover" />
      <div className="w-1/3 p-6 flex flex-col text-white">
        <div className="flex-grow">
          <h2 className="text-3xl font-bold mb-2">{artwork.title}</h2>
          <p className="text-zinc-400 mb-6">Theme: <span className="font-semibold text-zinc-200">{artwork.theme}</span></p>
          <div className="flex items-center mb-6">
            <img src={artwork.artist.avatarUrl} alt={artwork.artist.name} className="w-12 h-12 rounded-full mr-4" />
            <div>
              <p className="text-zinc-400 text-sm">Created by</p>
              <p className="font-bold text-lg">{artwork.artist.name}</p>
            </div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <p className="text-zinc-400 text-sm">Votes</p>
            <p className="text-2xl font-bold">{artwork.votes}</p>
          </div>
        </div>
        <div className="flex space-x-2">
          <button className="flex-1 bg-green-500 text-black font-bold py-3 rounded-md hover:bg-green-600 transition-colors">Vote</button>
          <button className="flex-1 bg-zinc-700 text-white font-bold py-3 rounded-md hover:bg-zinc-600 transition-colors">Share</button>
        </div>
      </div>
    </div>
  </div>
);

const GalleryPage: React.FC = () => {
  const [selectedArtwork, setSelectedArtwork] = useState<Artwork | null>(null);
  return (
    <div className="bg-black text-white min-h-full p-10">
      {selectedArtwork && <ArtworkModal artwork={selectedArtwork} onClose={() => setSelectedArtwork(null)} />}
      <header className="mb-10">
        <h1 className="text-5xl font-extrabold">Gallery</h1>
        <p className="mt-2 text-zinc-400">Explore artwork from the community. Vote for your favorites!</p>
      </header>
      <div className="flex justify-between items-center mb-8">
        <div className="flex space-x-2">
          <button className="bg-zinc-800 px-4 py-2 rounded-md font-semibold text-sm">Trending</button>
          <button className="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-4 py-2 rounded-md font-semibold text-sm">Newest</button>
          <button className="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-4 py-2 rounded-md font-semibold text-sm">Most Voted</button>
        </div>
        <div className="relative w-1/3">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
          <input type="text" placeholder="Search by artist or title..." className="w-full bg-zinc-800 border border-zinc-700 rounded-md pl-10 pr-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-green-500" />
        </div>
      </div>
      <div className="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-6 space-y-6">
        {mockArtworks.map(a => (
          <ArtworkCard key={a.id} artwork={a} onSelect={setSelectedArtwork} />
        ))}
      </div>
    </div>
  );
};

export default GalleryPage;
