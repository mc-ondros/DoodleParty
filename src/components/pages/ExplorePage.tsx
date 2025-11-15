import React from 'react';
import CountdownTimer from '../ui/CountdownTimer';
import type { User } from '../../types';
import type { Page } from '../../App';
import { PlayIcon, ClockIcon, TargetIcon, SwordsIcon, BookTextIcon } from '../../constants';

interface ExplorePageProps {
  setCurrentPage?: (page: Page) => void;
}

const ExplorePage: React.FC<ExplorePageProps> = ({ setCurrentPage }) => {
  const heroImageUrl =
    'https://images.unsplash.com/photo-1624884424146-7ae993a5f70e?auto=format&fit=crop&w=2400&q=80';
  const heroFallbackUrl = 'https://picsum.photos/id/0/2000/1200';

  const participants: User[] = [
    { name: 'User 1', avatarUrl: 'https://picsum.photos/seed/p1/40/40' },
    { name: 'User 2', avatarUrl: 'https://picsum.photos/seed/p2/40/40' },
    { name: 'User 3', avatarUrl: 'https://picsum.photos/seed/p3/40/40' },
    { name: 'User 4', avatarUrl: 'https://picsum.photos/seed/p4/40/40' },
    { name: 'User 5', avatarUrl: 'https://picsum.photos/seed/p5/40/40' },
    { name: 'User 6', avatarUrl: 'https://picsum.photos/seed/p6/40/40' },
  ];

  const handleHeroError = (event: React.SyntheticEvent<HTMLImageElement>) => {
    event.currentTarget.onerror = null;
    event.currentTarget.src = heroFallbackUrl;
  };

  return (
    <div className="relative min-h-screen bg-zinc-900">
      {/* Header Section */}
      <header id="explore" className="text-white bg-black">
        {/* Top banner section with background image */}
        <div className="relative h-full w-full overflow-hidden rounded-none bg-black md:rounded-lg">
          <img
            src={heroImageUrl}
            alt="Aquarium with lush aquatic plants"
            onError={handleHeroError}
            className="absolute inset-0 h-full w-full object-cover"
            style={{
              maskImage: 'linear-gradient(to bottom, black 70%, transparent 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, black 70%, transparent 100%)',
            }}
          />

          {/* Global overlay for contrast */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-transparent" />

          {/* Foreground content overlay */}
          <div className="relative z-10 flex min-h-[320px] flex-col justify-between px-6 py-10 md:min-h-[420px] md:px-12 md:py-12">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold uppercase tracking-wider">Art</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">Collaboration</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">Fun</span>
                </div>
              </div>
              <button className="bg-white text-black font-bold py-2 px-4 rounded-md text-sm hover:bg-zinc-200 transition-colors flex-shrink-0">
                JOIN THE CANVAS
              </button>
            </div>

            <div>
              <div className="flex items-center">
                <div className="flex -space-x-3">
                  {participants.map((p, i) => (
                    <img
                      key={i}
                      src={p.avatarUrl}
                      alt={p.name}
                      className="w-10 h-10 rounded-full border-2 border-black"
                    />
                  ))}
                  <div className="w-10 h-10 rounded-full bg-zinc-700 flex items-center justify-center text-xs font-bold border-2 border-black">
                    +18
                  </div>
                </div>
                <p className="ml-4 font-semibold">
                  <span className="text-green-400">PEOPLE DRAWING RIGHT NOW</span>
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Main title and stats below the banner */}
        <div className="px-6 pb-12 pt-8 md:px-12">
            <h1 className="text-6xl font-black leading-tight">
                Join The Real-Time Collaborative Canvas
            </h1>
            <div className="mt-8 flex gap-8">
                <div>
                    <p className="text-sm text-zinc-400">Canvas Status</p>
                    <p className="font-semibold flex items-center"><span className="mr-2 h-2 w-2 rounded-full bg-green-500"></span>Live</p>
                </div>
                <div>
                    <p className="text-sm text-zinc-400">Session Ends</p>
                    <p className="font-semibold">In 3 days</p>
                </div>
                <div>
                    <p className="text-sm text-zinc-400">Current Theme</p>
                    <p className="font-semibold">Cyberpunk Nature</p>
                </div>
            </div>
        </div>
      </header>
      
      {/* Countdown Bar */}
      <div className="sticky top-0 z-20 bg-green-500 text-black py-2 px-3 md:p-4 flex flex-col sm:flex-row items-center justify-between gap-2 md:gap-4">
        <CountdownTimer />
        <div className="text-center hidden sm:block">
          <p className="font-bold text-sm md:text-base">The drawing session ends soon</p>
          <p className="text-xs md:text-sm">Hurry up to not miss your chance to contribute!</p>
        </div>
        <button className="bg-black text-white font-bold py-1.5 px-4 md:py-2 md:px-6 rounded-md text-xs md:text-sm hover:bg-zinc-800 transition-colors whitespace-nowrap">
          START DRAWING
        </button>
      </div>

      {/* Content Section */}
      <div className="p-12 grid grid-cols-1 md:grid-cols-3 gap-12 bg-zinc-900">
        <div className="md:col-span-1">
            <div className="relative group mb-4">
                <img src="https://picsum.photos/seed/doodle-art/400/500" alt="Doodle art preview" className="rounded-lg w-full object-cover"/>
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="bg-white/20 backdrop-blur-sm p-4 rounded-full">
                        <PlayIcon className="w-8 h-8 text-white" />
                    </button>
                </div>
            </div>
            <h2 className="text-xl font-bold">About This Canvas</h2>
            <p className="text-zinc-400 text-sm">Watch a timelapse of the canvas creation so far.</p>
        </div>
        <div className="md:col-span-2 space-y-8">
            <div>
                <h2 className="text-2xl font-bold mb-4">Drawing Prompt:</h2>
                <p className="text-zinc-300 leading-relaxed">
                    This week's theme is <strong className="text-green-400">"Cyberpunk Nature"</strong>. Imagine a world where advanced technology and the natural world have merged in unexpected ways. Think neon forests, robotic animals, data streams flowing like rivers, or circuits growing on trees. Let your creativity run wild and add your vision to our collective masterpiece!
                </p>
            </div>
            <div>
                 <h3 className="text-xl font-bold mb-3">How It Works</h3>
                 <ul className="list-disc list-inside text-zinc-400 space-y-2">
                    <li>Anyone can draw, anytime. Just jump in and add to the canvas.</li>
                    <li>Be respectful. Our AI moderation is active to keep the space creative and safe.</li>
                    <li>Work together! Build on others' ideas to create something truly unique.</li>
                    <li>At the end of the session, a timelapse video of the entire creation will be generated and added to the gallery.</li>
                 </ul>
            </div>
             <div>
                 <h3 className="text-xl font-bold mb-3">Need Inspiration?</h3>
                 <p className="text-zinc-400">
                    How would a flower adapt in a city of chrome? What kind of creature would live in a digital forest? What does a sunset look like over a circuit board mountain range? Use these questions as a starting point, or just let your imagination guide you.
                 </p>
            </div>
        </div>
      </div>

      {/* Gallery Section - Preview */}
      <section id="gallery" className="py-12 px-6 md:px-10 bg-zinc-900">
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-4xl font-black mb-2">Gallery</h2>
              <p className="text-zinc-400">Explore artwork from the community</p>
            </div>
            <button onClick={() => setCurrentPage?.('gallery')} className="bg-zinc-700 hover:bg-zinc-600 text-white font-semibold py-2 px-6 rounded-lg transition-colors">
              View All
            </button>
          </div>
          <div className="columns-2 md:columns-3 lg:columns-4 gap-4 space-y-4">
            {[1,2,3,4,5,6,7,8].map((i) => (
              <div key={i} className="relative group cursor-pointer break-inside-avoid" onClick={() => setCurrentPage?.('gallery')}>
                <img src={`https://picsum.photos/seed/art${i}/400/${500 + i * 20}`} alt={`Artwork ${i}`} className="w-full h-auto object-cover rounded-lg transition-transform duration-300 group-hover:scale-105" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg">
                  <div className="absolute bottom-3 left-3 text-white">
                    <p className="font-bold text-sm">Cyberpunk Blossom {i}</p>
                    <p className="text-xs text-zinc-300">by Artist{i}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-14 px-6 md:px-10 bg-zinc-900">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-5xl font-extrabold mb-4">Find a plan to power your creativity</h2>
            <p className="text-lg text-zinc-400">
              Whether you're doodling for fun, running a community event, or deploying at scale, we have a plan that's right for you.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[
              { plan: 'Hobby', price: 'Free', description: 'For individuals and small groups just starting out.', features: ['Up to 10 concurrent users', 'Classic Canvas mode', 'Basic AI moderation', 'Community support'] },
              { plan: 'Pro', price: '$25', description: 'For event organizers, streamers, and growing communities.', features: ['Up to 100 concurrent users', 'All game modes included', 'Advanced AI moderation', 'Custom branding options', 'Priority support', 'Analytics dashboard'], featured: true },
              { plan: 'Enterprise', price: 'Custom', description: 'For large-scale deployments and unique requirements.', features: ['Unlimited concurrent users', 'On-premises RPi4 deployment option', 'Dedicated infrastructure', 'Custom LLM integration', '24/7 premium support', 'Service Level Agreement (SLA)'] },
            ].map((tier) => (
              <div
                key={tier.plan}
                className={`border rounded-xl p-8 flex flex-col ${
                  tier.featured ? 'border-green-500 bg-zinc-900' : 'border-zinc-700 bg-zinc-800/50'
                }`}
              >
                <h3 className="text-xl font-bold text-white">{tier.plan}</h3>
                <p className="mt-2 text-zinc-400">{tier.description}</p>
                <div className="mt-6">
                  <span className="text-5xl font-extrabold text-white">{tier.price}</span>
                  {tier.plan === 'Pro' && <span className="text-zinc-400">/ month</span>}
                </div>
                <ul className="mt-8 space-y-4 text-zinc-300 flex-grow">
                  {tier.features.map((f, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="text-green-500 mr-3 mt-1">✓</span>
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
                <button className={`mt-10 w-full py-3 font-semibold rounded-lg transition-colors ${
                  tier.featured ? 'bg-green-500 text-black hover:bg-green-600' : 'bg-zinc-700 text-white hover:bg-zinc-600'
                }`}>
                  {tier.plan === 'Enterprise' ? 'Contact Sales' : 'Get Started'}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Game Modes Section */}
      <section id="game-modes" className="py-14 px-6 md:px-10 bg-zinc-900">
        <div className="max-w-7xl mx-auto">
          <div className="mb-10">
            <h2 className="text-4xl font-black mb-2">Game Modes</h2>
            <p className="text-zinc-400">Choose your creative challenge</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Classic Canvas */}
            <div id="classic-canvas" className="group relative overflow-hidden rounded-xl bg-zinc-800 border border-zinc-700 hover:border-green-500 transition-all duration-300">
              <div className="p-6">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                  <PlayIcon className="w-6 h-6 text-green-500" />
                </div>
                <h3 className="text-2xl font-bold mb-2">Classic Canvas</h3>
                <p className="text-zinc-400 mb-6">Free-form collaborative drawing. Jump in and add your touch.</p>
                <button onClick={() => setCurrentPage?.('classic-canvas')} className="w-full bg-green-500 text-black font-semibold py-3 rounded-lg hover:bg-green-400 transition-colors">
                  Join Now
                </button>
              </div>
            </div>
            
            {/* Speed Sketch */}
            <div id="speed-sketch" className="group relative overflow-hidden rounded-xl bg-zinc-800 border border-zinc-700 hover:border-green-500 transition-all duration-300">
              <div className="p-6">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                  <ClockIcon className="w-6 h-6 text-green-500" />
                </div>
                <h3 className="text-2xl font-bold mb-2">Speed Sketch</h3>
                <p className="text-zinc-400 mb-6">Quick drawing challenges to test your speed and creativity.</p>
                <button onClick={() => setCurrentPage?.('classic-canvas')} className="w-full bg-green-500 text-black font-semibold py-3 rounded-lg hover:bg-green-400 transition-colors">
                  Play
                </button>
              </div>
            </div>

            {/* Guess the Doodle */}
            <div className="group relative overflow-hidden rounded-xl bg-zinc-800/50 border border-zinc-700">
              <div className="absolute inset-0 bg-zinc-900/60 backdrop-blur-[2px] flex items-center justify-center z-10">
                <span className="text-zinc-400 font-bold text-lg">COMING SOON</span>
              </div>
              <div className="p-6">
                <div className="w-12 h-12 bg-zinc-700 rounded-lg flex items-center justify-center mb-4">
                  <TargetIcon className="w-6 h-6 text-zinc-500" />
                </div>
                <h3 className="text-2xl font-bold mb-2 text-zinc-300">Guess the Doodle</h3>
                <p className="text-zinc-500 mb-6">Draw prompts and let others guess what you're creating.</p>
                <button disabled className="w-full bg-zinc-700 text-zinc-500 font-semibold py-3 rounded-lg cursor-not-allowed">
                  Coming Soon
                </button>
              </div>
            </div>

            {/* Battle Royale */}
            <div className="group relative overflow-hidden rounded-xl bg-zinc-800/50 border border-zinc-700">
              <div className="absolute inset-0 bg-zinc-900/60 backdrop-blur-[2px] flex items-center justify-center z-10">
                <span className="text-zinc-400 font-bold text-lg">COMING SOON</span>
              </div>
              <div className="p-6">
                <div className="w-12 h-12 bg-zinc-700 rounded-lg flex items-center justify-center mb-4">
                  <SwordsIcon className="w-6 h-6 text-zinc-500" />
                </div>
                <h3 className="text-2xl font-bold mb-2 text-zinc-300">Battle Royale</h3>
                <p className="text-zinc-500 mb-6">Compete against others in fast-paced drawing battles.</p>
                <button disabled className="w-full bg-zinc-700 text-zinc-500 font-semibold py-3 rounded-lg cursor-not-allowed">
                  Coming Soon
                </button>
              </div>
            </div>

            {/* Collaborative Story */}
            <div className="group relative overflow-hidden rounded-xl bg-zinc-800/50 border border-zinc-700">
              <div className="absolute inset-0 bg-zinc-900/60 backdrop-blur-[2px] flex items-center justify-center z-10">
                <span className="text-zinc-400 font-bold text-lg">COMING SOON</span>
              </div>
              <div className="p-6">
                <div className="w-12 h-12 bg-zinc-700 rounded-lg flex items-center justify-center mb-4">
                  <BookTextIcon className="w-6 h-6 text-zinc-500" />
                </div>
                <h3 className="text-2xl font-bold mb-2 text-zinc-300">Collaborative Story</h3>
                <p className="text-zinc-500 mb-6">Build a visual story together, one panel at a time.</p>
                <button disabled className="w-full bg-zinc-700 text-zinc-500 font-semibold py-3 rounded-lg cursor-not-allowed">
                  Coming Soon
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ExplorePage;
