import React from 'react';
import { TrophyIcon, StarIcon, FlameIcon, SparklesIcon } from '../../constants';

interface LeaderboardEntry {
  rank: number;
  player: string;
  score: number;
  streak: number;
  avatar: string;
  contributions: number;
  badges: string[];
}

const LeaderboardsPage: React.FC = () => {
  const weeklyLeaders: LeaderboardEntry[] = [
    { rank: 1, player: 'NovaSketch', score: 18420, streak: 12, avatar: 'https://picsum.photos/seed/nova/80/80', contributions: 234, badges: ['🏆', '🔥', '⭐'] },
    { rank: 2, player: 'PixelPilot', score: 17240, streak: 9, avatar: 'https://picsum.photos/seed/pilot/80/80', contributions: 198, badges: ['🥈', '🔥'] },
    { rank: 3, player: 'InkMage', score: 16510, streak: 7, avatar: 'https://picsum.photos/seed/mage/80/80', contributions: 176, badges: ['🥉', '⭐'] },
    { rank: 4, player: 'Brushstorm', score: 15100, streak: 6, avatar: 'https://picsum.photos/seed/brushstorm/64/64', contributions: 158, badges: ['⭐'] },
    { rank: 5, player: 'LumiLines', score: 14730, streak: 5, avatar: 'https://picsum.photos/seed/lumi/64/64', contributions: 142, badges: ['🔥'] },
    { rank: 6, player: 'ChromaDreamer', score: 13890, streak: 4, avatar: 'https://picsum.photos/seed/chroma/64/64', contributions: 128, badges: [] },
    { rank: 7, player: 'VectorVibe', score: 12560, streak: 8, avatar: 'https://picsum.photos/seed/vector/64/64', contributions: 115, badges: ['🔥'] },
    { rank: 8, player: 'ArtFlow', score: 11230, streak: 3, avatar: 'https://picsum.photos/seed/artflow/64/64', contributions: 98, badges: [] },
  ];

  const trendingClubs = [
    { name: 'Cyber Flora Collective', members: 248, growth: '+18%', icon: '🌿', color: 'from-green-500/20 to-emerald-500/20 border-green-500/50' },
    { name: 'Retro Pixel Guild', members: 192, growth: '+11%', icon: '🎮', color: 'from-purple-500/20 to-pink-500/20 border-purple-500/50' },
    { name: 'Chroma Storytellers', members: 165, growth: '+9%', icon: '📖', color: 'from-blue-500/20 to-cyan-500/20 border-blue-500/50' },
  ];

  const getMedalGradient = (rank: number) => {
    switch (rank) {
      case 1:
        return 'bg-gradient-to-br from-yellow-400 via-yellow-500 to-amber-600';
      case 2:
        return 'bg-gradient-to-br from-gray-300 via-gray-400 to-gray-500';
      case 3:
        return 'bg-gradient-to-br from-amber-600 via-amber-700 to-amber-800';
      default:
        return 'bg-zinc-800';
    }
  };

  const getMedalSize = (rank: number) => {
    if (rank === 1) return 'w-24 h-24';
    if (rank === 2) return 'w-20 h-20';
    if (rank === 3) return 'w-20 h-20';
    return 'w-12 h-12';
  };

  const topThree = weeklyLeaders.slice(0, 3);
  const restOfLeaders = weeklyLeaders.slice(3);

  return (
    <div className="bg-gradient-to-b from-black via-zinc-950 to-black text-white min-h-full">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="relative px-10 pt-12 pb-8">
          <div className="max-w-7xl mx-auto text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <TrophyIcon className="w-12 h-12 text-yellow-400" />
              <h1 className="text-5xl font-black bg-gradient-to-r from-yellow-400 via-yellow-200 to-yellow-400 bg-clip-text text-transparent">
                LEADERBOARDS
              </h1>
              <TrophyIcon className="w-12 h-12 text-yellow-400" />
            </div>
            <p className="text-xl text-zinc-300 mb-2">
              Celebrate the most talented creators in our community
            </p>
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/20 border border-green-500/50 text-green-400 text-sm font-semibold">
              <SparklesIcon className="w-4 h-4" />
              Season 3 · Week 8
            </div>
          </div>
        </div>
      </div>

      {/* Podium - Top 3 */}
      <section className="px-10 pb-12">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-end justify-center gap-6 mb-8">
            {/* 2nd Place */}
            <div className="flex-1 max-w-xs">
              <div className="bg-gradient-to-b from-zinc-900/80 to-zinc-900/40 border-2 border-gray-500/50 rounded-2xl p-6 text-center transform hover:scale-105 transition-transform">
                <div className="relative inline-block mb-4">
                  <img 
                    src={topThree[1].avatar} 
                    alt={topThree[1].player} 
                    className="w-20 h-20 rounded-full border-4 border-gray-400 mx-auto"
                  />
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-gray-300 to-gray-500 rounded-full flex items-center justify-center text-black font-bold text-sm">
                    2
                  </div>
                </div>
                <h3 className="text-xl font-bold mb-1">{topThree[1].player}</h3>
                <div className="flex justify-center gap-1 mb-3">
                  {topThree[1].badges.map((badge, i) => (
                    <span key={i} className="text-xl">{badge}</span>
                  ))}
                </div>
                <div className="text-3xl font-black text-gray-300 mb-2">
                  {topThree[1].score.toLocaleString()}
                </div>
                <div className="text-xs text-zinc-400 mb-1">🔥 {topThree[1].streak} day streak</div>
                <div className="text-xs text-zinc-400">✨ {topThree[1].contributions} contributions</div>
              </div>
              <div className="h-32 bg-gradient-to-b from-gray-500/30 to-gray-700/30 rounded-b-xl -mt-2 flex items-center justify-center">
                <span className="text-4xl font-black text-gray-400">2nd</span>
              </div>
            </div>

            {/* 1st Place - Elevated */}
            <div className="flex-1 max-w-xs">
              <div className="bg-gradient-to-b from-yellow-500/20 to-yellow-900/20 border-4 border-yellow-500/70 rounded-2xl p-8 text-center transform hover:scale-105 transition-transform shadow-2xl shadow-yellow-500/20">
                <div className="relative inline-block mb-4">
                  <div className="absolute inset-0 bg-yellow-400/30 rounded-full blur-xl" />
                  <img 
                    src={topThree[0].avatar} 
                    alt={topThree[0].player} 
                    className="relative w-24 h-24 rounded-full border-4 border-yellow-400 mx-auto"
                  />
                  <div className="absolute -top-3 -right-3 w-10 h-10 bg-gradient-to-br from-yellow-300 to-yellow-600 rounded-full flex items-center justify-center text-black font-black">
                    1
                  </div>
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2">
                    <TrophyIcon className="w-8 h-8 text-yellow-400" />
                  </div>
                </div>
                <h3 className="text-2xl font-black mb-2 text-yellow-300">{topThree[0].player}</h3>
                <div className="flex justify-center gap-1 mb-4">
                  {topThree[0].badges.map((badge, i) => (
                    <span key={i} className="text-2xl">{badge}</span>
                  ))}
                </div>
                <div className="text-4xl font-black text-yellow-400 mb-3">
                  {topThree[0].score.toLocaleString()}
                </div>
                <div className="text-sm text-yellow-200 mb-1">🔥 {topThree[0].streak} day streak</div>
                <div className="text-sm text-yellow-200">✨ {topThree[0].contributions} contributions</div>
              </div>
              <div className="h-40 bg-gradient-to-b from-yellow-500/40 to-yellow-700/40 rounded-b-xl -mt-2 flex items-center justify-center">
                <span className="text-5xl font-black text-yellow-400">1st</span>
              </div>
            </div>

            {/* 3rd Place */}
            <div className="flex-1 max-w-xs">
              <div className="bg-gradient-to-b from-zinc-900/80 to-zinc-900/40 border-2 border-amber-700/50 rounded-2xl p-6 text-center transform hover:scale-105 transition-transform">
                <div className="relative inline-block mb-4">
                  <img 
                    src={topThree[2].avatar} 
                    alt={topThree[2].player} 
                    className="w-20 h-20 rounded-full border-4 border-amber-700 mx-auto"
                  />
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-amber-600 to-amber-800 rounded-full flex items-center justify-center text-black font-bold text-sm">
                    3
                  </div>
                </div>
                <h3 className="text-xl font-bold mb-1">{topThree[2].player}</h3>
                <div className="flex justify-center gap-1 mb-3">
                  {topThree[2].badges.map((badge, i) => (
                    <span key={i} className="text-xl">{badge}</span>
                  ))}
                </div>
                <div className="text-3xl font-black text-amber-600 mb-2">
                  {topThree[2].score.toLocaleString()}
                </div>
                <div className="text-xs text-zinc-400 mb-1">🔥 {topThree[2].streak} day streak</div>
                <div className="text-xs text-zinc-400">✨ {topThree[2].contributions} contributions</div>
              </div>
              <div className="h-24 bg-gradient-to-b from-amber-700/30 to-amber-900/30 rounded-b-xl -mt-2 flex items-center justify-center">
                <span className="text-4xl font-black text-amber-700">3rd</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Rest of Rankings */}
      <section className="px-10 pb-12">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <StarIcon className="w-6 h-6 text-green-400" />
            Top Creators
          </h2>
          <div className="space-y-3">
            {restOfLeaders.map((entry) => (
              <div
                key={entry.rank}
                className="group bg-zinc-900/60 border border-zinc-800 hover:border-green-500/50 rounded-xl p-5 transition-all duration-300 hover:bg-zinc-900/80"
              >
                <div className="flex items-center gap-4">
                  <div className="flex-shrink-0 w-16 text-center">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-zinc-800 text-zinc-400 font-bold text-lg group-hover:bg-green-500/20 group-hover:text-green-400 transition-colors">
                      #{entry.rank}
                    </div>
                  </div>
                  <img 
                    src={entry.avatar} 
                    alt={entry.player} 
                    className="w-16 h-16 rounded-full border-2 border-zinc-700 group-hover:border-green-500 transition-colors"
                  />
                  <div className="flex-grow">
                    <h3 className="text-lg font-bold text-white mb-1">{entry.player}</h3>
                    <div className="flex items-center gap-4 text-sm text-zinc-400">
                      <span className="flex items-center gap-1">
                        <FlameIcon className="w-4 h-4 text-orange-400" />
                        {entry.streak} day streak
                      </span>
                      <span>✨ {entry.contributions} contributions</span>
                      {entry.badges.length > 0 && (
                        <span className="flex gap-1">
                          {entry.badges.map((badge, i) => (
                            <span key={i}>{badge}</span>
                          ))}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className="text-2xl font-black text-green-400">
                      {entry.score.toLocaleString()}
                    </div>
                    <div className="text-xs text-zinc-500 uppercase tracking-wider">Points</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trending Clubs */}
      <section className="px-10 pb-16">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">🌟 Trending Communities</h2>
          <div className="grid gap-6 md:grid-cols-3">
            {trendingClubs.map((club) => (
              <div 
                key={club.name} 
                className={`relative overflow-hidden rounded-2xl border-2 ${club.color} p-6 hover:scale-105 transition-transform`}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${club.color} opacity-50`} />
                <div className="relative">
                  <div className="text-5xl mb-4 text-center">{club.icon}</div>
                  <h3 className="text-xl font-bold text-white text-center mb-3">{club.name}</h3>
                  <div className="flex justify-between items-center">
                    <div className="text-center flex-1">
                      <div className="text-2xl font-black text-white">{club.members}</div>
                      <div className="text-xs text-zinc-400 uppercase">Members</div>
                    </div>
                    <div className="text-center flex-1">
                      <div className="text-2xl font-black text-green-400">{club.growth}</div>
                      <div className="text-xs text-zinc-400 uppercase">Growth</div>
                    </div>
                  </div>
                  <button className="mt-4 w-full bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 rounded-lg py-2 font-semibold transition-colors">
                    JOIN CLUB
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default LeaderboardsPage;
