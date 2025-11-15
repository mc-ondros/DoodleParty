import React from 'react'
import { Palette, Users, Zap, Shield, Gamepad2, Clock, WifiOff } from 'lucide-react'

export default function App() {
  return (
    <div className="relative min-h-screen bg-[#F5F1E8] text-gray-900">
      {/* Repeating Banner */}
      <div className="bg-[#FF1744] text-white py-2 overflow-hidden whitespace-nowrap">
        <div className="inline-block animate-marquee">
          <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
          <span className="text-sm font-black tracking-wider mx-8">★</span>
          <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
          <span className="text-sm font-black tracking-wider mx-8">★</span>
          <span className="text-sm font-black tracking-wider mx-8">AI-POWERED MODERATION</span>
          <span className="text-sm font-black tracking-wider mx-8">★</span>
          <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
          <span className="text-sm font-black tracking-wider mx-8">★</span>
          <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
          <span className="text-sm font-black tracking-wider mx-8">★</span>
        </div>
      </div>

      {/* Hero Section */}
      <header className="bg-[#F5F1E8] py-20">
        <div className="container mx-auto px-6 text-center">
          <div className="mb-8">
            <span className="inline-block px-6 py-2 bg-white border-4 border-black rounded-full font-black text-sm tracking-wider mb-4">COLLABORATIVE</span>
            <span className="inline-block px-6 py-2 bg-yellow-300 border-4 border-black rounded-full font-black text-sm tracking-wider mb-4 ml-3">REAL-TIME</span>
            <span className="inline-block px-6 py-2 bg-[#FF1744] text-white border-4 border-black rounded-full font-black text-sm tracking-wider mb-4 ml-3">AI-SAFE</span>
          </div>
          
          <h1 className="text-7xl md:text-9xl font-black leading-none mb-8 tracking-tight">
            DOODLE<br/>PARTY
          </h1>
          
          <p className="text-2xl md:text-4xl font-black mb-12 max-w-4xl mx-auto leading-tight">
            TURN YOUR EVENTS INTO<br/>EPIC ART EXPERIENCES
          </p>
          
          <button className="bg-black text-white font-black text-xl px-16 py-6 hover:bg-gray-900 transition-colors border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            START DRAWING NOW
          </button>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-20 px-6 md:px-12 bg-[#FF6B9D]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-6xl md:text-8xl font-black mb-6 text-white leading-none">WHY<br/>DOODLE<br/>PARTY?</h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard
              icon={<Users className="w-12 h-12" />}
              title="100+ PLAYERS"
              description="Raspberry Pi 4 or scale to 2,000+ in the cloud"
              color="bg-yellow-300"
            />
            <FeatureCard
              icon={<Zap className="w-12 h-12" />}
              title="LIGHTNING FAST"
              description="<50ms latency for smooth real-time drawing"
              color="bg-white"
            />
            <FeatureCard
              icon={<Shield className="w-12 h-12" />}
              title="AI SAFETY"
              description="Smart moderation in <50ms"
              color="bg-[#00BCD4]"
            />
            <FeatureCard
              icon={<Gamepad2 className="w-12 h-12" />}
              title="GAME MODES"
              description="Speed Sketch, Battle Royale, Story Canvas & more"
              color="bg-[#FF9800]"
            />
            <FeatureCard
              icon={<Clock className="w-12 h-12" />}
              title="EDGE ML"
              description="5MB model runs on Raspberry Pi 4"
              color="bg-[#8BC34A]"
            />
            <FeatureCard
              icon={<WifiOff className="w-12 h-12" />}
              title="OFFLINE MODE"
              description="No internet needed for events"
              color="bg-purple-400"
            />
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="py-20 px-6 md:px-10 bg-[#FFD600]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-6xl md:text-8xl font-black mb-8 leading-none">PRICING</h2>
          </div>
          <div className="flex flex-col lg:flex-row gap-8 max-w-6xl mx-auto justify-center">
            {[
              { 
                plan: 'HOBBY', 
                price: 'FREE', 
                description: 'Try it out', 
                features: ['10 players', 'Classic Canvas', 'AI moderation', 'Community support'],
                color: 'bg-white' 
              },
              { 
                plan: 'RASPBERRY PI 4', 
                price: '$499', 
                description: 'Hardware for events', 
                features: ['100 users', 'All game modes', 'Works offline', 'No internet', 'Full privacy', 'One-time buy'], 
                featured: true,
                color: 'bg-black' 
              },
              { 
                plan: 'CLOUD PRO', 
                price: '$199', 
                description: 'Managed hosting', 
                features: ['500 players', 'Auto-scaling', '99.9% uptime', 'AI & LLM', '24/7 support', 'Analytics'],
                perMonth: true,
                color: 'bg-[#FF1744]'
              },
            ].map((tier) => (
              <div
                key={tier.plan}
                className={`p-10 flex flex-col border-4 border-black transition-all hover:translate-y-[-4px] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] ${tier.color} ${
                  tier.featured ? 'text-white' : 'text-black'
                }`}
              >
                <h3 className="text-3xl font-black mb-2">{tier.plan}</h3>
                <p className={`mb-6 font-bold ${tier.featured ? 'text-gray-300' : 'text-gray-700'}`}>{tier.description}</p>
                <div className="mb-8">
                  <span className="text-6xl font-black">{tier.price}</span>
                  {tier.perMonth && <span className="text-xl font-black">/MO</span>}
                </div>
                <ul className="space-y-3 flex-grow mb-8 font-bold text-sm">
                  {tier.features.map((f, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span>★</span>
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
                <button className={`w-full py-4 font-black border-4 border-black transition-all ${
                  tier.featured ? 'bg-white text-black hover:bg-gray-100' : 'bg-black text-white hover:bg-gray-900'
                }`}>
                  {tier.plan === 'HOBBY' ? 'START FREE' : tier.featured ? 'ORDER NOW' : 'TRY FREE'}
                </button>
              </div>
            ))}
          </div>
          <div className="mt-12 text-center">
            <p className="text-2xl font-black">Enterprise: Pro ($399/mo) • Enterprise ($799/mo) • Custom</p>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-20 px-6 md:px-10 bg-[#00BCD4]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-6xl md:text-8xl font-black text-white leading-none mb-4">PERFECT<br/>FOR</h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <UseCase
              title="MUSIC FESTIVALS"
              description="Shared visuals during shows"
              color="bg-[#FF1744]"
            />
            <UseCase
              title="CONFERENCES"
              description="Interactive networking"
              color="bg-yellow-300"
            />
            <UseCase
              title="COMMUNITY"
              description="Bring neighborhoods together"
              color="bg-purple-400"
            />
            <UseCase
              title="CORPORATE"
              description="Team building workshops"
              color="bg-[#FF9800]"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6 md:px-10 bg-black">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-6xl md:text-8xl font-black mb-8 text-white leading-none">START<br/>DRAWING<br/>TODAY</h2>
          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <button className="px-12 py-6 bg-[#FF1744] text-white font-black text-xl border-4 border-white hover:bg-[#E01333] transition-colors">
              GET STARTED
            </button>
            <button className="px-12 py-6 bg-white text-black font-black text-xl border-4 border-white hover:bg-gray-100 transition-colors">
              SEE DEMO
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 md:px-10 bg-[#F5F1E8] border-t-8 border-black">
        <div className="max-w-7xl mx-auto text-center">
          <p className="mb-3 font-black text-3xl">
            DOODLEPARTY
          </p>
          <p className="text-sm font-bold text-gray-700">
            GNU GPL • NODE.JS • REACT • TENSORFLOW LITE
          </p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: { icon: React.ReactNode; title: string; description: string; color: string }) {
  return (
    <div className={`${color} p-8 border-4 border-black hover:translate-y-[-4px] transition-all shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]`}>
      <div className="mb-4">
        {icon}
      </div>
      <h3 className="text-2xl font-black mb-3">{title}</h3>
      <p className="font-bold text-sm leading-relaxed">{description}</p>
    </div>
  )
}

function UseCase({ title, description, color }: { title: string; description: string; color: string }) {
  return (
    <div className={`${color} p-8 text-center border-4 border-black hover:translate-y-[-4px] transition-all shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]`}>
      <h3 className="text-2xl font-black mb-3">{title}</h3>
      <p className="font-bold text-sm">{description}</p>
    </div>
  )
}
