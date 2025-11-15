import React from 'react'
import { Palette, Users, Zap, Shield, Gamepad2, Clock, WifiOff } from 'lucide-react'

const WAVE_PATH = 'M0 60 C 100 20 200 100 300 60 C 400 20 500 100 600 60 C 700 20 800 100 900 60 C 1000 20 1100 100 1200 60 V 120 H 0 Z'

type WaveDividerProps = {
  topColor: string
  waveColor: string
  reverse?: boolean
}

export default function App() {
  return (
    <div className="relative min-h-screen bg-[#F5F1E8] text-gray-900">
      {/* Repeating Banner */}
      <div className="bg-[#FF1744] text-white py-2 overflow-hidden">
        <div className="flex animate-marquee">
          <div className="flex-shrink-0 flex">
            <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">AI-POWERED MODERATION</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
          </div>
          <div className="flex-shrink-0 flex">
            <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">AI-POWERED MODERATION</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
          </div>
          <div className="flex-shrink-0 flex">
            <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">AI-POWERED MODERATION</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
          </div>
          <div className="flex-shrink-0 flex">
            <span className="text-sm font-black tracking-wider mx-8">REAL-TIME COLLABORATIVE DRAWING</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">100+ CONCURRENT USERS</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
            <span className="text-sm font-black tracking-wider mx-8">AI-POWERED MODERATION</span>
            <span className="text-sm font-black tracking-wider mx-8">★</span>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <header className="bg-[#F5F1E8] py-20">
        <div className="container mx-auto px-6 text-center">
          <div className="mb-8 flex flex-wrap justify-center gap-3">
            <span className="inline-block px-6 py-2 bg-white border-4 border-black rounded-full font-black text-sm tracking-wider transition-all duration-300 hover:scale-110 hover:rotate-2 cursor-default animate-fade-in-up" style={{animationDelay: '0.1s'}}>COLLABORATIVE</span>
            <span className="inline-block px-6 py-2 bg-yellow-300 border-4 border-black rounded-full font-black text-sm tracking-wider transition-all duration-300 hover:scale-110 hover:rotate-[-2deg] cursor-default animate-fade-in-up" style={{animationDelay: '0.2s'}}>REAL-TIME</span>
            <span className="inline-block px-6 py-2 bg-[#FF1744] text-white border-4 border-black rounded-full font-black text-sm tracking-wider transition-all duration-300 hover:scale-110 hover:rotate-2 cursor-default animate-fade-in-up" style={{animationDelay: '0.3s'}}>AI-SAFE</span>
          </div>
          
          <h1 className="text-7xl md:text-9xl font-black leading-none mb-8 tracking-tight transition-all duration-700 hover:tracking-wide">
            DOODLE<br/>PARTY
          </h1>
          
          <p className="text-2xl md:text-4xl font-black mb-12 max-w-4xl mx-auto leading-tight opacity-90">
            TURN YOUR EVENTS INTO<br/>EPIC ART EXPERIENCES
          </p>
          
          <button className="group relative bg-[#FF1744] text-white font-black text-xl px-16 py-6 transition-all duration-300 ease-out border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[-2px] hover:translate-y-[-2px] active:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] active:translate-x-[2px] active:translate-y-[2px]">
            <span className="relative z-10">START DRAWING NOW</span>
            <div className="absolute inset-0 bg-gradient-to-r from-[#FF1744] to-[#E01333] opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </button>
        </div>
      </header>

      {/* Wave Divider */}
      <WaveDivider topColor="bg-[#F5F1E8]" waveColor="#FF6B9D" />

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

      {/* Wave Divider */}
      <WaveDivider topColor="bg-[#FF6B9D]" waveColor="#FFD600" reverse />

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
                className={`group p-10 flex flex-col border-4 border-black transition-all duration-500 ease-out hover:translate-y-[-12px] hover:translate-x-[-4px] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[16px_16px_0px_0px_rgba(0,0,0,1)] ${tier.color} ${
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
                <button className={`w-full py-4 font-black border-4 border-black transition-all duration-300 ease-out hover:translate-y-[-2px] active:translate-y-[0px] ${
                  tier.featured ? 'bg-white text-black hover:bg-gray-100 hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]' : 'bg-black text-white hover:bg-gray-900 hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]'
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

      {/* Wave Divider */}
      <WaveDivider topColor="bg-[#FFD600]" waveColor="#00BCD4" />

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

      {/* Wave Divider */}
      <WaveDivider topColor="bg-[#00BCD4]" waveColor="#000000" reverse />

      {/* CTA Section */}
      <section className="py-24 px-6 md:px-10 bg-black">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-6xl md:text-8xl font-black mb-8 text-white leading-none">START<br/>DRAWING<br/>TODAY</h2>
          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <button className="group relative overflow-hidden px-12 py-6 bg-[#FF1744] text-white font-black text-xl border-4 border-black transition-all duration-300 ease-out hover:scale-105 hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] active:scale-95">
              <span className="relative z-10">GET STARTED</span>
              <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
            </button>
            <button className="group relative overflow-hidden px-12 py-6 bg-white text-black font-black text-xl border-4 border-black transition-all duration-300 ease-out hover:scale-105 hover:bg-gray-100 hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,0.3)] active:scale-95">
              <span className="relative z-10">SEE DEMO</span>
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

function WaveDivider({ topColor, waveColor, reverse }: WaveDividerProps) {
  return (
    <div className={`relative ${topColor} overflow-hidden`}>
      <div className={`wave-divider h-48 relative`} style={{ '--wave-color': waveColor, '--wave-reverse': reverse ? '1' : '0' } as React.CSSProperties & { '--wave-color': string; '--wave-reverse': string }}>
        <div className="wave-container absolute inset-0">
          <div className={`wave-layer ${reverse ? 'wave-layer-reverse' : ''}`}></div>
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: { icon: React.ReactNode; title: string; description: string; color: string }) {
  return (
    <div className={`group ${color} p-8 border-4 border-black transition-all duration-500 ease-out hover:translate-y-[-8px] hover:translate-x-[-2px] shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] cursor-pointer`}>
      <div className="mb-4 transform transition-transform duration-500 ease-out group-hover:scale-110 group-hover:rotate-3">
        {icon}
      </div>
      <h3 className="text-2xl font-black mb-3 transition-all duration-300">{title}</h3>
      <p className="font-bold text-sm leading-relaxed opacity-90 group-hover:opacity-100 transition-opacity duration-300">{description}</p>
    </div>
  )
}

function UseCase({ title, description, color }: { title: string; description: string; color: string }) {
  return (
    <div className={`group ${color} p-8 text-center border-4 border-black transition-all duration-500 ease-out hover:translate-y-[-8px] hover:translate-x-[-2px] shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] cursor-pointer`}>
      <h3 className="text-2xl font-black mb-3 transition-transform duration-300 group-hover:scale-105">{title}</h3>
      <p className="font-bold text-sm opacity-90 group-hover:opacity-100 transition-opacity duration-300">{description}</p>
    </div>
  )
}
