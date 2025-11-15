import React from 'react'
import { Palette, Users, Zap, Shield, Gamepad2, Clock, Check, Wifi, WifiOff, PlayCircle } from 'lucide-react'

export default function App() {
  const heroVideoUrl = 'https://www.pexels.com/download/video/27936889/'

  return (
    <div className="relative min-h-screen bg-black text-white">
      {/* Hero Section with Background Image */}
      <header className="bg-black">
        <div className="relative h-full w-full overflow-hidden">
          <video
            src={heroVideoUrl}
            autoPlay
            loop
            muted
            playsInline
            className="absolute inset-0 h-full w-full object-cover"
            style={{
              maskImage: 'linear-gradient(to bottom, black 70%, transparent 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, black 70%, transparent 100%)',
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-black/85 via-black/40 to-transparent" />
          
          <div className="relative z-10 flex min-h-[420px] flex-col justify-between px-6 py-10 md:min-h-[520px] md:px-12 md:py-12">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <Palette className="w-8 h-8 text-green-500" />
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold uppercase tracking-wider">Real-Time</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">Collaborative</span>
                  <span className="text-zinc-400">/</span>
                  <span className="text-sm font-semibold uppercase tracking-wider">AI-Powered</span>
                </div>
              </div>
              <button className="bg-green-500 text-black font-bold py-2 px-6 rounded-lg text-sm hover:bg-green-400 transition-colors flex-shrink-0">
                GET STARTED
              </button>
            </div>

            <div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 bg-green-500/20 backdrop-blur-sm px-4 py-2 rounded-full border border-green-500/30">
                  <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
                  <span className="text-sm font-semibold text-green-400">PRODUCTION READY</span>
                </div>
                <p className="text-sm text-zinc-300">
                  Built with Node.js, React & TensorFlow Lite
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="px-6 pb-12 pt-8 md:px-12">
          <h1 className="text-5xl md:text-7xl font-black leading-tight mb-6">
            DoodleParty
          </h1>
          <p className="text-xl md:text-2xl text-zinc-300 mb-8 max-w-4xl">
            Transform passive audiences into active participants through collaborative art. Real-time drawing platform with AI-powered content moderation for events, concerts, festivals, and public gatherings.
          </p>
          <div className="flex flex-wrap gap-8">
            <div>
              <p className="text-sm text-zinc-400">Concurrent Users</p>
              <p className="font-bold text-xl">100+ (RPi4) | 2,000+ (Cloud)</p>
            </div>
            <div>
              <p className="text-sm text-zinc-400">ML Inference</p>
              <p className="font-bold text-xl">&lt;50ms on RPi4</p>
            </div>
            <div>
              <p className="text-sm text-zinc-400">WebSocket Latency</p>
              <p className="font-bold text-xl">&lt;50ms (p95)</p>
            </div>
            <div>
              <p className="text-sm text-zinc-400">License</p>
              <p className="font-bold text-xl">GNU GPL</p>
            </div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-14 px-6 md:px-12 bg-black">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12">
            <h2 className="text-4xl font-black mb-2">Why DoodleParty?</h2>
            <p className="text-zinc-400">Built for performance, scale, and community</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard
              icon={<Users className="w-8 h-8 text-green-500" />}
              title="100+ Concurrent Users"
              description="Support up to 100 users on Raspberry Pi 4, or scale to 2,000+ in the cloud"
            />
            <FeatureCard
              icon={<Zap className="w-8 h-8 text-green-500" />}
              title="Real-Time Collaboration"
              description="<50ms latency for smooth, synchronized drawing experience across all devices"
            />
            <FeatureCard
              icon={<Shield className="w-8 h-8 text-green-500" />}
              title="AI-Powered Moderation"
              description="Custom TensorFlow Lite binary classifier with <50ms inference on edge devices"
            />
            <FeatureCard
              icon={<Gamepad2 className="w-8 h-8 text-green-500" />}
              title="Multiple Game Modes"
              description="Speed Sketch, Guess The Doodle, Battle Royale, Story Canvas, and more"
            />
            <FeatureCard
              icon={<Clock className="w-8 h-8 text-green-500" />}
              title="Edge ML Inference"
              description="TFLite INT8 model runs at <50ms on Raspberry Pi 4 with 5MB footprint"
            />
            <FeatureCard
              icon={<WifiOff className="w-8 h-8 text-green-500" />}
              title="Offline Capable"
              description="Raspberry Pi deployment works completely offline with full data privacy"
            />
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="py-14 px-6 md:px-10 bg-black">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-5xl font-extrabold mb-4">Find a plan to power your creativity</h2>
            <p className="text-lg text-zinc-400">
              Whether you're running a community event or deploying at scale, we have a plan that's right for you.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[
              { 
                plan: 'Hobby', 
                price: 'Free', 
                description: 'For individuals and small groups.', 
                features: ['Up to 10 concurrent users', 'Classic Canvas mode', 'Basic AI moderation', 'Community support'] 
              },
              { 
                plan: 'Raspberry Pi 4', 
                price: '$499', 
                description: 'Self-contained hardware for events.', 
                features: ['100 concurrent users', 'All game modes included', 'Complete offline operation', 'No internet required', 'Full data privacy', 'One-time purchase'], 
                featured: true 
              },
              { 
                plan: 'Cloud Pro', 
                price: '$199', 
                description: 'Fully managed cloud solution.', 
                features: ['Up to 500 concurrent users', 'Auto-scaling infrastructure', '99.9% uptime SLA', 'Full AI features & LLM', '24/7 support', 'Analytics dashboard'],
                perMonth: true
              },
            ].map((tier) => (
              <div
                key={tier.plan}
                className={`border rounded-xl p-8 flex flex-col ${
                  tier.featured ? 'border-green-500 bg-zinc-800' : 'border-zinc-700 bg-zinc-800/50'
                }`}
              >
                <h3 className="text-xl font-bold text-white">{tier.plan}</h3>
                <p className="mt-2 text-zinc-400">{tier.description}</p>
                <div className="mt-6">
                  <span className="text-5xl font-extrabold text-white">{tier.price}</span>
                  {tier.perMonth && <span className="text-zinc-400">/ month</span>}
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
                  {tier.plan === 'Hobby' ? 'Get Started Free' : tier.featured ? 'Order Hardware' : 'Start Free Trial'}
                </button>
              </div>
            ))}
          </div>
          <div className="mt-8 text-center text-zinc-400 text-sm">
            <p>Enterprise plans available: <strong className="text-white">Pro ($399/mo)</strong> up to 1,000 users | <strong className="text-white">Enterprise ($799/mo)</strong> up to 2,000+ users</p>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-14 px-6 md:px-10 bg-black">
        <div className="max-w-7xl mx-auto">
          <div className="mb-10">
            <h2 className="text-4xl font-black mb-2">Perfect For</h2>
            <p className="text-zinc-400">Bring communities together through creative expression</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <UseCase
              title="Music Festivals"
              description="Create shared visual experiences during performances"
            />
            <UseCase
              title="Conferences"
              description="Interactive networking and icebreaker activities"
            />
            <UseCase
              title="Community Events"
              description="Bring neighborhoods together through collaborative art"
            />
            <UseCase
              title="Corporate Events"
              description="Team building and creative workshops"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 md:px-10 bg-black">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-black mb-6">Ready to Transform Your Events?</h2>
          <p className="text-xl text-zinc-300 mb-8">
            Join the creative revolution. Bring your community together through collaborative art and shared experiences.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="px-8 py-4 bg-green-500 text-black hover:bg-green-400 rounded-lg font-semibold text-lg transition-colors">
              Start Free Trial
            </button>
            <button className="px-8 py-4 bg-transparent border-2 border-zinc-700 hover:bg-zinc-800 rounded-lg font-semibold text-lg transition-colors">
              Schedule Demo
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 md:px-10 bg-black border-t border-zinc-800">
        <div className="max-w-7xl mx-auto text-center text-zinc-400">
          <p className="mb-4 font-semibold">
            DoodleParty - Democratizing creativity at scale
          </p>
          <p className="text-sm">
            Open source under GNU General Public License | Built with Node.js, React & TensorFlow Lite
          </p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-zinc-800 rounded-xl p-6 border border-zinc-700 hover:border-green-500 transition-all duration-300">
      <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-zinc-400">{description}</p>
    </div>
  )
}

function UseCase({ title, description }: { title: string; description: string }) {
  return (
    <div className="bg-zinc-800 rounded-xl p-6 text-center border border-zinc-700 hover:border-zinc-600 transition-all">
      <h3 className="text-lg font-bold mb-2">{title}</h3>
      <p className="text-sm text-zinc-400">{description}</p>
    </div>
  )
}
