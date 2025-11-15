import React from 'react'
import { Palette, Users, Zap, Shield, Gamepad2, Clock, WifiOff, Sparkles } from 'lucide-react'

export default function App() {
  return (
    <div className="relative min-h-screen bg-white text-gray-900">
      {/* Hero Section */}
      <header className="bg-gradient-to-b from-yellow-50 to-white">
        <div className="container mx-auto px-6 py-16 text-center">
          <div className="flex items-center justify-center mb-6">
            <Sparkles className="w-12 h-12 text-yellow-500 animate-pulse" />
          </div>
          
          <h1 className="text-6xl md:text-8xl font-black leading-tight mb-6 text-gray-900">
            DoodleParty 🎨
          </h1>
          
          <p className="text-2xl md:text-3xl font-bold text-gray-700 mb-4 max-w-3xl mx-auto">
            Draw together. Create together. Celebrate together.
          </p>
          
          <p className="text-lg md:text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Turn your events into collaborative art experiences! Real-time drawing for concerts, festivals, and gatherings with built-in AI moderation.
          </p>
          
          <button className="bg-yellow-400 hover:bg-yellow-500 text-gray-900 font-bold text-lg px-12 py-5 rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all">
            Let's Draw! 🖍️
          </button>
          
          <div className="mt-12 flex flex-wrap justify-center gap-6 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
              <span>100+ players at once</span>
            </div>
            <div>•</div>
            <div>Real-time sync</div>
            <div>•</div>
            <div>AI-powered moderation</div>
            <div>•</div>
            <div>Works offline</div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-20 px-6 md:px-12 bg-blue-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-black mb-4 text-gray-900">Why DoodleParty?</h2>
            <p className="text-xl text-gray-600">Built for performance, scale, and community fun!</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Users className="w-10 h-10 text-purple-600" />}
              title="100+ Players Together"
              description="Support up to 100 users on Raspberry Pi 4, or scale to 2,000+ in the cloud"
            />
            <FeatureCard
              icon={<Zap className="w-10 h-10 text-yellow-600" />}
              title="Lightning Fast"
              description="<50ms latency for smooth, synchronized drawing experience across all devices"
            />
            <FeatureCard
              icon={<Shield className="w-10 h-10 text-blue-600" />}
              title="AI-Powered Safety"
              description="Keep your canvas safe with smart moderation that runs in <50ms"
            />
            <FeatureCard
              icon={<Gamepad2 className="w-10 h-10 text-pink-600" />}
              title="Multiple Game Modes"
              description="Speed Sketch, Guess The Doodle, Battle Royale, Story Canvas, and more!"
            />
            <FeatureCard
              icon={<Clock className="w-10 h-10 text-green-600" />}
              title="Edge ML Magic"
              description="Tiny 5MB AI model runs super fast on Raspberry Pi 4"
            />
            <FeatureCard
              icon={<WifiOff className="w-10 h-10 text-orange-600" />}
              title="Works Offline"
              description="Perfect for events - no internet needed with Raspberry Pi deployment"
            />
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="py-20 px-6 md:px-10 bg-gradient-to-b from-white to-purple-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-black mb-4 text-gray-900">Pick Your Perfect Plan! 🎯</h2>
            <p className="text-xl text-gray-600">
              From hobby projects to massive events, we've got you covered
            </p>
          </div>
          <div className="flex flex-col lg:flex-row gap-8 max-w-6xl mx-auto justify-center">
            {[
              { 
                plan: 'Hobby', 
                price: 'Free', 
                emoji: '🎨',
                description: 'Perfect for trying things out!', 
                features: ['Up to 10 players', 'Classic Canvas mode', 'AI moderation', 'Community support'] 
              },
              { 
                plan: 'Raspberry Pi 4', 
                price: '$499', 
                emoji: '🍓',
                description: 'Hardware box for events', 
                features: ['100 concurrent users', 'All game modes', 'Works offline', 'No internet needed', 'Full privacy', 'One-time purchase'], 
                featured: true 
              },
              { 
                plan: 'Cloud Pro', 
                price: '$199', 
                emoji: '☁️',
                description: 'Managed cloud hosting', 
                features: ['Up to 500 players', 'Auto-scaling', '99.9% uptime', 'AI features & LLM', '24/7 support', 'Analytics'],
                perMonth: true
              },
            ].map((tier) => (
              <div
                key={tier.plan}
                className={`rounded-3xl p-8 flex flex-col shadow-lg transition-all hover:scale-105 ${
                  tier.featured ? 'bg-gradient-to-br from-yellow-300 to-yellow-400 border-4 border-yellow-500' : 'bg-white border-2 border-gray-200'
                }`}
              >
                <div className="text-4xl mb-3">{tier.emoji}</div>
                <h3 className="text-2xl font-black text-gray-900">{tier.plan}</h3>
                <p className="mt-2 text-gray-700">{tier.description}</p>
                <div className="mt-6 mb-8">
                  <span className="text-6xl font-black text-gray-900">{tier.price}</span>
                  {tier.perMonth && <span className="text-gray-700 text-xl">/mo</span>}
                </div>
                <ul className="space-y-3 text-gray-800 flex-grow mb-8">
                  {tier.features.map((f, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-green-600 font-bold">✓</span>
                      <span className="font-medium">{f}</span>
                    </li>
                  ))}
                </ul>
                <button className={`w-full py-4 font-bold rounded-full transition-all transform hover:scale-105 shadow-md ${
                  tier.featured ? 'bg-gray-900 text-white hover:bg-gray-800' : 'bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600'
                }`}>
                  {tier.plan === 'Hobby' ? 'Start Free!' : tier.featured ? 'Order Now' : 'Try Free'}
                </button>
              </div>
            ))}
          </div>
          <div className="mt-12 text-center text-gray-600">
            <p className="text-lg">Need more? <strong className="text-gray-900">Pro ($399/mo)</strong> • <strong className="text-gray-900">Enterprise ($799/mo)</strong> • Custom solutions available!</p>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-20 px-6 md:px-10 bg-gradient-to-b from-green-50 to-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-black mb-4 text-gray-900">Perfect For... 🎪</h2>
            <p className="text-xl text-gray-600">Bring communities together through creative fun!</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <UseCase
              emoji="🎵"
              title="Music Festivals"
              description="Create shared visual experiences during performances"
            />
            <UseCase
              emoji="💼"
              title="Conferences"
              description="Interactive networking and icebreaker activities"
            />
            <UseCase
              emoji="🎉"
              title="Community Events"
              description="Bring neighborhoods together through collaborative art"
            />
            <UseCase
              emoji="🏢"
              title="Corporate Events"
              description="Team building and creative workshops"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6 md:px-10 bg-gradient-to-br from-purple-100 via-pink-100 to-yellow-100">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl md:text-6xl font-black mb-6 text-gray-900">Ready to Create Something Amazing?</h2>
          <p className="text-2xl text-gray-700 mb-10">
            Join thousands of creative communities bringing people together through art! 🎨✨
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="px-10 py-5 bg-yellow-400 hover:bg-yellow-500 text-gray-900 font-black text-xl rounded-full shadow-lg transform hover:scale-105 transition-all">
              Start Drawing Now!
            </button>
            <button className="px-10 py-5 bg-white hover:bg-gray-50 text-gray-900 font-bold text-xl rounded-full border-4 border-gray-900 shadow-lg transition-all">
              See a Demo
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 md:px-10 bg-gradient-to-r from-purple-50 via-pink-50 to-yellow-50 border-t-4 border-yellow-400">
        <div className="max-w-7xl mx-auto text-center">
          <p className="mb-3 font-black text-2xl text-gray-900">
            DoodleParty - Where creativity meets community! 🎨
          </p>
          <p className="text-sm text-gray-600 font-medium">
            Open source under GNU GPL • Built with Node.js, React & TensorFlow Lite
          </p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-white rounded-3xl p-8 border-2 border-gray-200 hover:border-blue-400 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
      <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-2xl font-black mb-3 text-gray-900">{title}</h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
  )
}

function UseCase({ emoji, title, description }: { emoji: string; title: string; description: string }) {
  return (
    <div className="bg-white rounded-2xl p-8 text-center border-2 border-gray-200 hover:border-purple-400 hover:shadow-lg transition-all transform hover:scale-105">
      <div className="text-5xl mb-4">{emoji}</div>
      <h3 className="text-xl font-black mb-2 text-gray-900">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  )
}
