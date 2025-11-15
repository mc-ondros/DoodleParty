import React, { useState } from 'react';
import { UserIcon, NotificationsIcon, PaletteIcon, ShieldIcon, CreditCardIcon, LockIcon, LockOpenIcon, MonitorIcon } from '../../constants';
import DrawingCanvas from '../canvas/DrawingCanvas';
import Toolbar from '../canvas/Toolbar';
import ChatPanel from '../chat/ChatPanel';
import { useSharedCanvas } from '../../context/SharedCanvasContext';

const ProjectionView: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  // Generate QR code URL for the current location
  const url = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:5173';
  const qrCodeUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(url)}`;

  const { paths } = useSharedCanvas();
  const [selectedColor] = useState('#000000');
  const [strokeSize] = useState(5);

  return (
    <div className="fixed inset-0 bg-black z-50 flex">
      {/* Chat sidebar on left - read only */}
      <div className="w-80 bg-zinc-900 border-r border-zinc-800 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0">
          <ChatPanel />
        </div>
      </div>

      {/* Canvas area - projection only */}
      <div className="flex-1 relative flex flex-col">
        {/* Canvas */}
        <div className="flex-1 relative bg-zinc-900">
          <DrawingCanvas
            paths={paths}
            color={selectedColor}
            tool="brush"
            strokeWeight={strokeSize}
            onDraw={() => {}}
            isReadOnly={true}
          />
          
          {/* QR Code in bottom right */}
          <div className="absolute bottom-8 right-8 bg-white p-4 rounded-lg shadow-2xl">
            <img src={qrCodeUrl} alt="QR Code" className="w-48 h-48" />
            <p className="text-center text-sm text-black font-semibold mt-2">Scan to Join</p>
          </div>
        </div>

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 bg-zinc-800 hover:bg-zinc-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors z-10"
        >
          Exit Projection
        </button>
      </div>
    </div>
  );
};

type SettingsCategory = 'profile' | 'notifications' | 'appearance' | 'privacy' | 'billing' | 'admin';

interface SettingsCategoryItem {
  id: SettingsCategory;
  label: string;
  icon: React.ElementType;
  adminOnly?: boolean;
}

const SettingsPage: React.FC<{ isAdmin?: boolean; activeCategory?: string; setActiveCategory?: (cat: string) => void }> = ({ isAdmin = true, activeCategory: externalCategory, setActiveCategory: externalSetter }) => {
  const [internalCategory, setInternalCategory] = useState<SettingsCategory>('profile');
  const activeCategory = (externalCategory || internalCategory) as SettingsCategory;
  const setActiveCategory = externalSetter || setInternalCategory;

  const categories: SettingsCategoryItem[] = [
    { id: 'profile', label: 'Profile', icon: UserIcon },
    { id: 'notifications', label: 'Notifications', icon: NotificationsIcon },
    { id: 'appearance', label: 'Appearance', icon: PaletteIcon },
    { id: 'privacy', label: 'Privacy & Security', icon: ShieldIcon },
    { id: 'billing', label: 'Billing', icon: CreditCardIcon },
    { id: 'admin', label: 'Admin Dashboard', icon: ShieldIcon, adminOnly: true },
  ];

  const visibleCategories = categories.filter(cat => !cat.adminOnly || isAdmin);

  return (
    <div className="flex-1 overflow-y-auto bg-zinc-900">
      <div className="max-w-4xl mx-auto p-10">
        {activeCategory === 'profile' && <ProfileSettings />}
        {activeCategory === 'notifications' && <NotificationSettings />}
        {activeCategory === 'appearance' && <AppearanceSettings />}
        {activeCategory === 'privacy' && <PrivacySettings />}
        {activeCategory === 'billing' && <BillingSettings />}
        {activeCategory === 'admin' && isAdmin && <AdminDashboard />}
      </div>
    </div>
  );
};

const ProfileSettings: React.FC = () => (
  <div>
    <h1 className="text-3xl font-bold mb-2">Profile Settings</h1>
    <p className="text-zinc-400 mb-8">Manage your account information and public profile</p>
    
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-semibold mb-2">Profile Picture</label>
        <div className="flex items-center space-x-4">
          <img src="https://picsum.photos/seed/enrique/80/80" alt="Profile" className="w-20 h-20 rounded-full" />
          <button className="bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded-lg transition-colors">Change Photo</button>
        </div>
      </div>

      <div>
        <label className="block text-sm font-semibold mb-2">Username</label>
        <input type="text" defaultValue="EnriqueP" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
      </div>

      <div>
        <label className="block text-sm font-semibold mb-2">Email</label>
        <input type="email" defaultValue="enrique@example.com" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
      </div>

      <div>
        <label className="block text-sm font-semibold mb-2">Bio</label>
        <textarea rows={4} placeholder="Tell us about yourself..." className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
      </div>

      <button className="bg-green-500 hover:bg-green-600 text-black font-semibold px-6 py-3 rounded-lg transition-colors">
        Save Changes
      </button>
    </div>
  </div>
);

const NotificationSettings: React.FC = () => (
  <div>
    <h1 className="text-3xl font-bold mb-2">Notification Settings</h1>
    <p className="text-zinc-400 mb-8">Choose what notifications you want to receive</p>
    
    <div className="space-y-4">
      {[
        { label: 'Canvas Activity', desc: 'Get notified when someone interacts with your drawings' },
        { label: 'Community Updates', desc: 'Stay updated on community events and announcements' },
        { label: 'New Followers', desc: 'Be notified when someone follows you' },
        { label: 'Weekly Digest', desc: 'Receive a weekly summary of platform activity' },
      ].map((item) => (
        <div key={item.label} className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg">
          <div>
            <p className="font-semibold">{item.label}</p>
            <p className="text-sm text-zinc-400">{item.desc}</p>
          </div>
          <input type="checkbox" className="w-5 h-5" defaultChecked />
        </div>
      ))}
    </div>
  </div>
);

const AppearanceSettings: React.FC = () => (
  <div>
    <h1 className="text-3xl font-bold mb-2">Appearance</h1>
    <p className="text-zinc-400 mb-8">Customize how DoodleParty looks for you</p>
    
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-semibold mb-3">Theme</label>
        <div className="grid grid-cols-3 gap-4">
          {['Dark', 'Light', 'System'].map((theme) => (
            <button key={theme} className={`p-4 border rounded-lg transition-colors ${theme === 'Dark' ? 'border-green-500 bg-green-500/10' : 'border-zinc-700 hover:border-zinc-600'}`}>
              {theme}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-semibold mb-3">Accent Color</label>
        <div className="flex space-x-3">
          {['#22c55e', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6'].map((color) => (
            <button key={color} className="w-10 h-10 rounded-full border-2 border-zinc-700 hover:scale-110 transition-transform" style={{ backgroundColor: color }} />
          ))}
        </div>
      </div>
    </div>
  </div>
);

const PrivacySettings: React.FC = () => (
  <div>
    <h1 className="text-3xl font-bold mb-2">Privacy & Security</h1>
    <p className="text-zinc-400 mb-8">Manage your privacy and security settings</p>
    
    <div className="space-y-4">
      {[
        { label: 'Make Profile Public', desc: 'Allow anyone to view your profile and artworks' },
        { label: 'Allow Comments', desc: 'Let others comment on your drawings' },
        { label: 'Show Online Status', desc: 'Display when you are active on the platform' },
      ].map((item) => (
        <div key={item.label} className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg">
          <div>
            <p className="font-semibold">{item.label}</p>
            <p className="text-sm text-zinc-400">{item.desc}</p>
          </div>
          <input type="checkbox" className="w-5 h-5" defaultChecked />
        </div>
      ))}

      <div className="mt-8 pt-8 border-t border-zinc-800">
        <h3 className="text-xl font-bold mb-4">Change Password</h3>
        <div className="space-y-4">
          <input type="password" placeholder="Current Password" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
          <input type="password" placeholder="New Password" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
          <input type="password" placeholder="Confirm New Password" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500" />
          <button className="bg-green-500 hover:bg-green-600 text-black font-semibold px-6 py-3 rounded-lg transition-colors">
            Update Password
          </button>
        </div>
      </div>
    </div>
  </div>
);

const BillingSettings: React.FC = () => (
  <div>
    <h1 className="text-3xl font-bold mb-2">Billing</h1>
    <p className="text-zinc-400 mb-8">Manage your subscription and payment methods</p>
    
    <div className="bg-zinc-800/50 p-6 rounded-lg mb-6">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold">Hobby Plan</h3>
          <p className="text-zinc-400">Free forever</p>
        </div>
        <span className="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-semibold">Active</span>
      </div>
      <button className="bg-green-500 hover:bg-green-600 text-black font-semibold px-6 py-3 rounded-lg transition-colors">
        Upgrade Plan
      </button>
    </div>

    <div>
      <h3 className="text-xl font-bold mb-4">Payment Methods</h3>
      <div className="bg-zinc-800/50 p-4 rounded-lg flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-12 h-8 bg-zinc-700 rounded mr-4"></div>
          <div>
            <p className="font-semibold">No payment methods</p>
            <p className="text-sm text-zinc-400">Add a payment method to upgrade</p>
          </div>
        </div>
        <button className="bg-zinc-700 hover:bg-zinc-600 px-4 py-2 rounded-lg transition-colors">Add Card</button>
      </div>
    </div>
  </div>
);

const AdminDashboard: React.FC = () => {
  const [sessionLocked, setSessionLocked] = useState(false);
  const [showProjection, setShowProjection] = useState(false);
  const [players] = useState([
    { id: '1', name: 'Alex', isHost: true, status: 'online' as const },
    { id: '2', name: 'Jamie', status: 'idle' as const },
    { id: '3', name: 'Riley', status: 'online' as const },
  ]);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">Admin Dashboard</h1>
      <p className="text-zinc-400 mb-8">Manage platform settings and monitor activity</p>
      
      <div className="flex justify-between items-center mb-6">
        <div></div>
        <button
          onClick={() => setShowProjection(!showProjection)}
          className="flex items-center space-x-2 bg-green-500 hover:bg-green-600 text-black font-semibold px-6 py-3 rounded-lg transition-colors"
        >
          <MonitorIcon className="w-5 h-5" />
          <span>{showProjection ? 'Exit Projection' : 'Project Canvas'}</span>
        </button>
      </div>

      {showProjection && <ProjectionView onClose={() => setShowProjection(false)} />}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {[
          { label: 'Total Users', value: '12,345', change: '+12%' },
          { label: 'Active Sessions', value: '234', change: '+5%' },
          { label: 'Drawings Today', value: '1,567', change: '+23%' },
        ].map((stat) => (
          <div key={stat.label} className="bg-zinc-800/50 p-6 rounded-lg border border-zinc-700">
            <p className="text-sm text-zinc-400 mb-1">{stat.label}</p>
            <div className="flex items-baseline justify-between">
              <p className="text-3xl font-bold">{stat.value}</p>
              <span className="text-green-400 text-sm font-semibold">{stat.change}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-6">
        <div>
          <h3 className="text-xl font-bold mb-4">Game Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700">
              <label className="block text-sm font-semibold mb-2 text-zinc-400">Max Players</label>
              <select defaultValue="16" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500">
                <option>4</option>
                <option>8</option>
                <option>16</option>
                <option>32</option>
                <option>Unlimited</option>
              </select>
            </div>
            <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700">
              <label className="block text-sm font-semibold mb-2 text-zinc-400">Ink Limit</label>
              <select defaultValue="Medium" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500">
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
                <option>Unlimited</option>
              </select>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xl font-bold mb-4">Moderation Settings</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700">
              <label className="block text-sm font-semibold mb-2 text-zinc-400">AI Strictness</label>
              <select defaultValue="Normal" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500">
                <option>Relaxed</option>
                <option>Normal</option>
                <option>Strict</option>
              </select>
            </div>
            <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700">
              <label className="block text-sm font-semibold mb-2 text-zinc-400">Auto-kick After Violations</label>
              <select defaultValue="5" className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500">
                <option>Disabled</option>
                <option>3</option>
                <option>5</option>
                <option>10</option>
              </select>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xl font-bold mb-4">Active Players</h3>
          <div className="bg-zinc-800/50 rounded-lg border border-zinc-700 p-4">
            <div className="space-y-3">
              {players.map(p => (
                <div key={p.id} className="flex items-center justify-between p-3 bg-zinc-800 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <span className={`w-2 h-2 rounded-full ${p.status === 'online' ? 'bg-green-500' : 'bg-zinc-500'}`}></span>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold">{p.name}</span>
                        {p.isHost && <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded-full">HOST</span>}
                      </div>
                      <span className="text-xs text-zinc-400">{p.status}</span>
                    </div>
                  </div>
                  <button className="bg-red-500/20 text-red-400 px-3 py-1 rounded text-sm font-semibold hover:bg-red-500/30">Kick</button>
                </div>
              ))}
            </div>
            <button
              onClick={() => setSessionLocked(!sessionLocked)}
              className={`mt-4 w-full py-2 rounded-lg font-semibold transition-colors flex items-center justify-center space-x-2 ${
                sessionLocked ? 'bg-zinc-700 text-white hover:bg-zinc-600' : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              }`}
            >
              {sessionLocked ? (
                <>
                  <LockIcon className="w-4 h-4" />
                  <span>Unlock Session</span>
                </>
              ) : (
                <>
                  <LockOpenIcon className="w-4 h-4" />
                  <span>Lock Session</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
