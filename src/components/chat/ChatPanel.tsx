import React, { useState } from 'react';

interface Message { id: number; user: string; text: string; time: string; }

const ChatPanel: React.FC = () => {
  const [msgs, setMsgs] = useState<Message[]>([
    { id: 1, user: 'Nova', text: 'Let\'s block out the sky!', time: '09:01' },
    { id: 2, user: 'Pixel', text: 'I\'ll take bottom-left flora.', time: '09:02' },
  ]);
  const [input, setInput] = useState('');

  const send = () => {
    if (!input.trim()) return;
    setMsgs(m => [...m, { id: Date.now(), user: 'You', text: input.trim(), time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }]);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      <h2 className="text-xs font-semibold text-zinc-500 uppercase px-3 mb-2">Player Chat</h2>
      <div className="flex-1 overflow-y-auto px-3 py-1">
        <div className="flex flex-col gap-1.5">
          {msgs.map(m => (
            <div key={m.id} className="leading-snug">
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-semibold text-white">{m.user}</span>
                <span className="text-[11px] text-zinc-500">{m.time}</span>
              </div>
              <div className="text-sm text-zinc-100 whitespace-pre-wrap break-words">
                {m.text}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="px-3 pb-2 pt-2 border-t border-zinc-800 pr-2">
        <input
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>{ if(e.key==='Enter') send(); }}
          placeholder="Message players..."
          className="min-w-0 w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-green-500"
        />
      </div>
    </div>
  );
};

export default ChatPanel;
