import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import type { DrawData } from '../types';

interface SharedCanvasContextType {
  paths: DrawData[];
  addPath: (path: DrawData) => void;
  clearPaths: () => void;
  isConnected: boolean;
}

const SharedCanvasContext = createContext<SharedCanvasContextType | undefined>(undefined);

const SERVER_URL = import.meta.env.VITE_SERVER_URL || 'http://localhost:3001';
const SESSION_ID = 'default-session'; // In production, this would be dynamic

export const SharedCanvasProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [paths, setPaths] = useState<DrawData[]>([]);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize Socket.io connection
    const socketInstance = io(SERVER_URL, {
      transports: ['websocket', 'polling'],
    });

    socketInstance.on('connect', () => {
      console.log('✅ Connected to DoodleParty server');
      setIsConnected(true);
      
      // Create/join default session
      fetch(`${SERVER_URL}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Default Session', maxPlayers: 100, mode: 'classic' }),
      })
        .then(res => res.json())
        .then(data => {
          // Join the session
          socketInstance.emit('join-session', {
            sessionId: SESSION_ID,
            username: 'User' + Math.floor(Math.random() * 1000),
          });
        })
        .catch(() => {
          // Session might already exist, just join it
          socketInstance.emit('join-session', {
            sessionId: SESSION_ID,
            username: 'User' + Math.floor(Math.random() * 1000),
          });
        });
    });

    socketInstance.on('disconnect', () => {
      console.log('❌ Disconnected from server');
      setIsConnected(false);
    });

    // Receive initial canvas state
    socketInstance.on('canvas-state', (state: { paths: DrawData[] }) => {
      console.log('📥 Received canvas state:', state.paths.length, 'paths');
      setPaths(state.paths);
    });

    // Receive new drawing updates
    socketInstance.on('canvas-update', (data: DrawData) => {
      console.log('📥 Received canvas update');
      setPaths(prev => [...prev, data]);
    });

    // Receive full canvas state (after undo)
    socketInstance.on('canvas-full-state', (state: { paths: DrawData[] }) => {
      console.log('📥 Received full canvas state');
      setPaths(state.paths);
    });

    // Canvas cleared
    socketInstance.on('canvas-cleared', () => {
      console.log('🗑️ Canvas cleared');
      setPaths([]);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  const addPath = (path: DrawData) => {
    console.log('🎨 Adding path locally:', path.path.length, 'points');
    // Optimistically add to local state
    setPaths(prev => [...prev, path]);
    
    // Send to server
    if (socket && isConnected) {
      console.log('📤 Emitting draw event to server');
      socket.emit('draw', path);
    } else {
      console.warn('❌ Cannot send - socket not connected. Connected:', isConnected);
    }
  };

  const clearPaths = () => {
    setPaths([]);
    if (socket && isConnected) {
      socket.emit('clear-canvas');
    }
  };

  return (
    <SharedCanvasContext.Provider value={{ paths, addPath, clearPaths, isConnected }}>
      {children}
    </SharedCanvasContext.Provider>
  );
};

export const useSharedCanvas = () => {
  const context = useContext(SharedCanvasContext);
  if (!context) {
    throw new Error('useSharedCanvas must be used within SharedCanvasProvider');
  }
  return context;
};
