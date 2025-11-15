import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: process.env.VITE_CLIENT_URL || 'http://localhost:5173',
    methods: ['GET', 'POST'],
  },
});

// Middleware
app.use(cors());
app.use(express.json());

// Store active sessions and canvas state
const sessions = new Map();
const canvasStates = new Map();

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/sessions', (req, res) => {
  const sessionList = Array.from(sessions.entries()).map(([id, data]) => ({
    id,
    name: data.name,
    players: data.players.size,
    maxPlayers: data.maxPlayers,
    mode: data.mode,
  }));
  res.json({ sessions: sessionList });
});

app.post('/api/sessions', (req, res) => {
  const { name, maxPlayers = 16, mode = 'classic' } = req.body;
  const sessionId = Math.random().toString(36).substring(2, 9);
  
  sessions.set(sessionId, {
    id: sessionId,
    name,
    maxPlayers,
    mode,
    players: new Set(),
    createdAt: new Date(),
  });
  
  canvasStates.set(sessionId, {
    paths: [],
  });
  
  res.json({ sessionId, message: 'Session created successfully' });
});

// Socket.io real-time events
io.on('connection', (socket) => {
  console.log(`User connected: ${socket.id}`);
  
  // Join a session
  socket.on('join-session', (data) => {
    const { sessionId, username } = data;
    const session = sessions.get(sessionId);
    
    if (!session) {
      socket.emit('error', { message: 'Session not found' });
      return;
    }
    
    if (session.players.size >= session.maxPlayers) {
      socket.emit('error', { message: 'Session is full' });
      return;
    }
    
    socket.join(sessionId);
    session.players.add(socket.id);
    socket.data.sessionId = sessionId;
    socket.data.username = username;
    
    // Send current canvas state to the new user
    const canvasState = canvasStates.get(sessionId);
    socket.emit('canvas-state', canvasState);
    
    // Notify others
    socket.to(sessionId).emit('player-joined', {
      playerId: socket.id,
      username,
      playerCount: session.players.size,
    });
    
    console.log(`${username} joined session ${sessionId}`);
  });
  
  // Handle drawing strokes
  socket.on('draw', (data) => {
    const { sessionId } = socket.data;
    if (!sessionId) return;
    
    const canvasState = canvasStates.get(sessionId);
    if (canvasState) {
      canvasState.paths.push(data);
      
      // Broadcast to all users in the session (including sender for projection sync)
      io.to(sessionId).emit('canvas-update', data);
    }
  });
  
  // Handle chat messages
  socket.on('chat-message', (data) => {
    const { sessionId, username } = socket.data;
    if (!sessionId) return;
    
    const message = {
      id: Date.now(),
      username: username || 'Anonymous',
      message: data.message,
      timestamp: new Date().toISOString(),
    };
    
    io.to(sessionId).emit('chat-message', message);
  });
  
  // Handle undo
  socket.on('undo', () => {
    const { sessionId } = socket.data;
    if (!sessionId) return;
    
    const canvasState = canvasStates.get(sessionId);
    if (canvasState && canvasState.paths.length > 0) {
      canvasState.paths.pop();
      io.to(sessionId).emit('canvas-full-state', canvasState);
    }
  });
  
  // Handle clear canvas
  socket.on('clear-canvas', () => {
    const { sessionId } = socket.data;
    if (!sessionId) return;
    
    const canvasState = canvasStates.get(sessionId);
    if (canvasState) {
      canvasState.paths = [];
      io.to(sessionId).emit('canvas-cleared');
    }
  });
  
  // Handle disconnection
  socket.on('disconnect', () => {
    const { sessionId, username } = socket.data;
    
    if (sessionId) {
      const session = sessions.get(sessionId);
      if (session) {
        session.players.delete(socket.id);
        
        socket.to(sessionId).emit('player-left', {
          playerId: socket.id,
          username,
          playerCount: session.players.size,
        });
        
        // Clean up empty sessions
        if (session.players.size === 0) {
          sessions.delete(sessionId);
          canvasStates.delete(sessionId);
          console.log(`Session ${sessionId} deleted (empty)`);
        }
      }
    }
    
    console.log(`User disconnected: ${socket.id}`);
  });
});

// Start server
const PORT = process.env.PORT || 3001;
httpServer.listen(PORT, () => {
  console.log(`🚀 DoodleParty server running on port ${PORT}`);
  console.log(`📡 WebSocket server ready for connections`);
  console.log(`🎨 Canvas synchronization active`);
});

export { app, httpServer, io };
