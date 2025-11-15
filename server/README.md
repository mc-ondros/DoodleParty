# DoodleParty Express Server

Real-time collaborative drawing server built with Express and Socket.io.

## Features

- **Express HTTP Server** - REST API for session management
- **Socket.io WebSocket Engine** - Real-time canvas synchronization
- **Session Management** - Multi-room support with player limits
- **Canvas State Persistence** - Server-side canvas state storage
- **Player Management** - Join/leave notifications and player tracking
- **Chat System** - Real-time chat with message broadcasting

## Quick Start

### Start the server

```bash
npm run server
```

The server will start on `http://localhost:3001`

### Development mode (with auto-reload)

```bash
npm run server:dev
```

## API Endpoints

### Health Check
```
GET /api/health
Response: { status: "ok", timestamp: "..." }
```

### List Sessions
```
GET /api/sessions
Response: {
  sessions: [
    {
      id: "abc123",
      name: "My Session",
      players: 5,
      maxPlayers: 16,
      mode: "classic"
    }
  ]
}
```

### Create Session
```
POST /api/sessions
Body: {
  name: "Session Name",
  maxPlayers: 16,
  mode: "classic"
}
Response: {
  sessionId: "abc123",
  message: "Session created successfully"
}
```

## Socket.io Events

### Client → Server

- `join-session` - Join a drawing session
  ```js
  socket.emit('join-session', { sessionId, username });
  ```

- `draw` - Send drawing stroke
  ```js
  socket.emit('draw', { path, color, strokeWeight, isFill });
  ```

- `chat-message` - Send chat message
  ```js
  socket.emit('chat-message', { message: "Hello!" });
  ```

- `undo` - Undo last drawing stroke
  ```js
  socket.emit('undo');
  ```

- `clear-canvas` - Clear entire canvas
  ```js
  socket.emit('clear-canvas');
  ```

### Server → Client

- `canvas-state` - Initial canvas state on join
- `canvas-update` - New drawing stroke broadcast
- `canvas-full-state` - Complete canvas state (after undo)
- `canvas-cleared` - Canvas was cleared
- `chat-message` - New chat message
- `player-joined` - Player joined notification
- `player-left` - Player left notification
- `error` - Error message

## Configuration

Environment variables:

- `PORT` - Server port (default: 3001)
- `VITE_CLIENT_URL` - Client URL for CORS (default: http://localhost:5173)

## Architecture

```
server/
├── index.js          # Main server file
└── README.md         # This file
```

The server maintains:
- **sessions** Map - Active drawing sessions
- **canvasStates** Map - Canvas paths for each session

## Connection Flow

1. Client connects via Socket.io
2. Client emits `join-session` with sessionId and username
3. Server validates and adds player to session
4. Server sends current `canvas-state` to new player
5. Server broadcasts `player-joined` to other players
6. Client can now send `draw` events
7. Server broadcasts `canvas-update` to all players in session

## Production Deployment

For production, consider:
- Using Redis for session/state persistence
- Implementing authentication/authorization
- Adding rate limiting
- Enabling WebSocket compression
- Using a process manager (PM2, systemd)
- Setting up SSL/TLS for WSS (secure WebSocket)

## Related Documentation

- [Architecture](../.documentation/architecture.md) - System design overview
- [API Reference](../.documentation/api.md) - Complete API documentation
