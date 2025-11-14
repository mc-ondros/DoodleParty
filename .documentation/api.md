# DoodleParty API Reference

**Purpose:** Complete API documentation for WebSocket events, REST endpoints, and ML inference.

**Status: Updated to match actual implementation**

## Table of Contents

### WebSocket API
- [WebSocket API](#websocket-api)
  - [Connection Setup](#connection-setup)
  - [Client → Server Events](#client--server-events)
    - [`join-canvas`](#join-canvas)
    - [`stroke`](#stroke)
    - [`vote`](#vote)
    - [`chat`](#chat)
    - [`timer-ready`](#timer-ready)
    - [`leave-canvas`](#leave-canvas)
  - [Server → Client Events](#server--client-events)
    - [`stroke`](#stroke-1)
    - [`canvas-update`](#canvas-update)
    - [`moderation-alert`](#moderation-alert)
    - [`prompt`](#prompt)
    - [`timer-tick`](#timer-tick)
    - [`results`](#results)
    - [`leaderboard-update`](#leaderboard-update)
    - [`player-joined`](#player-joined)
    - [`player-left`](#player-left)
    - [`error`](#error)

### REST API
- [REST API](#rest-api)
  - [Health Check](#health-check)
    - [`GET /health`](#get-health)
  - [Canvas Management](#canvas-management)
    - [`GET /canvases`](#get-canvases)
    - [`POST /canvases`](#post-canvases)
    - [`GET /canvases/:canvasId`](#get-canvasescanvasid)
    - [`POST /canvases/:canvasId/export`](#post-canvasescanvasidexport)
  - [Leaderboard & Achievements](#leaderboard--achievements)
    - [`GET /leaderboard`](#get-leaderboard)
    - [`GET /achievements`](#get-achievements)

### ML Inference API
- [ML Inference API](#ml-inference-api)
  - [Standard Classification](#standard-classification)
    - [`POST /api/predict`](#post-apipredict)
  - [Advanced Detection Methods](#advanced-detection-methods)
    - [`POST /api/predict/shape`](#post-apipredictshape)
    - [`POST /api/predict/tile`](#post-apipredicttile)
  - [Service Health](#service-health)
    - [`GET /api/health`](#get-apihealth)

### Game Mode APIs
- [Game Mode APIs](#game-mode-apis)
  - [Speed Sketch Challenge](#speed-sketch-challenge)
    - [`POST /api/games/speed-sketch/start`](#post-apigamesspeed-sketchstart)
  - [Guess The Doodle](#guess-the-doodle)
    - [`POST /api/games/guess-doodle/start`](#post-apigamesguess-doodlestart)
  - [Battle Royale Doodle](#battle-royale-doodle)
    - [`POST /api/games/battle-royale/start`](#post-apigamesbattle-royalestart)

### Error Handling
- [Error Handling](#error-handling)
  - [Error Response Format](#error-response-format)
  - [Error Codes](#error-codes)
  - [Exception Handling](#exception-handling)
    - [WebSocket Errors](#websocket-errors)
    - [REST API Errors](#rest-api-errors)

## Related Documentation

### Implementation Guides
- [Architecture Overview](architecture.md) - System design and component interactions
- [ML Pipeline](ml-pipeline.md) - Content moderation implementation details
- [Installation Guide](installation.md) - Setup instructions for development and deployment

### Development
- [Testing Strategy](testing.md) - API testing approaches and implementation
- [Code Style Guide](../STYLE_GUIDE.md) - Development standards and conventions

## WebSocket API

**Base URL:** `ws://localhost:3000` (development) or `wss://api.doodleparty.io` (production)

### Connection

```javascript
const socket = io('http://localhost:3000', {
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 5,
});

socket.on('connect', () => {
  console.log('Connected to server');
});

socket.on('disconnect', () => {
  console.log('Disconnected from server');
});
```

### Client → Server Events

#### `join-canvas`

Join a collaborative canvas session.

**Payload:**
```json
{
  "canvasId": "canvas-123",
  "userId": "user-456",
  "username": "Alice",
  "gameMode": "classic"
}
```

**Response:**
```json
{
  "success": true,
  "canvasState": {
    "width": 512,
    "height": 512,
    "strokes": [],
    "timer": 300,
    "currentPlayers": 5
  }
}
```

#### `stroke`

Send a drawing stroke to the canvas.

**Payload:**
```json
{
  "points": [
    {"x": 100, "y": 150, "t": 1699900000000},
    {"x": 105, "y": 155, "t": 1699900000050}
  ],
  "color": "#000000",
  "brushSize": 3,
  "inkUsed": 2.5,
  "timestamp": 1699900000000
}
```

**Response (if approved):**
```json
{
  "success": true,
  "strokeId": "stroke-789",
  "moderated": false
}
```

**Response (if rejected):**
```json
{
  "success": false,
  "reason": "Content flagged by moderation",
  "confidence": 0.87,
  "message": "This drawing was removed for violating community guidelines"
}
```

**Status Codes:**
- `200 OK` - Stroke accepted
- `400 Bad Request` - Invalid stroke data
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Moderation service error

#### `vote`

Submit a vote in game modes (Speed Sketch, Battle Royale, etc.).

**Payload:**
```json
{
  "gameRoundId": "round-123",
  "votedUserId": "user-456",
  "voteType": "best-drawing"
}
```

**Response:**
```json
{
  "success": true,
  "voteCount": 5,
  "message": "Vote recorded"
}
```

#### `chat`

Send a text message (if enabled).

**Payload:**
```json
{
  "message": "Nice drawing!",
  "timestamp": 1699900000000
}
```

**Response:**
```json
{
  "success": true,
  "messageId": "msg-123",
  "displayName": "Alice"
}
```

#### `timer-ready`

Indicate player is ready for next round.

**Payload:**
```json
{
  "userId": "user-456",
  "ready": true
}
```

#### `leave-canvas`

Leave the current canvas session.

**Payload:**
```json
{
  "canvasId": "canvas-123",
  "userId": "user-456"
}
```

### Server → Client Events

#### `stroke`

Broadcast a stroke to all connected clients.

**Payload:**
```json
{
  "userId": "user-456",
  "username": "Alice",
  "points": [
    {"x": 100, "y": 150},
    {"x": 105, "y": 155}
  ],
  "color": "#000000",
  "brushSize": 3,
  "strokeId": "stroke-789"
}
```

#### `canvas-update`

Full canvas state update (periodic or on join).

**Payload:**
```json
{
  "canvasId": "canvas-123",
  "width": 512,
  "height": 512,
  "strokes": [
    {
      "userId": "user-456",
      "points": [...],
      "color": "#000000",
      "brushSize": 3
    }
  ],
  "currentPlayers": 5,
  "timer": 245,
  "gameMode": "classic"
}
```

#### `moderation-alert`

Notify user that their stroke was flagged/removed.

**Payload:**
```json
{
  "strokeId": "stroke-789",
  "reason": "Content flagged by moderation",
  "confidence": 0.87,
  "action": "removed",
  "message": "Your drawing was removed for violating community guidelines"
}
```

#### `prompt`

New game mode prompt (Speed Sketch, Battle Royale, etc.).

**Payload:**
```json
{
  "gameRoundId": "round-123",
  "prompt": "Draw your hometown in 30 seconds",
  "category": "location",
  "timeLimit": 30,
  "difficulty": "medium"
}
```

#### `timer-tick`

Countdown timer update.

**Payload:**
```json
{
  "timeRemaining": 245,
  "totalTime": 300
}
```

#### `results`

Game round results (voting complete, winner announced).

**Payload:**
```json
{
  "gameRoundId": "round-123",
  "winner": {
    "userId": "user-456",
    "username": "Alice",
    "points": 100
  },
  "topVoted": [
    {
      "userId": "user-456",
      "username": "Alice",
      "votes": 12
    },
    {
      "userId": "user-789",
      "username": "Bob",
      "votes": 8
    }
  ]
}
```

#### `leaderboard-update`

Leaderboard changed (new scores, rankings).

**Payload:**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "userId": "user-456",
      "username": "Alice",
      "points": 450,
      "level": 5,
      "achievements": 12
    },
    {
      "rank": 2,
      "userId": "user-789",
      "username": "Bob",
      "points": 380,
      "level": 4,
      "achievements": 8
    }
  ]
}
```

#### `player-joined`

New player joined the canvas.

**Payload:**
```json
{
  "userId": "user-999",
  "username": "Charlie",
  "joinedAt": 1699900000000,
  "totalPlayers": 6
}
```

#### `player-left`

Player left the canvas.

**Payload:**
```json
{
  "userId": "user-999",
  "username": "Charlie",
  "leftAt": 1699900000000,
  "totalPlayers": 5
}
```

#### `error`

Error occurred on server.

**Payload:**
```json
{
  "code": "MODERATION_SERVICE_ERROR",
  "message": "ML inference service temporarily unavailable",
  "severity": "warning"
}
```

## REST API

**Base URL:** `http://localhost:3000/api` (development) or `https://api.doodleparty.io` (production)

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "uptime": 3600,
  "version": "1.0.0",
  "services": {
    "websocket": "ok",
    "ml_inference": "ok",
    "database": "ok"
  }
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `503 Service Unavailable` - Service unhealthy

### `GET /canvases`

List all active canvases.

**Query Parameters:**
- `gameMode` (optional) - Filter by game mode
- `limit` (optional) - Max results (default: 10)
- `offset` (optional) - Pagination offset (default: 0)

**Response:**
```json
{
  "canvases": [
    {
      "canvasId": "canvas-123",
      "gameMode": "classic",
      "currentPlayers": 5,
      "maxPlayers": 100,
      "createdAt": 1699900000000,
      "timerRemaining": 245
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### `POST /canvases`

Create a new canvas session.

**Request Body:**
```json
{
  "gameMode": "speed-sketch",
  "maxPlayers": 50,
  "timerDuration": 300,
  "theme": "nature"
}
```

**Response:**
```json
{
  "success": true,
  "canvasId": "canvas-123",
  "gameMode": "speed-sketch",
  "joinUrl": "http://localhost:3000/join/canvas-123"
}
```

**Status Codes:**
- `201 Created` - Canvas created
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Server error

### `GET /canvases/:canvasId`

Get canvas details.

**Response:**
```json
{
  "canvasId": "canvas-123",
  "gameMode": "classic",
  "width": 512,
  "height": 512,
  "currentPlayers": 5,
  "maxPlayers": 100,
  "createdAt": 1699900000000,
  "timerRemaining": 245,
  "strokes": 1250,
  "moderationStats": {
    "totalStrokes": 1250,
    "flaggedStrokes": 3,
    "flagRate": 0.24
  }
}
```

### `POST /canvases/:canvasId/export`

Export canvas as image or JSON.

**Request Body:**
```json
{
  "format": "png",
  "width": 1024,
  "height": 1024
}
```

**Response:**
```json
{
  "success": true,
  "downloadUrl": "https://cdn.doodleparty.io/exports/canvas-123.png",
  "expiresIn": 3600
}
```

### `GET /leaderboard`

Get global leaderboard.

**Query Parameters:**
- `timeframe` (optional) - "day", "week", "month", "all" (default: "week")
- `limit` (optional) - Max results (default: 10)

**Response:**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "userId": "user-456",
      "username": "Alice",
      "points": 4500,
      "level": 12,
      "achievements": 25
    }
  ],
  "timeframe": "week",
  "generatedAt": 1699900000000
}
```

### `GET /achievements`

List all available achievements.

**Response:**
```json
{
  "achievements": [
    {
      "id": "first-stroke",
      "name": "First Stroke",
      "description": "Draw your first line on any canvas",
      "icon": "https://cdn.doodleparty.io/icons/first-stroke.png",
      "rarity": "common"
    },
    {
      "id": "ink-master",
      "name": "Ink Master",
      "description": "Use exactly 95-100% of your ink before timer expires",
      "icon": "https://cdn.doodleparty.io/icons/ink-master.png",
      "rarity": "rare"
    }
  ]
}
```

## ML Inference API

**Base URL:** `http://localhost:5001` (RPi4) or `https://ml.doodleparty.io` (cloud)

**RPi4 Performance Targets:**
- Single image: <50ms (INT8 TFLite)
- Batch (10 images): <200ms
- Model size: <5MB
- Memory usage: <500MB

*For detailed information about the ML model architecture, detection strategies, and performance optimization, see the [ML Pipeline documentation](ml-pipeline.md).*

### `POST /api/predict`

Standard single-image classification.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "APPROVED",
  "confidence": 0.15,
  "threshold": 0.50,
  "model_info": "TFLite INT8 Binary Classifier",
  "drawing_statistics": {
    "response_time_ms": 45.2,
    "preprocess_time_ms": 12.3,
    "inference_time_ms": 28.7
  }
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid image data
- `500 Internal Server Error` - Model inference failed

### `POST /api/predict/shape`

Shape-based detection with stroke awareness.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "stroke_history": [
    {
      "points": [
        {"x": 120, "y": 260, "t": 1730980000000},
        {"x": 160, "y": 260, "t": 1730980000100}
      ],
      "timestamp": 1730980000000
    }
  ],
  "min_shape_area": 100
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "APPROVED",
  "confidence": 0.12,
  "detection_details": {
    "num_shapes_analyzed": 3,
    "shape_predictions": [
      {
        "shape_id": 0,
        "confidence": 0.15,
        "is_positive": false,
        "area": 12800
      }
    ]
  },
  "drawing_statistics": {
    "response_time_ms": 95.3,
    "preprocess_time_ms": 20.1,
    "inference_time_ms": 65.0
  }
}
```

### `POST /api/predict/tile`

Tile-based detection with grid partitioning.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "tile_size": 64,
  "canvas_width": 512,
  "canvas_height": 512,
  "force_full_analysis": false
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "APPROVED",
  "confidence": 0.18,
  "detection_details": {
    "num_tiles_analyzed": 12,
    "total_tiles": 64,
    "grid_size": 8,
    "cached": false
  },
  "drawing_statistics": {
    "response_time_ms": 85.3,
    "preprocess_time_ms": 22.1,
    "inference_time_ms": 58.7
  }
}
```

### `GET /api/health`

ML service health check.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "TFLite INT8 (Optimized)",
  "model_type": "TFLite",
  "threshold": 0.50
}
```

## Game Mode APIs

*For complete implementation details of game mode logic and architecture, see the [Architecture Overview](architecture.md#game-modes-architecture).*

### Speed Sketch Challenge

**Endpoint:** `POST /api/games/speed-sketch/start`

**Request Body:**
```json
{
  "canvasId": "canvas-123",
  "roundDuration": 30,
  "category": "nature"
}
```

**Response:**
```json
{
  "success": true,
  "gameRoundId": "round-123",
  "prompt": "Draw a tree",
  "timeLimit": 30,
  "startedAt": 1699900000000
}
```

### Guess The Doodle

**Endpoint:** `POST /api/games/guess-doodle/start`

**Request Body:**
```json
{
  "canvasId": "canvas-123",
  "drawerUserId": "user-456",
  "roundDuration": 60
}
```

**Response:**
```json
{
  "success": true,
  "gameRoundId": "round-123",
  "drawerId": "user-456",
  "timeLimit": 60,
  "startedAt": 1699900000000
}
```

### Battle Royale Doodle

**Endpoint:** `POST /api/games/battle-royale/start`

**Request Body:**
```json
{
  "canvasId": "canvas-123",
  "maxRounds": 5,
  "eliminationRate": 0.25
}
```

**Response:**
```json
{
  "success": true,
  "gameRoundId": "round-123",
  "totalRounds": 5,
  "currentRound": 1,
  "activePlayers": 20,
  "startedAt": 1699900000000
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_STROKE",
    "message": "Stroke data is invalid",
    "details": "Missing required field: points",
    "timestamp": 1699900000000
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_STROKE` | Stroke data is malformed | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests from user | 429 |
| `MODERATION_SERVICE_ERROR` | ML inference service error | 500 |
| `CANVAS_NOT_FOUND` | Canvas session doesn't exist | 404 |
| `UNAUTHORIZED` | Authentication failed | 401 |
| `INTERNAL_ERROR` | Unexpected server error | 500 |

### Exception Handling

**WebSocket Errors:**

```javascript
socket.on('error', (error) => {
  console.error('Socket error:', error);
  // Implement reconnection logic
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error);
});
```

**REST API Errors:**

```javascript
fetch('/api/canvases', {
  method: 'POST',
  body: JSON.stringify(data),
})
  .then(res => {
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    return res.json();
  })
  .catch(error => {
    console.error('API error:', error);
  });
```

## Further Documentation

### Implementation
- [Architecture Overview](architecture.md) - Complete system design and component interactions
- [ML Pipeline](ml-pipeline.md) - Detailed content moderation implementation
- [Project Structure](structure.md) - Code organization and file layout

### Operations
- [Installation Guide](installation.md) - Complete setup for development and deployment
- [Nix Usage Guide](nix-usage.md) - NixOS-specific deployment instructions
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### Development
- [Testing Strategy](testing.md) - API testing approach and implementation
- [Code Style Guide](../STYLE_GUIDE.md) - Development standards and conventions
- [Development Roadmap](roadmap.md) - Future API features and enhancements

*API Reference for DoodleParty v1.0*
