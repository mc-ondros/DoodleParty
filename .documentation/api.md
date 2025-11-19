# API Reference

## HTTP REST API

The Game Server provides REST endpoints for configuration and state management.

### Admin & Configuration
*   `GET /api/admin-config`: Retrieve current admin configuration.
    *   Response: `AdminConfig` object (see [Schemas](#schemas)).
*   `POST /api/admin-config`: Update admin configuration.
    *   Body: Partial `AdminConfig` object.
    *   Response: Updated `AdminConfig`.
*   `POST /api/players/kick`: Kick a player.
    *   Body: `{ "id": "socket_id" }`
    *   Response: `{ "success": boolean, "message": string }`

### Game State
*   `GET /api/state`: Retrieve full game state.
    *   Response: `{ "sessionId": string, "config": AdminConfig, "timer": number, "playerCount": number }`
*   `GET /api/timer`: Retrieve current timer snapshot.
    *   Response: `{ "remaining": number, "total": number, "running": boolean }`
*   `POST /api/timer`: Control the timer (Admin only).
    *   Body: `{ "action": "start" | "stop" | "pause" | "reset", "seconds": number }`
    *   Response: `{ "success": boolean, "timer": number }`
*   `GET /api/players/count`: Get current number of connected players.
    *   Response: `{ "count": number }`

## Socket.io Events

Real-time events drive the system. All events use JSON payloads.

### Drawing Events
*   `quickdraw.stroke`: Broadcasts a single stroke.
    *   Payload: `{ "points": [{"x": number, "y": number, "t": number}], "color": string, "size": number }`
*   `quickdraw.batch`: Broadcasts multiple strokes.
    *   Payload: `{ "strokes": [...] }` (array of stroke objects)
*   `quickdraw.clear`: Clears the canvas for all users.
    *   Payload: `{ "sessionId": string }`
*   `quickdraw.undo`: Undo last stroke.
    *   Payload: `{ "sessionId": string }`

### ML & Moderation Events
*   `ml.detectObjects` (Client → Server → ML): Request object detection.
    *   Payload: `{ "sessionId": string, "timestamp": number, "objects": [{ "image": "data:image/png;base64,...", "boundingBox": {"x1": number, "y1": number, "x2": number, "y2": number} }] }`
*   `ml.detectionResults` (ML → Server → Client): Returns inference results.
    *   Payload: `{ "success": boolean, "sessionId": string, "results": [{"prediction": number, "class": "positive"|"negative", "confidence": number}], "summary": {"total": number, "positive": number, "negative": number} }`
*   `quickdraw.eraseRegion` (Server → Client): Instructions to erase a region.
    *   Payload: `{ "sessionId": string, "x1": number, "y1": number, "x2": number, "y2": number, "reason": string }`

### System Events
*   `state:init`: Sent to new clients with initial state.
    *   Payload: `{ "sessionId": string, "config": AdminConfig, "playerCount": number, "timer": number }`
*   `timer:update`: Broadcasts timer changes.
    *   Payload: `{ "remaining": number, "total": number, "running": boolean }`
*   `players:update`: Broadcasts player count changes.
    *   Payload: `{ "count": number }`
*   `config:update`: Broadcasts configuration changes.
    *   Payload: `{ "config": AdminConfig }`

## Schemas

### AdminConfig
```json
{
  "Timer": number,
  "TimerPreset": string,
  "Game Mode": string,
  "Max Players": number,
  "Ink Limit": string,
  "Teams": string,
  "Visibility": string,
  "Password": string,
  "Custom Prompt": string,
  "Content Mode": string,
  "Session": string
}
```

### Stroke
```json
{
  "points": [
    { "x": number, "y": number, "t": number }
  ],
  "color": string,
  "size": number
}
```

### DetectionResult
```json
{
  "prediction": number,
  "class": "positive" | "negative",
  "confidence": number,
  "demo_mode": boolean
}
```

## Environment Variables

*   `EXPRESS_URL`: ML server connection URL (default: `http://localhost:3000`)
*   `ENABLE_VISUALIZATIONS`: Save debug visualizations (default: `false`)
*   `DOODLEPARTY_MODEL`: Path to TFLite model (default: `models/quickdraw_model_int8.tflite`)

## Model Details

*   **Input**: 128×128 grayscale image, normalized to [0, 1]
*   **Output**: Binary classification (0 = safe, 1 = offensive)
*   **Threshold**: 0.5 (prediction > 0.5 → positive/offensive)
*   **Format**: TensorFlow Lite (`.tflite`)

## Error Handling

*   **ML Service offline**: Strokes are buffered; detection requests fail gracefully with error in response.
*   **Invalid image**: Detection returns `{ "success": false, "error": "..." }`.
*   **Malformed payload**: Server responds with HTTP 400 or Socket.io error event.
