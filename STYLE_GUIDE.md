# DoodleParty - Code Style Guide

**Purpose:** Defines consistent coding and documentation standards for all DoodleParty modules.

**Status: Updated**

This guide covers Node.js backend (Express, Socket.io), React frontend, ML pipeline (Python/TensorFlow), and data processing components.

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Code Formatting](#code-formatting)
3. [Import Ordering](#import-ordering)
4. [Language-Specific Guidelines](#language-specific-guidelines)
5. [Comments and Documentation](#comments-and-documentation)
6. [Security Guidelines](#security-guidelines)
7. [Code Review Checklist](#code-review-checklist)

## Naming Conventions

### Files

- JavaScript/TypeScript: `camelCase.ts` or `camelCase.js`
- React Components: `PascalCase.tsx`
- Styles: `kebab-case.css`
- Shell scripts: `kebab-case.sh`

**Examples:**
- `canvasManager.ts`
- `DrawingCanvas.tsx`
- `moderation-service.ts`
- `deploy-rpi4.sh`

### Code Elements

**Classes/Components:**
- PascalCase

```typescript
class CanvasManager {
  // ...
}

function DrawingCanvas() {
  // ...
}
```

**Functions/Methods:**
- camelCase

```typescript
function handleStroke(stroke: Stroke): void {
  // ...
}

const normalizeCoordinates = (x: number, y: number): Point => {
  // ...
};
```

**Constants:**
- UPPER_SNAKE_CASE

```typescript
const MAX_CONCURRENT_USERS = 100;
const INFERENCE_TIMEOUT_MS = 50;
const DEFAULT_INK_CAPACITY = 100;
```

## Code Formatting

### Indentation

- JavaScript/TypeScript: **2 spaces**
- Python: **4 spaces**
- Shell scripts: **2 spaces**

### Line Length

- Maximum: **100 characters**

### Quotes

- Single quotes (`'`) for JavaScript/TypeScript
- Double quotes for JSON and docstrings

```typescript
const message = 'Drawing received';
const config = { brushSize: 5 }; // JSON-like object
```

### Trailing Commas

- Always use in multi-line arrays/objects

```typescript
const config = {
  maxUsers: 100,
  inferenceTimeout: 50,
  inkCapacity: 100,
};
```

## Import Ordering

### JavaScript/TypeScript

1. Standard library (Node.js)
2. Third-party packages
3. Local imports

```typescript
import fs from 'fs';
import path from 'path';

import express from 'express';
import { io } from 'socket.io';

import { CanvasManager } from './services/canvasManager';
import { ModerationService } from './services/moderationService';
```

### Python

1. Standard library
2. Third-party packages
3. Local imports

```python
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.core.inference import predict_image
from src.data.loaders import load_model
```

## Language-Specific Guidelines

### JavaScript/TypeScript

**Type Hints:**

```typescript
interface Stroke {
  points: Point[];
  timestamp: number;
  userId: string;
  inkUsed: number;
}

function processStroke(stroke: Stroke): Promise<ModeratedStroke> {
  // ...
}

const validateStroke = (stroke: Stroke): boolean => {
  return stroke.points.length > 0 && stroke.timestamp > 0;
};
```

**JSDoc Comments:**

```typescript
/**
 * Processes a drawing stroke through the moderation pipeline.
 *
 * @param stroke - The stroke data to process
 * @param threshold - Confidence threshold for moderation (0.0-1.0)
 * @returns Promise resolving to moderation result
 * @throws Error if moderation service is unavailable
 */
async function moderateStroke(
  stroke: Stroke,
  threshold: number = 0.5,
): Promise<ModerationResult> {
  // ...
}
```

### React Components

**Functional Component Pattern:**

```typescript
interface DrawingCanvasProps {
  width: number;
  height: number;
  onStroke: (stroke: Stroke) => void;
  disabled?: boolean;
}

export function DrawingCanvas({
  width,
  height,
  onStroke,
  disabled = false,
}: DrawingCanvasProps): JSX.Element {
  // Component implementation
  return <canvas width={width} height={height} />;
}
```

### Python

**Type Hints:**

```python
from typing import Tuple, Optional
import numpy as np

def preprocess_image(
    image: np.ndarray,
    target_size: int = 28
) -> Tuple[np.ndarray, bool]:
    """
    Preprocesses image for model input.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size for resizing

    Returns:
        Tuple of (preprocessed_image, is_valid)
    """
    if image.shape[0] < target_size or image.shape[1] < target_size:
        return None, False
    
    processed = resize_image(image, target_size)
    return processed, True
```

## Comments and Documentation

### File Headers

**Format:**

```javascript
/**
 * Real-time Canvas Drawing Handler
 *
 * Manages WebSocket connections for collaborative drawing.
 * Handles stroke synchronization, ink depletion, and rate limiting.
 *
 * Related:
 * - src/services/moderationService.ts (content moderation)
 * - src/hooks/useDraggable.tsx (drawing input)
 * - src/components/DrawingCanvas.tsx (UI rendering)
 *
 * Exports:
 * - CanvasManager, handleStroke, validateStroke
 */
```

**Rules:**
1. First line: Brief description of the module
2. Blank comment line after first line
3. Bullet points use `-` character (standard markdown)
4. Each section separated by blank comment line
5. Include Usage section where relevant

### Standalone Comments

```javascript
// Normalize stroke coordinates to canvas bounds
const normalizedStroke = normalizeCoordinates(stroke, canvasWidth, canvasHeight);
```

**Rules:**
1. Single line whenever possible
2. Multi-line if necessary, each line starts with `//`
3. No trailing comments after closing braces
4. Explain **why**, not **what**

### Comment Content Guidelines

**Do:**
- ✓ Explain **why**, not **what**
- ✓ Document edge cases and gotchas
- ✓ Reference external resources
- ✓ Note data format expectations

**Don't:**
- ✗ State the obvious (`i = 0  // Set i to 0`)
- ✗ Write overly verbose comments
- ✗ Use decorative separators

### Special Comment Types

**TODOs:**

```javascript
// TODO: Add support for pressure-sensitive drawing
// TODO(DoodleParty-team): Implement crew-based relay races
```

**Notes:**

```javascript
// NOTE: This requires Node.js 18+ for native fetch support
```

**Security:**

```javascript
// SECURITY: Validate all WebSocket messages before processing
// Prevent injection attacks in canvas data
```

### Multi-Line Comments

```javascript
// First line of explanation
// Second line of explanation
// Third line of explanation
const result = computeValue();
```

**Rules:**
1. Each line starts with `//` and one space
2. No blank lines within multi-line comment
3. Blank line before code block

## Security Guidelines

### Input Validation

```typescript
function validateCanvasData(data: unknown): data is CanvasData {
  if (!data || typeof data !== 'object') {
    return false;
  }

  const obj = data as Record<string, unknown>;
  
  // Check required fields
  if (typeof obj.strokes !== 'object' || !Array.isArray(obj.strokes)) {
    return false;
  }

  // Prevent excessively large payloads
  if (JSON.stringify(obj).length > MAX_CANVAS_SIZE) {
    return false;
  }

  return true;
}
```

### Rate Limiting

```typescript
// Prevent bot-like drawing patterns
const validateDrawingRate = (userId: string, points: number): boolean => {
  const now = Date.now();
  const userRate = userDrawingRates.get(userId) || [];
  
  // Remove old entries (older than 1 second)
  const recentRate = userRate.filter(t => now - t < 1000);
  
  // Max 100 points per second per user
  if (recentRate.reduce((sum, _) => sum + points, 0) > 100) {
    return false;
  }

  recentRate.push(now);
  userDrawingRates.set(userId, recentRate);
  return true;
};
```

### WebSocket Security

```typescript
// Validate and sanitize all incoming WebSocket messages
socket.on('stroke', (data: unknown) => {
  if (!validateCanvasData(data)) {
    socket.emit('error', { message: 'Invalid stroke data' });
    return;
  }

  // Process validated data
  handleStroke(data);
});
```

## Code Review Checklist

### Documentation & Comments
- [ ] File headers follow standard format
- [ ] Comments explain **why** not **what**
- [ ] Standalone comments preferred over inline
- [ ] No obvious/redundant comments
- [ ] Data format expectations documented
- [ ] TODO/FIXME comments use proper format

### Formatting
- [ ] 2-space indent for JavaScript/TypeScript
- [ ] 4-space indent for Python
- [ ] Line length ≤100 characters
- [ ] Trailing commas in multi-line structures

### Naming
- [ ] Files: camelCase.ts or PascalCase.tsx
- [ ] Classes/Components: PascalCase
- [ ] Functions: camelCase
- [ ] Constants: UPPER_SNAKE_CASE

### Language-Specific
- [ ] TypeScript: Type hints and JSDoc comments
- [ ] React: Proper prop typing and component structure
- [ ] Python: Type hints and docstrings
- [ ] Node.js: Proper error handling and async/await

### Security
- [ ] Input validation present
- [ ] No hardcoded secrets
- [ ] Rate limiting implemented
- [ ] Error handling prevents info leaks

### Real-Time Features
- [ ] WebSocket messages validated
- [ ] Proper event namespacing
- [ ] Connection state managed
- [ ] Reconnection logic implemented

## Documentation Standards

### Project Documentation

Keep documentation in existing files:

- `README.md` - Project overview
- `STYLE_GUIDE.md` - This file
- Individual file docstrings - Implementation details

**Rule:** No new `.md` files. Extend existing files instead.

## Related Documentation

- [Architecture](.documentation/architecture.md) - System design
- [API Reference](.documentation/api.md) - API documentation
- [Installation](.documentation/installation.md) - Setup guide
- [README](README.md) - Project overview

*Code style guide for DoodleParty v1.0*
