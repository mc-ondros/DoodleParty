# DoodleParty Testing Strategy

**Purpose:** Comprehensive testing documentation for DoodleParty collaborative drawing platform.

**Status: Production Ready** - Complete testing framework for real-time collaboration, content moderation, and game modes.

## Table of Contents

### Testing Foundation
- [Testing Philosophy](#testing-philosophy)
  - [Core Principles](#core-principles)
  - [Testing Stack](#testing-stack)
- [Test Architecture](#test-architecture)
  - [File Structure](#file-structure)

### Testing Levels
- [Unit Testing](#unit-testing)
  - [Canvas Logic Tests](#canvas-logic-tests)
    - [Key Test Cases](#key-test-cases)
  - [Moderation Service Tests](#moderation-service-tests)
    - [Key Test Cases](#key-test-cases-1)
  - [Game Mode Tests](#game-mode-tests)
    - [Key Test Cases](#key-test-cases-2)
  - [Leaderboard Tests](#leaderboard-tests)
    - [Key Test Cases](#key-test-cases-3)

- [Integration Testing](#integration-testing)
  - [WebSocket Communication](#websocket-communication)
    - [Key Test Cases](#key-test-cases-4)
  - [Database Integration](#database-integration)
    - [Key Test Cases](#key-test-cases-5)
  - [ML Pipeline Integration](#ml-pipeline-integration)
    - [Key Test Cases](#key-test-cases-6)

- [End-to-End Testing](#end-to-end-testing)
  - [Key Test Scenarios](#key-test-scenarios)

### Performance & Accessibility
- [Performance Testing](#performance-testing)
  - [Load Testing](#load-testing)
    - [Targets](#targets)
  - [Stress Testing](#stress-testing)
    - [Targets](#targets-1)
  - [Accessibility Testing](#accessibility-testing)
    - [Key Test Cases](#key-test-cases-7)

### Coverage & Execution
- [Test Coverage](#test-coverage)
  - [Coverage Targets](#coverage-targets)
- [Running Tests](#running-tests)
  - [Commands](#commands)
  - [CI/CD Pipeline](#cicd-pipeline)
    - [Key Steps](#key-steps)

### Resources
- [Related Documentation](#related-documentation)

## Testing Philosophy

### Core Principles

**1. Test-Driven Development** - Write tests before implementation to ensure code quality and prevent regressions.

**2. Comprehensive Coverage** - Follow testing pyramid: Unit tests (60%), Integration tests (30%), E2E tests (10%).

**3. Real-Time Focus** - Prioritize testing WebSocket communication, concurrent users, and real-time state synchronization.

**4. Performance Validation** - Monitor inference latency, memory usage, and canvas rendering performance.

### Testing Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Unit Testing** | Jest | JavaScript/TypeScript unit tests |
| **E2E Testing** | Playwright | Browser automation and user workflows |
| **API Testing** | Supertest | HTTP and WebSocket testing |
| **Performance** | K6, Lighthouse | Load and performance testing |
| **Coverage** | Istanbul/NYC | Code coverage reporting |

## Test Architecture

### File Structure

```
tests/
├── unit/
│   ├── canvas.test.ts
│   ├── moderation.test.ts
│   ├── gameMode.test.ts
│   └── leaderboard.test.ts
├── integration/
│   ├── websocket.test.ts
│   ├── database.test.ts
│   └── mlPipeline.test.ts
├── e2e/
│   ├── speedSketch.spec.ts
│   └── guessTheDoodle.spec.ts
└── performance/
    ├── loadTest.ts
    └── stressTest.ts
```

## Unit Testing

### Canvas Logic Tests

**File:** `tests/unit/canvas.test.ts`

Tests for stroke management, undo/redo functionality, canvas state serialization, and image export. Verifies that strokes are properly added, removed, and managed with full undo/redo stack support.

**Key Test Cases:**
- Add stroke to canvas
- Remove stroke by ID
- Undo/redo operations
- Clear all strokes
- Export canvas as image
- Serialize canvas state

### Moderation Service Tests

**File:** `tests/unit/moderation.test.ts`

Tests for content classification, confidence scoring, and latency requirements. Ensures moderation meets <50ms per-stroke and <200ms batch processing targets while maintaining >88% accuracy.

**Key Test Cases:**
- Classify appropriate content as approved
- Classify inappropriate content as rejected
- Verify latency <50ms for single stroke
- Verify batch processing <200ms for 10 strokes
- Apply confidence threshold correctly
- Handle edge cases and ambiguous content

### Game Mode Tests

**File:** `tests/unit/gameMode.test.ts`

Tests for Speed Sketch, Guess The Doodle, Battle Royale, and Collaborative Story modes. Verifies game initialization, prompt generation, voting tallying, and point allocation.

**Key Test Cases:**
- Initialize game with correct duration
- Generate prompts from LLM
- Add and manage player drawings
- Tally votes correctly
- Award points to winners
- Track round progression
- Handle player elimination (Battle Royale)

### Leaderboard Tests

**File:** `tests/unit/leaderboard.test.ts`

Tests for score tracking, ranking calculations, and leaderboard updates. Ensures scores are properly updated and rankings reflect current standings.

**Key Test Cases:**
- Create and update player scores
- Calculate rankings correctly
- Handle tied scores
- Filter by game mode
- Filter by time period (current, today, all-time)

## Integration Testing

### WebSocket Communication

**File:** `tests/integration/websocket.test.ts`

Tests real-time communication between clients and server. Verifies stroke broadcasting, message delivery, rate limiting, and connection management.

**Key Test Cases:**
- Broadcast stroke to all connected clients
- Handle multiple concurrent strokes
- Enforce rate limits (100 strokes/sec per user)
- Manage client connections and disconnections
- Handle reconnection scenarios
- Verify message ordering

### Database Integration

**File:** `tests/integration/database.test.ts`

Tests database operations for player management, game results, and leaderboard persistence. Ensures data integrity and query performance.

**Key Test Cases:**
- Create and retrieve player records
- Update player scores
- Store and retrieve game results
- Query top players by score
- Handle concurrent database operations
- Verify transaction integrity

### ML Pipeline Integration

**File:** `tests/integration/mlPipeline.test.ts`

Tests model loading, inference performance, accuracy retention, and memory usage. Validates that quantized model meets all performance requirements.

**Key Test Cases:**
- Load TFLite model successfully
- Verify model size <5MB
- Classify strokes within <50ms latency
- Maintain >88% accuracy after quantization
- Memory usage <500MB for 100 concurrent inferences
- Handle batch inference efficiently

## End-to-End Testing

**File:** `tests/e2e/speedSketch.spec.ts`

Tests complete user workflows using Playwright for browser automation. Simulates multiple players joining games, drawing, voting, and viewing results.

**Key Test Scenarios:**
- Complete Speed Sketch game flow (join → draw → vote → results)
- Guess The Doodle game flow (drawer selection → guessing → scoring)
- Moderation alert handling (inappropriate content detection)
- Real-time leaderboard updates
- Multiple concurrent players
- Reconnection after disconnect
- Game mode transitions

## Performance Testing

### Load Testing

**File:** `tests/performance/loadTest.ts`

Uses K6 to simulate 100 concurrent virtual users over 5 minutes. Measures response times, throughput, and failure rates.

**Targets:**
- 95% of requests <500ms
- <10% failure rate
- Sustained 100 concurrent users
- WebSocket connection stability

### Stress Testing

**File:** `tests/performance/stressTest.ts`

Tests system behavior under extreme load. Simulates 100 concurrent users drawing simultaneously, 1000 moderation inferences, and database operations.

**Targets:**
- Handle 100 concurrent users drawing
- Process 1000 moderation inferences within 60 seconds
- Memory usage remains stable
- No memory leaks over extended load

### Accessibility Testing

**File:** `tests/accessibility/a11y.test.ts`

Validates WCAG AA compliance, keyboard navigation, color contrast, and screen reader support.

**Key Test Cases:**
- WCAG AA compliance with axe-core
- Keyboard navigation through all interactive elements
- Color contrast ratios (4.5:1 minimum)
- Respect `prefers-reduced-motion` preference
- Focus indicators visible
- Screen reader compatibility

## Test Coverage

### Coverage Targets

| Component | Target | Status |
|-----------|--------|--------|
| Canvas Logic | 90% | Critical |
| Moderation Service | 95% | Critical |
| Game Modes | 85% | High |
| WebSocket Handlers | 90% | Critical |
| Database Layer | 80% | High |
| UI Components | 70% | Medium |

## Running Tests

### Commands

```bash
npm test                    # Run all tests
npm run test:unit          # Unit tests only
npm run test:integration   # Integration tests
npm run test:e2e           # End-to-end tests
npm run test:performance   # Performance tests
npm run test:coverage      # Generate coverage report
npm run test:watch         # Watch mode
```

### CI/CD Pipeline

Tests run automatically on push and pull requests via GitHub Actions:

1. **Unit Tests** - Fast feedback on code changes
2. **Integration Tests** - Verify component interactions
3. **E2E Tests** - Validate complete user workflows
4. **Coverage Report** - Track code coverage trends
5. **Performance Baseline** - Monitor performance regressions

Workflow file: `.github/workflows/test.yml`

**Key Steps:**
- Install dependencies
- Run all test suites
- Generate coverage report
- Upload to Codecov
- Fail build if coverage drops below threshold

## Related Documentation

- [Architecture](architecture.md) - System design and components
- [Design](design.md) - Visual design system
- [API Reference](api.md) - WebSocket and REST API
- [Installation](installation.md) - Setup instructions
- [ML Pipeline](ml-pipeline.md) - Content moderation details
- [README](../README.md) - Project overview

*Testing Strategy for DoodleParty v1.0*
