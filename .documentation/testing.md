# Testing Strategy

DoodleParty uses a split testing strategy for Frontend/Server and Backend/Integration.

## Frontend & Server Tests

The Node.js server and React components are tested using **Vitest**.

### Running Tests
Navigate to `frontend/server`:

```bash
npm test
```

This runs unit tests for:
*   React components (`src/**/*.test.tsx`)
*   Server utility functions
*   Shared logic

## Integration Tests

End-to-end and integration tests are written in Python using **Pytest**, located in the `tests/` directory at the root.

### Prerequisites
Ensure your Python environment is active and dependencies are installed.

### Running Tests
From the project root:

```bash
pytest
```

### Test Scope
*   **Smoke Tests** (`test_smoke.py`): Verifies basic system health and configuration loading.
*   **Integration Tests**: specific interactions between the Game Server and ML Service.
