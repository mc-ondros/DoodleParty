# Installation Guide

## Prerequisites

*   **Node.js**: v20 or higher.
*   **Python**: v3.10 or higher.
*   **Nix** (Optional): For reproducible development environments.

## Setup

### 1. Using Nix (Recommended)

If you have Nix installed with flakes enabled:

```bash
nix develop
```

This will drop you into a shell with all dependencies (Node.js, Python) available.

### 2. Manual Setup

#### Backend (ML Service)

1.  Navigate to the project root.
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
    Or for ML service only:
    ```bash
    pip install -r ml/requirements.txt
    ```

#### Frontend / Game Server

1.  Navigate to the server directory:
    ```bash
    cd frontend/server
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```

## Running the Application

### Quick Start (Recommended)

From the project root, use the provided startup script:

```bash
./scripts/dev/start_doodleparty.sh
```

This will start both the Game Server and ML Service automatically.

### Manual Startup

#### 1. Start the Game Server

In `frontend/server`:

```bash
# Development mode (Vite)
npm run dev

# Production Serve
npm run serve
```

The server typically runs on `http://localhost:3000`.

#### 2. Start the ML Service

The ML service connects to the game server via Socket.io. Ensure the Game Server is running first.

From the project root:

```bash
python ml/socket_client/ml_server.py
```

## Verification

1.  Open `http://localhost:3000/doodleparty` in your browser (Sender interface).
2.  Open `http://localhost:3000/` in another window or display (Viewer interface).
3.  Draw something on the canvas in the Sender.
4.  Verify strokes appear in real-time on the Viewer.
5.  Check the ML service console logs to ensure it is receiving detection requests and performing inference.
6.  If a drawing is flagged as inappropriate, verify it is erased from both Sender and Viewer.
