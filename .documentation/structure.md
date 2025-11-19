# Project Structure

## Directory Map

```
DoodleParty/
├── AdminConfig.json        # Global game configuration
├── flake.nix               # Nix development environment definition
├── frontend/               # Frontend and Game Server
│   ├── canvas/             # (Legacy) Canvas component logic
│   ├── server/             # Main Node.js/Express Game Server
│   │   ├── express-server.cjs  # Server entry point
│   │   ├── canvas_public/      # Static assets for Game Clients
│   │   │   ├── doodleparty.html    # The "Sender" (Mobile drawing interface)
│   │   │   └── index.html          # The "Viewer" (Main display)
│   │   ├── src/                # React Admin UI source
│   │   └── vite.config.ts      # Vite configuration for Admin UI
│   └── user/               # (Legacy) User-facing interface components
├── ml/                     # Machine Learning Service (Content Moderation)
│   ├── socket_client/      # Python Socket.io client
│   │   └── ml_server.py    # ML Service entry point
│   ├── service/            # Core ML logic
│   ├── data_processing/    # Scripts for training data
│   └── requirements.txt    # Python dependencies
├── tests/                  # Integration and End-to-End tests (Python)
├── infra/                  # Infrastructure configurations
└── scripts/                # Utility scripts
```

## Key Dependencies

*   **Frontend/Server**: `express`, `socket.io`, `vite`, `react`.
*   **ML Service**: `python-socketio`, `tensorflow` (TensorFlow Lite), `numpy`, `pillow`.
