# DoodleParty

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

**Real-time collaborative drawing canvas with ML-powered content moderation.**

DoodleParty is a system for shared drawing experiences where multiple users can draw on a canvas simultaneously. It features a dedicated Machine Learning service that monitors the canvas in real-time to detect and censor inappropriate content (NSFW).

It is designed for "party" or "installation" settings where users join via mobile devices ("Senders") and their drawings appear on a main display ("Viewer"). There is no waiting room; users join and draw immediately.

> [!NOTE]
> 🏆 **<span style="color: #FFD700;">2nd Place Winner - European UniHack 2025, Culture & Entertainment Track</span>**

## Documentation Index

### Operations & Deployment
* **[Installation Guide](.documentation/installation.md)**
    Prerequisites (Node.js, Python, Nix), local environment setup, and startup instructions.
* **[Configuration Reference](.documentation/configuration.md)**
    `AdminConfig.json` settings, environment variables, and game modes.

### System Internals
* **[Architecture Overview](.documentation/architecture.md)**
    High-level design, data flow between Node.js server and ML Python service.
* **[Project Structure](.documentation/structure.md)**
    Directory map and component organization.
* **[API Reference](.documentation/api.md)**
    REST endpoints and Socket.io event protocols.

### Development Standards
* **[Testing Strategy](.documentation/testing.md)**
    Unit (Vitest) and Integration (Pytest) testing workflows.
* **[Design System](.documentation/design.md)**
    UI/UX guidelines and asset management.
* **[Roadmap](.documentation/roadmap.md)**
    Future plans and technical debt.

## Quick References

* **Server Entry Point:** `frontend/server/express-server.cjs`
* **ML Entry Point:** `ml/socket_client/ml_server.py`
* **Config:** `AdminConfig.json`
* **Run Server:** `npm run serve` (in `frontend/server`)

## License

MIT License - see [LICENSE](./LICENSE).
