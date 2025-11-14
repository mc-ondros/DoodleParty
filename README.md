# DoodleParty

**Real-time collaborative drawing platform with AI-powered content moderation**

Transform passive audiences into active participants through collaborative art. DoodleParty brings communities together at events, concerts, festivals, and public gatherings—creating shared cultural moments through creative expression.

**Status: Experimental** - This project is under active development. Documentation may not always match code.

[![Node.js](https://img.shields.io/badge/node.js-18%2B-green.svg)](https://nodejs.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
|[![License](https://img.shields.io/badge/license-GPL-blue.svg)](./LICENSE)

## Core Technologies

DoodleParty uses a modern tech stack optimized for real-time collaboration and lightweight RPi4 deployment:

- **Backend:** Node.js + Express + Socket.io (real-time communication)
- **Frontend:** React with real-time canvas engine (optimized for low-end devices)
- **ML:** TensorFlow Lite INT8 binary classifier (5MB, <50ms inference on RPi4)
- **LLM:** DigitalOcean AI integration for prompts and narration
- **Deployment:** Raspberry Pi 4 (4GB RAM, CPU-only) or DigitalOcean Kubernetes

|*See [Architecture Overview](.documentation/architecture.md) for complete system design, deployment architectures, and performance targets.*

## Features

- **Real-time Collaborative Drawing** - 100+ concurrent users with <50ms ML inference
- **Multiple Game Modes** - Classic Canvas, Speed Sketch, Guess The Doodle, Battle Royale, Story Canvas
- **AI-Powered Moderation** - Custom TFLite binary classifier with shape-based, tile-based, and region-based detection
- **LLM Integration** - Dynamic prompt generation and real-time story narration via DigitalOcean AI
- **Cross-Platform** - Native mobile apps, web browser, and projector display
- **Offline Capable** - Raspberry Pi 4 deployment for events with no internet required
- **Gamification** - Achievement system, progression levels, crew teams, and leaderboards

## Documentation

### Getting Started
|- [Installation & Setup](.documentation/installation.md) - Complete setup instructions
|- [Nix Usage Guide](.documentation/nix-usage.md) - Nix-specific deployment

### Understanding the System
|- [Architecture Overview](.documentation/architecture.md) - System design and components
|- [Project Structure](.documentation/structure.md) - Code organization and file layout
|- [ML Pipeline](.documentation/ml-pipeline.md) - Content moderation implementation

### Development & APIs
|- [API Reference](.documentation/api.md) - WebSocket and REST API documentation
|- [Code Style Guide](./STYLE_GUIDE.md) - Development standards and conventions
|- [Testing Strategy](.documentation/testing.md) - Testing approach and implementation
|- [Development Roadmap](.documentation/roadmap.md) - Future plans and features

## Deployment Options

**Raspberry Pi 4 (On-Premises)**
- Self-contained hardware for events
- Supports 100 concurrent users
- Works offline with complete data privacy
- Perfect for festivals, conferences, and community events

**Cloud Hosting (Managed SaaS)**
- Fully managed, no setup required
- Scales from 100 to 2,000+ concurrent users
- Full AI features and game modes
- Tiered pricing based on usage

|**System Requirements:** See [Installation Guide](.documentation/installation.md#prerequisites)
|
|**Core Technologies:** See [Architecture Overview](.documentation/architecture.md#core-technologies)

## Table of Contents

### Overview
- [Features](#features)
- [Core Technologies](#core-technologies)
- [Cultural Impact](#cultural-impact)

### Getting Started
- [Quick Start](#quick-start)
  - [With Nix](#with-nix)
  - [Without Nix](#without-nix)
- [System Requirements](#system-requirements)
  - [Raspberry Pi 4 Deployment](#raspberry-pi-4-deployment)
  - [Development Environment](#development-environment)

### Deployment Options
- [Raspberry Pi 4 (On-Premises)](#raspberry-pi-4-on-premises)
- [Cloud Hosting (Managed SaaS)](#cloud-hosting-managed-saas)

### Resources
- [Documentation](#documentation)
  - [Getting Started](#getting-started)
  - [Understanding the System](#understanding-the-system)
  - [Development & APIs](#development--apis)
- [Contributing](#contributing)
- [License](#license)

## Cultural Impact

DoodleParty democratizes creativity, preserves event memories, bridges generations, celebrates diversity, and encourages playfulness. It becomes a digital campfire where communities create, laugh, and connect—leaving behind a visual record of their shared experience.

## Contributing

|We welcome contributions! See [STYLE_GUIDE.md](./STYLE_GUIDE.md) for code standards, [Development Roadmap](.documentation/roadmap.md) for planned features, and [Testing Strategy](.documentation/testing.md) for our comprehensive testing approach.

## License

|GNU General Public License - see [LICENSE](./LICENSE) for details.
