# DoodleParty Nix Usage Guide

**Purpose:** Instructions for using DoodleParty with Nix and NixOS, including model training and system integration.

**Status: Updated - Nov 2025**

## Table of Contents

### Quick Start
- [Quick Start with Nix Flakes](#quick-start-with-nix-flakes)
  - [Development Shell](#development-shell)
  - [Running Applications](#running-applications)
    - [Start web server](#start-web-server)
    - [Start ML service](#start-ml-service)
    - [Run training](#run-training)
    - [Run evaluation](#run-evaluation)
  - [Building the Package](#building-the-package)

### NixOS Module
- [NixOS Module](#nixos-module)
  - [Installation](#installation)
  - [Configuration Options](#configuration-options)
    - **`services.doodleparty.enable`** (services.doodleparty.enable)
    - **`services.doodleparty.port`** (services.doodleparty.port)
    - **`services.doodleparty.host`** (services.doodleparty.host)
    - **`services.doodleparty.modelPath`** (services.doodleparty.modelpath)
    - **`services.doodleparty.mlPort`** (services.doodleparty.mlport)
    - **`services.doodleparty.dataDir`** (services.doodleparty.datadir)
    - **`services.doodleparty.user`** (services.doodleparty.user)
    - **`services.doodleparty.group`** (services.doodleparty.group)
    - **`services.doodleparty.openFirewall`** (services.doodleparty.openfirewall)
    - **`services.doodleparty.enableML`** (services.doodleparty.enableml)
  - [Example Configurations](#example-configurations)
    - [Basic local development](#basic-local-development)
    - [Production deployment](#production-deployment)
    - [Event deployment (RPi4)](#event-deployment-rpi4)
  - [Service Management](#service-management)
    - [Start the service](#start-the-service)
    - [Check status](#check-status)
    - [View logs](#view-logs)
    - [Restart the service](#restart-the-service)
    - [Stop the service](#stop-the-service)

### Development Workflow
- [Development Workflow](#development-workflow)
  - [1. Enter Development Shell](#1-enter-development-shell)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Start Web Server](#3-start-web-server)
  - [4. Start ML Service](#4-start-ml-service)
  - [5. Run Tests](#5-run-tests)
  - [6. Format Code](#6-format-code)

### Technical Details
- [Flake Structure](#flake-structure)
  - [Packages](#packages)
  - [Development Shells](#development-shells)
  - [Apps](#apps)
- [Environment Variables](#environment-variables)
- [Advanced Usage](#advanced-usage)
  - [Custom Development Environment](#custom-development-environment)
  - [Pinned Dependencies](#pinned-dependencies)

### Platform Specific
- [Raspberry Pi 4 with NixOS](#raspberry-pi-4-with-nixos)
  - [1. Install NixOS on RPi4](#1-install-nixos-on-rpi4)
  - [2. Add DoodleParty Module](#2-add-doodleparty-module)
  - [3. Update configuration.nix](#3-update-configurationnix)
  - [4. Rebuild and Switch](#4-rebuild-and-switch)
  - [5. Verify Services](#5-verify-services)

### Troubleshooting
- [Troubleshooting](#troubleshooting)
  - [Nix Flake Not Found](#nix-flake-not-found)
  - [TensorFlow Not Available](#tensorflow-not-available)
  - [Model Path Issues](#model-path-issues)
  - [Port Already in Use](#port-already-in-use)
  - [Memory Issues](#memory-issues)

### Resources
- [Related Documentation](#related-documentation)

## Quick Start with Nix Flakes

### Development Shell

Enter a development environment with all dependencies:

```bash
nix develop
```

This provides:
- Node.js 22 with npm
- Python 3.11 with TensorFlow and ML dependencies
- Development tools (eslint, prettier, pytest, black)
- Rich library for beautiful terminal output
- All required system libraries for training and inference

### Running Applications

**Start web server:**
```bash
nix run .#web
```

**Start ML service:**
```bash
nix run .#ml
```

**Run training:**
```bash
nix run .#train -- --epochs 50 --batch-size 32
```

**Run evaluation:**
```bash
nix run .#evaluate -- --model models/quickdraw_model.h5
```

### Building the Package

Build DoodleParty as a Nix package:

```bash
nix build
```

The result will be in `./result/`.

## NixOS Module

### Installation

#### Option 1: Using Flakes (Recommended)

Add DoodleParty to your system flake:

```nix
# /etc/nixos/flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    doodleparty.url = "path:/home/diatom/Documents/DoodleParty";
    # Or from git:
    # doodleparty.url = "github:yourusername/DoodleParty";
  };

  outputs = { self, nixpkgs, doodleparty, ... }: {
    nixosConfigurations.yourhostname = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        doodleparty.nixosModules.doodleparty
      ];
    };
  };
}
```

Then in your configuration.nix:

```nix
# configuration.nix
{ config, pkgs, ... }:

{
  services.doodleparty = {
    enable = true;
    port = 5000;
    openFirewall = true;
    
    # AI features
    enableAI = true;
    moderationThreshold = 0.85;
  };
}
```

#### Option 2: Without Flakes

Add to your NixOS configuration:

```nix
# configuration.nix
{ config, pkgs, ... }:

{
  imports = [
    /home/diatom/Documents/DoodleParty/module.nix
  ];

  services.doodleparty = {
    enable = true;
    port = 5000;
    openFirewall = true;
  };
}
```

#### Rebuild Your System

```bash
sudo nixos-rebuild switch
```

### Configuration Options

**Core Settings:**

**`services.doodleparty.enable`**
- Type: boolean
- Default: `false`
- Enable the DoodleParty service

**`services.doodleparty.port`**
- Type: port
- Default: `5000`
- Port for the Flask web server

**`services.doodleparty.host`**
- Type: string
- Default: `"0.0.0.0"`
- Host address to bind to

**`services.doodleparty.dataDir`**
- Type: path
- Default: `"/var/lib/doodleparty"`
- Directory for DoodleParty data and state

**`services.doodleparty.modelsDir`**
- Type: path
- Default: `"/var/lib/doodleparty/models"`
- Directory for ML models

**`services.doodleparty.user`**
- Type: string
- Default: `"doodleparty"`
- User account under which DoodleParty runs

**`services.doodleparty.group`**
- Type: string
- Default: `"doodleparty"`
- Group under which DoodleParty runs

**`services.doodleparty.openFirewall`**
- Type: boolean
- Default: `false`
- Whether to open the firewall for the configured port

**Performance Settings:**

**`services.doodleparty.workers`**
- Type: int
- Default: `4`
- Number of Gunicorn worker processes

**`services.doodleparty.maxConcurrentUsers`**
- Type: int
- Default: `100`
- Maximum number of concurrent users

**AI/ML Settings:**

**`services.doodleparty.enableAI`**
- Type: boolean
- Default: `true`
- Enable AI-powered content moderation

**`services.doodleparty.moderationThreshold`**
- Type: float
- Default: `0.85`
- Threshold for content moderation (0.0 - 1.0, lower = stricter)

**`services.doodleparty.enableLLM`**
- Type: boolean
- Default: `false`
- Enable LLM integration for prompts and narration

**`services.doodleparty.llmApiKeyFile`**
- Type: nullOr path
- Default: `null`
- File containing the API key for LLM service

**Logging & Monitoring:**

**`services.doodleparty.logLevel`**
- Type: enum ["DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL"]
- Default: `"INFO"`
- Logging level for the application

**`services.doodleparty.enableMetrics`**
- Type: boolean
- Default: `true`
- Enable Prometheus metrics endpoint

**`services.doodleparty.extraConfig`**
- Type: attrs
- Default: `{}`
- Extra configuration options as attribute set

### Example Configurations

**Basic local development:**
```nix
services.doodleparty = {
  enable = true;
  openFirewall = true;
};
```

**Production deployment:**
```nix
services.doodleparty = {
  enable = true;
  port = 5000;
  host = "0.0.0.0";
  openFirewall = true;
  
  # Performance
  workers = 8;
  maxConcurrentUsers = 500;
  
  # AI features
  enableAI = true;
  moderationThreshold = 0.85;
  
  # LLM integration
  enableLLM = true;
  llmApiKeyFile = "/run/secrets/doodleparty-llm-key";
  
  # Monitoring
  logLevel = "INFO";
  enableMetrics = true;
  
  # Custom config
  extraConfig = {
    CANVAS_WIDTH = 1920;
    CANVAS_HEIGHT = 1080;
    SAVE_INTERVAL = 60;
  };
};
```

**Event deployment (RPi4):**
```nix
services.doodleparty = {
  enable = true;
  port = 5000;
  host = "0.0.0.0";
  openFirewall = true;
  
  # Optimize for RPi4 resources
  workers = 2;
  maxConcurrentUsers = 100;
  
  # Use INT8 quantized models
  extraConfig = {
    MODEL_TYPE = "tflite";
    USE_INT8_QUANTIZATION = "true";
    CANVAS_WIDTH = 1280;
    CANVAS_HEIGHT = 720;
    JPEG_QUALITY = 75;
  };
  
  # Disable LLM for offline events
  enableLLM = false;
};

# Performance tuning for RPi4
boot.kernelParams = [ "cma=256M" ];
```

**With Nginx reverse proxy:**
```nix
services.doodleparty = {
  enable = true;
  port = 5000;
  # Don't open firewall; only accessible via nginx
};

services.nginx = {
  enable = true;
  
  virtualHosts."doodleparty.example.com" = {
    serverName = "doodleparty.example.com";
    enableACME = true;
    forceSSL = true;
    # Module automatically configures proxy
  };
};

security.acme = {
  acceptTerms = true;
  defaults.email = "admin@example.com";
};
```

### Service Management

**Start the service:**
```bash
sudo systemctl start doodleparty
```

**Check status:**
```bash
sudo systemctl status doodleparty
```

**View logs:**
```bash
sudo journalctl -u doodleparty -f
```

**Restart the service:**
```bash
sudo systemctl restart doodleparty
```

**Stop the service:**
```bash
sudo systemctl stop doodleparty
```

## Development Workflow

### 1. Enter Development Shell

```bash
cd /path/to/DoodleParty
nix develop
```

### 2. Install Dependencies

```bash
npm install
pip install -r requirements.txt
```

### 3. Start Web Server

```bash
npm run dev
```

### 4. Start ML Service (in another terminal)

```bash
python src-py/web/app.py
```

### 5. Run Tests

```bash
npm test
pytest tests/
```

### 6. Format Code

```bash
npm run format
black src-py/ scripts/
```

## Flake Structure

The flake provides:

**Packages:**
- `default` - DoodleParty web application
- `ml` - ML inference service
- `cli` - Command-line tools

**Development Shells:**
- `default` - Full development environment with Node.js and Python

**Apps:**
- `web` - Start web server
- `ml` - Start ML service
- `train` - Run training script
- `evaluate` - Run evaluation script

## Environment Variables

Create `.env` file in project root:

```bash
# Node.js
NODE_ENV=development
PORT=3000
ML_SERVICE_URL=http://localhost:5001

# ML Model
MODEL_PATH=models/quickdraw_model_int8.tflite
IMAGE_SIZE=28
THRESHOLD=0.5

```

## Troubleshooting

### Nix Flake Not Found

If `nix develop` fails with "flake not found":

```bash
# Ensure flake.nix exists in project root
ls -l flake.nix

# Update flake lock file
nix flake update

# Try again
nix develop
```

### TensorFlow Not Available

If TensorFlow is not available in the Nix shell:

```bash
# Check Python packages
python -c "import tensorflow; print(tensorflow.__version__)"

# If missing, ensure flake.nix includes tensorflow in propagatedBuildInputs
# Rebuild shell
nix develop --rebuild
```

### Model Path Issues

Ensure the model path is accessible:

```bash
# Check file exists
ls -l /var/lib/doodleparty/models/quickdraw_model_int8.tflite

# Check permissions
sudo chown doodleparty:doodleparty /var/lib/doodleparty/models/

# Verify path in configuration
grep modelPath /etc/nixos/configuration.nix
```

### Port Already in Use

Change the port in your configuration:

```nix
services.doodleparty.port = 3001;
```

Or kill the process using the port:

```bash
lsof -i :3000
kill -9 <PID>
```

### Memory Issues

If Nix build runs out of memory:

```bash
# Increase available memory
# Or use limited parallelism
nix build --max-jobs 1
```

## Raspberry Pi 4 with NixOS

### 1. Install NixOS on RPi4

Download NixOS for Raspberry Pi 4 from https://hydra.nixos.org/

### 2. Add DoodleParty Module

```bash
git clone https://github.com/yourusername/doodleparty.git /etc/nixos/doodleparty
```

### 3. Update configuration.nix

```nix
{ config, pkgs, ... }:

{
  imports = [
    /etc/nixos/doodleparty/module.nix
  ];

  services.doodleparty = {
    enable = true;
    port = 3000;
    host = "0.0.0.0";
    modelPath = "/var/lib/doodleparty/models/quickdraw_model_int8.tflite";
    openFirewall = true;
  };

  # Performance tuning for RPi4
  boot.kernelParams = [ "cma=256M" ];
  
  # Disable unnecessary services
  services.avahi.enable = false;
  services.bluetooth.enable = false;
}
```

### 4. Rebuild and Switch

```bash
sudo nixos-rebuild switch
```

### 5. Verify Services

```bash
sudo systemctl status doodleparty
sudo systemctl status doodleparty-ml
```

## Advanced Usage

### Custom Development Environment

Create a custom `shell.nix`:

```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_18
    python311
    nodePackages.typescript
    nodePackages.eslint
  ];

  shellHook = ''
    export PATH=$PWD/node_modules/.bin:$PATH
    echo "DoodleParty development environment loaded"
  '';
}
```

### Pinned Dependencies

Pin specific package versions in `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  # ...
}
```

## Related Documentation

- [Installation](installation.md) - Standard installation
- [Architecture](architecture.md) - System design
- [Development Roadmap](roadmap.md) - Planned features
- [README](../README.md) - Project overview
- [Testing Strategy](testing.md) - Testing approach and implementation

*Nix usage guide for DoodleParty v1.0*
